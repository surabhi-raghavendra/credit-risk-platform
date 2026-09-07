"""SHAP explanations for the LightGBM default-risk model.

Individual predictions are translated into plain English so the reasoning can be
shown to a credit officer rather than only to a data scientist.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd
import shap

from src.ml.predict import ModelArtifacts, load_model_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_CONTRIBUTORS = 5
GLOBAL_TOP_FEATURES = 15

# Plain-English names for the features that dominate the importance ranking.
FEATURE_LABEL_MAP: dict[str, str] = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "DAYS_BIRTH": "Applicant age",
    "DAYS_EMPLOYED": "Time in current employment",
    "DAYS_REGISTRATION": "Time since registration was last changed",
    "DAYS_ID_PUBLISH": "Time since identity document was issued",
    "DAYS_LAST_PHONE_CHANGE": "Time since the phone number last changed",
    "AMT_CREDIT": "Loan amount requested",
    "AMT_ANNUITY": "Annual repayment amount",
    "AMT_GOODS_PRICE": "Price of the goods being financed",
    "AMT_INCOME_TOTAL": "Declared annual income",
    "REGION_POPULATION_RELATIVE": "Population density of the applicant's region",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Household size",
    "REGION_RATING_CLIENT": "Internal risk rating of the region",
    "REGION_RATING_CLIENT_W_CITY": "Internal risk rating of the region and city",
    "bureau_loan_count": "Number of loans on file at the credit bureau",
    "bureau_avg_credit_sum": "Average size of past credit-bureau loans",
    "bureau_overdue_count": "Number of prior loans currently overdue",
    "previous_application_count": "Number of previous applications to this lender",
    "previous_avg_amount": "Average amount of previous applications",
    "previous_approval_rate": "Share of previous applications that were approved",
    "is_retiree_placeholder": "No current employment record (typically retired)",
}

# Human-readable base names for the one-hot encoded categoricals.
CATEGORY_LABEL_MAP: dict[str, str] = {
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_CONTRACT_TYPE": "Contract type",
    "OCCUPATION_TYPE": "Occupation",
    "CODE_GENDER": "Gender",
    "NAME_TYPE_SUITE": "Accompanied by",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
    "WEEKDAY_APPR_PROCESS_START": "Weekday the application started",
}

# Features stored as negative day counts that read better as years.
DAYS_AS_YEARS = {
    "DAYS_BIRTH": "years old",
    "DAYS_EMPLOYED": "years in the job",
    "DAYS_REGISTRATION": "years since registration",
    "DAYS_ID_PUBLISH": "years since ID issue",
    "DAYS_LAST_PHONE_CHANGE": "years since phone change",
}
CURRENCY_FEATURES = {
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "bureau_avg_credit_sum",
    "previous_avg_amount",
}


@lru_cache(maxsize=1)
def get_shap_explainer() -> shap.TreeExplainer:
    """Return a cached TreeExplainer built on the trained LightGBM model."""
    artifacts = load_model_artifacts()
    explainer = shap.TreeExplainer(artifacts.model)
    logger.info("Built SHAP TreeExplainer on the trained LightGBM model")
    return explainer


def categorical_base(feature: str, artifacts: ModelArtifacts) -> str | None:
    """Return the source column of a one-hot feature, or None if it is not one."""
    for base in getattr(artifacts.preprocessor, "categorical_columns_", []):
        if feature.startswith(f"{base}_"):
            return base
    return None


def category_label(base: str) -> str:
    return CATEGORY_LABEL_MAP.get(base, base.replace("_", " ").capitalize())


def describe_feature(feature: str, artifacts: ModelArtifacts | None = None) -> str:
    """Translate an encoded feature name into a readable label."""
    if feature in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[feature]

    if feature.startswith("FLAG_DOCUMENT_"):
        return f"Supporting document {feature.rsplit('_', 1)[-1]}"

    artifacts = artifacts or load_model_artifacts()
    base = categorical_base(feature, artifacts)
    if base is not None:
        return f"{category_label(base)} is '{feature[len(base) + 1:]}'"

    return feature.replace("_", " ").capitalize()


def format_value(feature: str, value: Any) -> str:
    """Render a raw feature value the way a person would say it."""
    if value is None:
        return "not provided"
    if isinstance(value, str):
        return "yes" if value == "Y" else "no" if value == "N" else value

    # np.int64 is not a subclass of int, so numpy scalars need naming explicitly.
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)

    numeric = float(value)
    if np.isnan(numeric):
        return "not provided"

    if feature in DAYS_AS_YEARS:
        return f"{abs(numeric) / 365.25:.1f} {DAYS_AS_YEARS[feature]}"
    if feature in CURRENCY_FEATURES:
        return f"{numeric:,.0f}"
    if feature == "is_retiree_placeholder" or feature.startswith("FLAG_"):
        return "yes" if numeric >= 0.5 else "no"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.3f}"


def _normalise_shap(values: Any) -> np.ndarray:
    """Reduce SHAP output to a 2-D array for the positive class."""
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def _shap_values(explainer: shap.TreeExplainer, matrix: np.ndarray) -> np.ndarray:
    """SHAP values for the positive class, as a 2-D array."""
    with warnings.catch_warnings():
        # SHAP warns that LightGBM binary output is now a list of arrays;
        # _normalise_shap already handles both shapes.
        warnings.filterwarnings(
            "ignore", message=".*list of ndarray.*", category=UserWarning
        )
        return _normalise_shap(explainer.shap_values(matrix))


def _normalise_base_value(explainer: shap.TreeExplainer) -> float:
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        return float(np.asarray(expected).ravel()[-1])
    return float(expected)


def _to_model_features(
    features: Any, artifacts: ModelArtifacts
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (model-space features, raw values) for one or more applicants.

    Accepts a raw applicant mapping, a raw DataFrame, or an already-preprocessed
    frame whose columns are the model feature names.
    """
    if isinstance(features, Mapping):
        features = pd.DataFrame([dict(features)])
    elif isinstance(features, pd.Series):
        features = features.to_frame().T

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "explain_prediction expects a mapping, Series, or DataFrame of applicant fields."
        )

    if list(features.columns) == artifacts.model_features:
        return features, features

    raw = features.reindex(columns=artifacts.raw_input_columns)
    return artifacts.preprocessor.transform(raw), raw


def explain_prediction(features: Any, top_n: int = TOP_CONTRIBUTORS) -> dict[str, Any]:
    """Explain a single prediction as its strongest plain-English drivers."""
    artifacts = load_model_artifacts()
    model_features, raw = _to_model_features(features, artifacts)
    if len(model_features) != 1:
        raise ValueError("explain_prediction handles one applicant at a time.")

    matrix = model_features.to_numpy(dtype="float32")
    explainer = get_shap_explainer()
    shap_values = _shap_values(explainer, matrix)[0]
    base_value = _normalise_base_value(explainer)

    probability = float(artifacts.model.predict_proba(matrix)[0, 1])
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]

    contributors = []
    for rank, index in enumerate(order, start=1):
        feature = artifacts.model_features[index]
        contribution = float(shap_values[index])

        # Prefer the raw pre-encoding value; it is what a person recognises. For
        # one-hot columns that means the applicant's actual category, so the text
        # reads "Contract type (Revolving loans)" rather than "... 'Cash loans' (0)".
        base = categorical_base(feature, artifacts)
        if base is not None and base in raw.columns:
            label = category_label(base)
            rendered = format_value(base, raw.iloc[0][base])
        else:
            label = describe_feature(feature, artifacts)
            source = raw if feature in raw.columns else model_features
            rendered = format_value(feature, source.iloc[0][feature])
        verb = "increases" if contribution > 0 else "decreases"
        contributors.append(
            {
                "rank": rank,
                "feature": feature,
                "label": label,
                "value": rendered,
                "shap_value": round(contribution, 6),
                "direction": verb,
                "explanation": f"{label} ({rendered}) {verb} the default risk.",
            }
        )

    return {
        "probability": round(probability, 6),
        "base_value_log_odds": round(base_value, 6),
        "top_contributors": contributors,
    }


def get_global_shap_summary(
    X_test_sample: pd.DataFrame, top_n: int = GLOBAL_TOP_FEATURES
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across a sample.

    Returns a frame of ``feature``, ``label``, and ``mean_abs_shap`` ready to
    plot as a horizontal bar chart in the UI.
    """
    artifacts = load_model_artifacts()
    model_features, _ = _to_model_features(X_test_sample, artifacts)

    explainer = get_shap_explainer()
    shap_values = _shap_values(
        explainer, model_features.to_numpy(dtype="float32")
    )
    logger.info(
        "Computed SHAP values for %d rows x %d features", *shap_values.shape
    )

    summary = (
        pd.DataFrame(
            {
                "feature": artifacts.model_features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    summary["label"] = [
        describe_feature(feature, artifacts) for feature in summary["feature"]
    ]
    summary["mean_abs_shap"] = summary["mean_abs_shap"].round(6)
    return summary[["feature", "label", "mean_abs_shap"]]
