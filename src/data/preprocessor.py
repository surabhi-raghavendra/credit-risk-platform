"""Feature grouping, data-quality reporting, and model preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


TARGET_COLUMN = "TARGET"
RETIREE_PLACEHOLDER = 365243
RETIREE_FLAG = "is_retiree_placeholder"


def categorize_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group model features into broad business-domain categories."""
    groups: dict[str, list[str]] = {
        "demographic": [],
        "financial": [],
        "credit_history": [],
        "behavioral": [],
    }

    demographic_terms = (
        "GENDER",
        "BIRTH",
        "CHILDREN",
        "FAM_MEMBERS",
        "FAMILY_STATUS",
        "EDUCATION",
        "OCCUPATION",
        "ORGANIZATION",
        "HOUSING",
        "REGION",
    )
    financial_terms = (
        "AMT_",
        "INCOME",
        "ANNUITY",
        "GOODS_PRICE",
        "OWN_CAR_AGE",
    )
    credit_terms = (
        "BUREAU_",
        "PREVIOUS_",
        "CREDIT",
        "EXT_SOURCE",
        "OVERDUE",
    )

    for column in df.columns:
        upper = column.upper()
        if upper == TARGET_COLUMN or upper.startswith("SK_ID"):
            continue
        if any(term in upper for term in credit_terms):
            groups["credit_history"].append(column)
        elif any(term in upper for term in financial_terms):
            groups["financial"].append(column)
        elif any(term in upper for term in demographic_terms):
            groups["demographic"].append(column)
        else:
            groups["behavioral"].append(column)

    return groups


def data_quality_report(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> dict[str, Any]:
    """Summarize missingness, data types, and binary-target imbalance."""
    columns = pd.DataFrame(
        {
            "missing_percentage": df.isna().mean().mul(100).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    columns.index.name = "column"

    target_distribution: dict[Any, int] = {}
    imbalance_ratio: float | None = None
    if target_column in df.columns:
        counts = df[target_column].dropna().value_counts()
        target_distribution = counts.to_dict()
        if len(counts) >= 2:
            imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])
        elif len(counts) == 1:
            imbalance_ratio = float("inf")

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "target_distribution": target_distribution,
        "target_imbalance_ratio": imbalance_ratio,
    }


class Preprocessor(BaseEstimator, TransformerMixin):
    """Prepare application features for scikit-learn estimators.

    Continuous numeric values are median-imputed and standardized.
    Low-cardinality categorical values are constant-imputed and one-hot encoded;
    high-cardinality categorical values are excluded to avoid an uncontrolled
    feature expansion. The retiree indicator remains binary.
    """

    def __init__(
        self,
        missing_threshold: float = 0.65,
        categorical_cardinality_threshold: int = 20,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.categorical_cardinality_threshold = categorical_cardinality_threshold

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Preprocessor expects a pandas DataFrame.")

    @staticmethod
    def _fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if "DAYS_EMPLOYED" in prepared.columns:
            placeholder_mask = prepared["DAYS_EMPLOYED"].eq(RETIREE_PLACEHOLDER)
            prepared[RETIREE_FLAG] = placeholder_mask.astype("int8")
            prepared.loc[placeholder_mask, "DAYS_EMPLOYED"] = np.nan
        elif RETIREE_FLAG not in prepared.columns:
            prepared[RETIREE_FLAG] = np.int8(0)
        return prepared

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._fix_days_employed(df)
        return prepared.drop(columns=self.drop_columns_, errors="ignore")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Preprocessor":
        """Learn exclusions, imputations, encoding, and scaling from training data."""
        self._validate_dataframe(X)
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1.")
        if self.categorical_cardinality_threshold < 1:
            raise ValueError("categorical_cardinality_threshold must be positive.")

        prepared = self._fix_days_employed(X)
        id_columns = [
            column
            for column in prepared.columns
            if column.upper().startswith("SK_ID") or column.upper() == TARGET_COLUMN
        ]
        sparse_columns = prepared.columns[
            prepared.isna().mean().gt(self.missing_threshold)
        ].tolist()
        self.drop_columns_ = sorted(set(id_columns + sparse_columns))
        prepared = prepared.drop(columns=self.drop_columns_, errors="ignore")

        numeric_columns = prepared.select_dtypes(include=np.number).columns.tolist()
        numeric_columns = [
            column for column in numeric_columns if column != RETIREE_FLAG
        ]
        categorical_columns = prepared.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
        self.categorical_columns_ = [
            column
            for column in categorical_columns
            if prepared[column].nunique(dropna=True)
            <= self.categorical_cardinality_threshold
        ]
        self.high_cardinality_columns_ = sorted(
            set(categorical_columns).difference(self.categorical_columns_)
        )
        self.numeric_columns_ = numeric_columns

        transformers = []
        if self.numeric_columns_:
            numeric_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, self.numeric_columns_))
        if self.categorical_columns_:
            categorical_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Missing"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, self.categorical_columns_)
            )
        if RETIREE_FLAG in prepared.columns:
            transformers.append(("retiree_flag", "passthrough", [RETIREE_FLAG]))
        if not transformers:
            raise ValueError("No usable numeric or low-cardinality features found.")

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.column_transformer_.fit(prepared, y)
        self.feature_names_out_ = self.column_transformer_.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing steps and return named model features."""
        check_is_fitted(self, ["column_transformer_", "feature_names_out_"])
        self._validate_dataframe(X)
        prepared = self._prepare(X)
        transformed = self.column_transformer_.transform(prepared)
        return pd.DataFrame(
            transformed,
            columns=self.feature_names_out_,
            index=X.index,
        )

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        """Return output feature names learned during fitting."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
