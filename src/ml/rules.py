"""Derive readable business rules from the LightGBM model with a surrogate tree.

A shallow decision tree is fitted to reproduce the *primary model's* risk bands,
not the raw target. The result is a handful of if-then rules a credit policy team
can read, plus a fidelity score saying how faithfully those rules stand in for
the real model.

Two variants are produced at the same depth so the tradeoff is explicit rather
than decided silently: ``standard`` maximises overall band agreement, while
``risk_focused`` uses balanced class weights to catch far more of the High band
at the cost of overall accuracy.

Run with ``python -m src.ml.rules``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.data.preprocessor import RETIREE_FLAG, RETIREE_PLACEHOLDER
from src.ml.predict import load_model_artifacts, risk_band
from src.ml.train import RANDOM_STATE, load_training_frame, split_data
from src.utils.config import PROJECT_ROOT, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUSINESS_RULES_PATH = PROJECT_ROOT / "models" / "business_rules.json"

MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 500
BAND_ORDER = ["High", "Medium", "Low"]

# Both variants share depth and leaf size; only the class weighting differs.
VARIANTS: dict[str, dict[str, Any]] = {
    "standard": {
        "class_weight": None,
        "label": "Standard (unweighted)",
        "description": (
            "Maximises overall agreement with the primary model's bands. Best "
            "when the rules are used as a general-purpose approximation."
        ),
    },
    "risk_focused": {
        "class_weight": "balanced",
        "label": "Risk-focused (balanced class weights)",
        "description": (
            "Reweights the bands so the minority High band is not sacrificed. "
            "Catches far more high-risk applicants at the cost of overall "
            "agreement. Best when a missed default costs more than a false alarm."
        ),
    },
}

# Raw, human-meaningful inputs. The scaled 178-column model matrix would give
# thresholds like "EXT_SOURCE_3 <= -0.42", which is useless in a policy document.
SURROGATE_FEATURES: dict[str, str] = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "age_years": "Age",
    "years_employed": "Time in current job",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Annual repayment",
    "AMT_INCOME_TOTAL": "Annual income",
    "credit_to_income": "Credit-to-income multiple",
    "annuity_to_income": "Annuity-to-income ratio",
    "bureau_loan_count": "Bureau loans on file",
    "bureau_overdue_count": "Prior overdue loans",
    "previous_application_count": "Previous applications",
    "previous_approval_rate": "Previous approval rate",
    RETIREE_FLAG: "No employment record (retired)",
}

CURRENCY = {"AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"}
COUNTS = {
    "bureau_loan_count",
    "bureau_overdue_count",
    "previous_application_count",
}


def build_surrogate_frame(master: pd.DataFrame) -> pd.DataFrame:
    """Assemble the interpretable feature frame the surrogate is fitted on."""
    frame = pd.DataFrame(index=master.index)

    for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        frame[column] = master[column]

    frame["age_years"] = -master["DAYS_BIRTH"] / 365.25

    employed = master["DAYS_EMPLOYED"].mask(
        master["DAYS_EMPLOYED"].eq(RETIREE_PLACEHOLDER)
    )
    frame["years_employed"] = -employed / 365.25
    frame[RETIREE_FLAG] = (
        master["DAYS_EMPLOYED"].eq(RETIREE_PLACEHOLDER).astype("int8")
    )

    frame["AMT_CREDIT"] = master["AMT_CREDIT"]
    frame["AMT_ANNUITY"] = master["AMT_ANNUITY"]
    frame["AMT_INCOME_TOTAL"] = master["AMT_INCOME_TOTAL"]

    income = master["AMT_INCOME_TOTAL"].where(master["AMT_INCOME_TOTAL"] > 0)
    frame["credit_to_income"] = master["AMT_CREDIT"] / income
    frame["annuity_to_income"] = master["AMT_ANNUITY"] / income

    for column in (
        "bureau_loan_count",
        "bureau_overdue_count",
        "previous_application_count",
        "previous_approval_rate",
    ):
        frame[column] = master[column]

    return frame[list(SURROGATE_FEATURES)]


def primary_bands(master: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Risk bands and probabilities from the trained LightGBM model."""
    artifacts = load_model_artifacts()
    features = artifacts.preprocessor.transform(master).to_numpy(dtype="float32")
    probabilities = artifacts.model.predict_proba(features)[:, 1]
    bands = np.array([risk_band(float(value)) for value in probabilities])
    return bands, probabilities


def format_threshold(feature: str, threshold: float) -> str:
    """Render a split threshold in the feature's natural units."""
    if feature in CURRENCY:
        return f"{threshold:,.0f}"
    if feature in COUNTS:
        # Integer counts split on halves; state the integer a person would use.
        return f"{int(np.floor(threshold))}"
    if feature == RETIREE_FLAG:
        return "yes" if threshold >= 0.5 else "no"
    if feature in {"age_years", "years_employed"}:
        return f"{threshold:.1f} years"
    return f"{threshold:.3f}"


def render_conditions(conditions: Sequence[tuple[str, str, float]]) -> list[str]:
    """Collapse a root-to-leaf path into one readable clause per feature."""
    bounds: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for feature, operator, threshold in conditions:
        if feature not in bounds:
            bounds[feature] = {}
            order.append(feature)
        key = "upper" if operator == "<=" else "lower"
        current = bounds[feature].get(key)
        if current is None:
            bounds[feature][key] = threshold
        else:
            bounds[feature][key] = (
                min(current, threshold) if key == "upper" else max(current, threshold)
            )

    clauses = []
    for feature in order:
        label = SURROGATE_FEATURES[feature]
        lower = bounds[feature].get("lower")
        upper = bounds[feature].get("upper")

        if feature == RETIREE_FLAG:
            clauses.append(f"{label} is {'yes' if lower is not None else 'no'}")
            continue

        if lower is not None and upper is not None:
            clauses.append(
                f"{label} is between {format_threshold(feature, lower)} and "
                f"{format_threshold(feature, upper)}"
            )
        elif upper is not None:
            clauses.append(f"{label} <= {format_threshold(feature, upper)}")
        else:
            clauses.append(f"{label} > {format_threshold(feature, lower)}")
    return clauses


def fit_surrogate(
    X: pd.DataFrame,
    bands: np.ndarray,
    max_depth: int = MAX_DEPTH,
    class_weight: str | None = None,
    medians: pd.Series | None = None,
) -> tuple[DecisionTreeClassifier, pd.Series]:
    """Fit the shallow tree that mimics the primary model's banding."""
    if medians is None:
        medians = X.median(numeric_only=True)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    tree.fit(X.fillna(medians), bands)
    logger.info(
        "Fitted depth-%d surrogate (class_weight=%s) with %d leaves",
        max_depth,
        class_weight,
        tree.get_n_leaves(),
    )
    return tree, medians


def extract_rules(
    tree: DecisionTreeClassifier,
    X: pd.DataFrame,
    y_true: np.ndarray,
    medians: pd.Series,
    probabilities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Turn each leaf of the surrogate into an if-then rule with its statistics."""
    feature_names = list(X.columns)
    structure = tree.tree_
    leaves: list[tuple[int, list[tuple[str, str, float]]]] = []

    def walk(node: int, conditions: list[tuple[str, str, float]]) -> None:
        if structure.feature[node] == _tree.TREE_UNDEFINED:
            leaves.append((node, conditions))
            return
        feature = feature_names[structure.feature[node]]
        threshold = float(structure.threshold[node])
        walk(structure.children_left[node], conditions + [(feature, "<=", threshold)])
        walk(structure.children_right[node], conditions + [(feature, ">", threshold)])

    walk(0, [])

    filled = X.fillna(medians)
    leaf_ids = tree.apply(filled)
    total = len(filled)

    rules = []
    for node, conditions in leaves:
        mask = leaf_ids == node
        covered = int(mask.sum())
        if covered == 0:
            continue

        clauses = render_conditions(conditions)
        predicted = str(tree.classes_[np.argmax(structure.value[node][0])])
        rule = {
            "rule_id": len(rules) + 1,
            "conditions": clauses,
            "rule": f"IF {' AND '.join(clauses)} THEN risk band = {predicted}",
            "predicted_band": predicted,
            "applicants_covered": covered,
            "coverage_pct": round(covered / total * 100, 2),
            "actual_default_rate_pct": round(float(y_true[mask].mean() * 100), 2),
        }
        if probabilities is not None:
            rule["mean_model_probability"] = round(float(probabilities[mask].mean()), 4)
        rules.append(rule)

    rules.sort(key=lambda item: (-item["actual_default_rate_pct"], -item["applicants_covered"]))
    for index, rule in enumerate(rules, start=1):
        rule["rule_id"] = index
    return rules


def get_surrogate_fidelity(
    tree: DecisionTreeClassifier,
    X: pd.DataFrame,
    bands: np.ndarray,
    medians: pd.Series,
) -> dict[str, Any]:
    """Agreement between the surrogate's bands and the primary model's bands."""
    predicted = tree.predict(X.fillna(medians))
    agreement = float((predicted == bands).mean() * 100)

    per_band = {}
    for name in BAND_ORDER:
        mask = bands == name
        per_band[name] = {
            "applicants": int(mask.sum()),
            "agreement_pct": (
                round(float((predicted[mask] == name).mean() * 100), 2)
                if mask.any()
                else None
            ),
        }

    return {
        "overall_agreement_pct": round(agreement, 2),
        "by_primary_band": per_band,
        "surrogate_band_distribution": {
            name: int((predicted == name).sum()) for name in BAND_ORDER
        },
    }


def build_variant(
    name: str,
    surrogate_train: pd.DataFrame,
    train_bands: np.ndarray,
    train_probabilities: np.ndarray,
    y_train: np.ndarray,
    surrogate_test: pd.DataFrame,
    test_bands: np.ndarray,
    medians: pd.Series,
) -> dict[str, Any]:
    """Fit one surrogate variant and package its rules and fidelity."""
    config = VARIANTS[name]
    tree, _ = fit_surrogate(
        surrogate_train,
        train_bands,
        class_weight=config["class_weight"],
        medians=medians,
    )

    return {
        "label": config["label"],
        "description": config["description"],
        "config": {
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": config["class_weight"],
            "n_leaves": int(tree.get_n_leaves()),
        },
        "fidelity": {
            "train": get_surrogate_fidelity(
                tree, surrogate_train, train_bands, medians
            ),
            "holdout": get_surrogate_fidelity(
                tree, surrogate_test, test_bands, medians
            ),
        },
        "rules": extract_rules(
            tree, surrogate_train, y_train, medians, train_probabilities
        ),
    }


def print_variant(name: str, variant: dict[str, Any], train_rows: int) -> None:
    fidelity = variant["fidelity"]
    print("\n" + "=" * 78)
    print(f"VARIANT '{name}' - {variant['label']}")
    print("=" * 78)
    print(
        f"Fidelity vs primary bands: train {fidelity['train']['overall_agreement_pct']:.2f}%"
        f"   holdout {fidelity['holdout']['overall_agreement_pct']:.2f}%"
    )
    print("Agreement by primary band (holdout):")
    for band, block in fidelity["holdout"]["by_primary_band"].items():
        print(
            f"  {band:<7} {block['applicants']:>7,} applicants   "
            f"{block['agreement_pct']:.2f}%"
        )

    rules = variant["rules"]
    print(f"\n{len(rules)} rules (training split, {train_rows:,} applicants):\n")
    for rule in rules:
        print(f"Rule {rule['rule_id']}  ->  {rule['predicted_band']} risk")
        for clause in rule["conditions"]:
            print(f"    - {clause}")
        print(
            f"    covers {rule['applicants_covered']:,} applicants "
            f"({rule['coverage_pct']:.2f}%)   "
            f"actual default rate {rule['actual_default_rate_pct']:.2f}%\n"
        )


def main() -> None:
    master, target = load_training_frame()
    X_train, X_test, y_train, y_test = split_data(master, target)

    logger.info("Scoring both splits with the primary model...")
    train_bands, train_probabilities = primary_bands(X_train)
    test_bands, _ = primary_bands(X_test)

    surrogate_train = build_surrogate_frame(X_train)
    surrogate_test = build_surrogate_frame(X_test)
    medians = surrogate_train.median(numeric_only=True)

    variants = {
        name: build_variant(
            name,
            surrogate_train,
            train_bands,
            train_probabilities,
            y_train.to_numpy(),
            surrogate_test,
            test_bands,
            medians,
        )
        for name in VARIANTS
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {"low": RISK_THRESHOLD_LOW, "high": RISK_THRESHOLD_HIGH},
        "variants": list(VARIANTS),
        "default_variant": "standard",
        "surrogate": {
            "fitted_on": "primary model risk bands, training split",
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
            "features": SURROGATE_FEATURES,
            "missing_value_medians": {
                key: round(float(value), 6) for key, value in medians.items()
            },
        },
        "comparison": {
            name: {
                "label": variant["label"],
                "n_leaves": variant["config"]["n_leaves"],
                "holdout_agreement_pct": variant["fidelity"]["holdout"][
                    "overall_agreement_pct"
                ],
                "holdout_agreement_by_band_pct": {
                    band: block["agreement_pct"]
                    for band, block in variant["fidelity"]["holdout"][
                        "by_primary_band"
                    ].items()
                },
            }
            for name, variant in variants.items()
        },
        **variants,
    }

    BUSINESS_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUSINESS_RULES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, variant in variants.items():
        print_variant(name, variant, len(X_train))

    print("\n" + "=" * 78)
    print("TRADEOFF SUMMARY (holdout agreement %)")
    print("=" * 78)
    print(f"{'variant':<14}{'leaves':>8}{'overall':>10}{'High':>9}{'Medium':>9}{'Low':>9}")
    for name, block in payload["comparison"].items():
        by_band = block["holdout_agreement_by_band_pct"]
        print(
            f"{name:<14}{block['n_leaves']:>8}{block['holdout_agreement_pct']:>10.2f}"
            f"{by_band['High']:>9.2f}{by_band['Medium']:>9.2f}{by_band['Low']:>9.2f}"
        )

    logger.info("Saved %s", BUSINESS_RULES_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
