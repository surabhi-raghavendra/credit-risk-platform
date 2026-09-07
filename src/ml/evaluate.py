"""Score the trained model on the held-out split and persist the metrics.

Run with ``python -m src.ml.evaluate``. Charts are deliberately omitted; the UI
reads the numbers straight out of models/eval_results.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.predict import load_model_artifacts, risk_band
from src.ml.train import (
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    load_training_frame,
    split_data,
)
from src.utils.config import PROJECT_ROOT, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVAL_RESULTS_PATH = PROJECT_ROOT / "models" / "eval_results.json"
DECISION_THRESHOLD = 0.50


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1, and the confusion matrix at one cut-off."""
    predictions = (probabilities >= threshold).astype(int)
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "confusion_matrix": {
            "true_negative": int(true_neg),
            "false_positive": int(false_pos),
            "false_negative": int(false_neg),
            "true_positive": int(true_pos),
        },
    }


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Sweep cut-offs and return the metrics at the F1-maximising one."""
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scored = [
        (f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0), threshold)
        for threshold in candidates
    ]
    _, best = max(scored)
    return threshold_metrics(y_true, probabilities, float(best))


def band_distribution(probabilities: np.ndarray, y_true: np.ndarray) -> dict:
    """Population share and realised default rate inside each configured band."""
    bands = np.array([risk_band(float(value)) for value in probabilities])
    summary = {}
    for name in ("Low", "Medium", "High"):
        mask = bands == name
        summary[name] = {
            "applicants": int(mask.sum()),
            "share_pct": round(float(mask.mean() * 100), 2),
            "actual_default_rate_pct": (
                round(float(y_true[mask].mean() * 100), 2) if mask.any() else None
            ),
        }
    return summary


def main() -> None:
    master, target = load_training_frame()
    _, X_test, _, y_test = split_data(master, target)
    logger.info("Evaluating on %d held-out rows", len(X_test))

    artifacts = load_model_artifacts()
    features = artifacts.preprocessor.transform(X_test).to_numpy(dtype="float32")
    probabilities = artifacts.model.predict_proba(features)[:, 1]
    y_true = y_test.to_numpy()

    at_default = threshold_metrics(y_true, probabilities, DECISION_THRESHOLD)
    at_best_f1 = best_f1_threshold(y_true, probabilities)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_rows": int(len(y_true)),
        "test_positive_rate_pct": round(float(y_true.mean() * 100), 4),
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS},
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 6),
        "at_default_threshold": at_default,
        "at_best_f1_threshold": at_best_f1,
        "risk_bands": {
            "thresholds": {"low": RISK_THRESHOLD_LOW, "high": RISK_THRESHOLD_HIGH},
            "distribution": band_distribution(probabilities, y_true),
        },
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nHold-out set: {results['test_rows']:,} rows, "
          f"{results['test_positive_rate_pct']:.2f}% positive")
    print(f"ROC-AUC: {results['roc_auc']:.5f}")
    print(f"PR-AUC : {results['pr_auc']:.5f}")
    for label, block in (
        (f"threshold {DECISION_THRESHOLD:.2f}", at_default),
        (f"best-F1 threshold {at_best_f1['threshold']:.2f}", at_best_f1),
    ):
        matrix = block["confusion_matrix"]
        print(
            f"\nAt {label}:"
            f"\n  precision {block['precision']:.4f}   recall {block['recall']:.4f}   "
            f"F1 {block['f1']:.4f}"
            f"\n  TN {matrix['true_negative']:,}   FP {matrix['false_positive']:,}   "
            f"FN {matrix['false_negative']:,}   TP {matrix['true_positive']:,}"
        )

    print("\nRisk band distribution:")
    for name, block in results["risk_bands"]["distribution"].items():
        print(
            f"  {name:<7} {block['applicants']:>7,} applicants "
            f"({block['share_pct']:>5.2f}%)   actual default rate "
            f"{block['actual_default_rate_pct']}%"
        )

    logger.info("Saved %s", EVAL_RESULTS_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
