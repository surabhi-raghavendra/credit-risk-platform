"""Train the LightGBM default-risk model and persist its artifacts.

Run with ``python -m src.ml.train``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data.loader import build_master_dataset
from src.data.preprocessor import TARGET_COLUMN, Preprocessor
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


def load_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Return the master dataset alongside its binary target."""
    master = build_master_dataset()
    if TARGET_COLUMN not in master.columns:
        raise ValueError(f"Master dataset has no '{TARGET_COLUMN}' column.")
    return master, master[TARGET_COLUMN].astype(int)


def split_data(
    master: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split. Shared with evaluate.py so the holdout matches."""
    return train_test_split(
        master,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def compute_scale_pos_weight(target: pd.Series) -> float:
    """Negative-to-positive ratio used to counter the class imbalance."""
    positives = int(target.sum())
    if positives == 0:
        raise ValueError("Training target contains no positive class.")
    return float(len(target) - positives) / positives


def build_model(scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)


def main() -> None:
    master, target = load_training_frame()
    logger.info("Master dataset: %d rows, %d columns", *master.shape)

    X_train, X_test, y_train, y_test = split_data(master, target)
    logger.info(
        "Split -> train %d rows (%.2f%% positive), test %d rows (%.2f%% positive)",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )

    # Fitted on the training split only, so the holdout stays untouched.
    preprocessor = Preprocessor()
    train_features = preprocessor.fit_transform(X_train)
    feature_names = list(train_features.columns)
    logger.info(
        "Preprocessor produced %d features (dropped %d source columns)",
        len(feature_names),
        len(preprocessor.drop_columns_),
    )

    # LightGBM rejects feature names containing JSON metacharacters, and the
    # one-hot columns include values like "Spouse, partner". The names are kept
    # in feature_names.json instead and the matrix is passed positionally.
    train_matrix = train_features.to_numpy(dtype="float32")
    del train_features

    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info("scale_pos_weight = %.4f", scale_pos_weight)

    model = build_model(scale_pos_weight)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Running %d-fold stratified CV on ROC-AUC...", CV_FOLDS)
    cv_scores = cross_val_score(
        model, train_matrix, y_train, cv=folds, scoring="roc_auc"
    )

    print("\nCross-validated ROC-AUC (train split):")
    for index, score in enumerate(cv_scores, start=1):
        print(f"  fold {index}: {score:.5f}")
    print(f"  mean:   {cv_scores.mean():.5f}  (std {cv_scores.std():.5f})")

    logger.info("Fitting final model on the full training split...")
    model.fit(train_matrix, y_train)

    importances = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
    print("\nTop 15 features by LightGBM gain-split importance:")
    print(importances.head(15).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    dump(preprocessor, PREPROCESSOR_PATH)

    payload = {
        "model_features": feature_names,
        "n_model_features": len(feature_names),
        "raw_input_columns": [
            column for column in X_train.columns if column != TARGET_COLUMN
        ],
        "dropped_columns": preprocessor.drop_columns_,
        "high_cardinality_excluded": preprocessor.high_cardinality_columns_,
        "training": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "scale_pos_weight": round(scale_pos_weight, 6),
            "cv_roc_auc_folds": [round(float(score), 6) for score in cv_scores],
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 6),
            "cv_roc_auc_std": round(float(cv_scores.std()), 6),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "lgbm_params": LGBM_PARAMS,
        },
    }
    FEATURE_NAMES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for path in (MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH):
        logger.info(
            "Saved %s (%.1f KB)",
            path.relative_to(PROJECT_ROOT),
            path.stat().st_size / 1024,
        )


if __name__ == "__main__":
    main()
