"""Score a single applicant against the trained default-risk model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd
from joblib import load

from src.ml.train import FEATURE_NAMES_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.utils.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.utils.logger import get_logger

logger = get_logger(__name__)

# An applicant this close to a band edge is routed to a human rather than
# being auto-decisioned on the wrong side of a rounding error.
BORDERLINE_MARGIN = 0.05


@dataclass(frozen=True)
class ModelArtifacts:
    """The trained model plus the schema needed to feed it."""

    model: Any
    preprocessor: Any
    model_features: list[str]
    raw_input_columns: list[str]


@lru_cache(maxsize=1)
def load_model_artifacts() -> ModelArtifacts:
    """Load and cache the persisted training artifacts."""
    missing = [
        path.name
        for path in (MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {', '.join(missing)}. Run 'python -m src.ml.train' first."
        )

    schema = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    artifacts = ModelArtifacts(
        model=load(MODEL_PATH),
        preprocessor=load(PREPROCESSOR_PATH),
        model_features=schema["model_features"],
        raw_input_columns=schema["raw_input_columns"],
    )
    logger.info(
        "Loaded model artifacts (%d model features)", len(artifacts.model_features)
    )
    return artifacts


def risk_band(probability: float) -> str:
    """Map a default probability onto the configured risk bands."""
    if probability < RISK_THRESHOLD_LOW:
        return "Low"
    if probability < RISK_THRESHOLD_HIGH:
        return "Medium"
    return "High"


def borderline_assessment(probability: float) -> dict[str, Any]:
    """Flag probabilities sitting within the review margin of a band edge."""
    distances = {
        "low": abs(probability - RISK_THRESHOLD_LOW),
        "high": abs(probability - RISK_THRESHOLD_HIGH),
    }
    nearest = min(distances, key=distances.get)
    distance = distances[nearest]
    return {
        "borderline_review": bool(distance <= BORDERLINE_MARGIN),
        "nearest_threshold": nearest,
        "distance_to_threshold": round(distance, 6),
    }


def to_feature_frame(applicant: Mapping[str, Any], artifacts: ModelArtifacts) -> pd.DataFrame:
    """Coerce a raw applicant mapping into the column layout the model expects."""
    if not isinstance(applicant, Mapping):
        raise TypeError("predict_risk expects a mapping of applicant fields.")

    unknown = sorted(set(applicant).difference(artifacts.raw_input_columns))
    if unknown:
        logger.warning("Ignoring %d unrecognised field(s): %s", len(unknown), unknown)

    frame = pd.DataFrame([dict(applicant)]).reindex(
        columns=artifacts.raw_input_columns
    )
    return artifacts.preprocessor.transform(frame)


def predict_risk(applicant: Mapping[str, Any]) -> dict[str, Any]:
    """Return the default probability and risk band for one applicant.

    Fields absent from the mapping are treated as missing and imputed by the
    preprocessor, so partial applications can still be scored.
    """
    artifacts = load_model_artifacts()
    features = to_feature_frame(applicant, artifacts)
    probability = float(
        artifacts.model.predict_proba(features.to_numpy(dtype="float32"))[0, 1]
    )

    supplied = sum(1 for column in artifacts.raw_input_columns if column in applicant)
    result = {
        "probability": round(probability, 6),
        "risk_band": risk_band(probability),
        "thresholds": {
            "low": RISK_THRESHOLD_LOW,
            "high": RISK_THRESHOLD_HIGH,
            "borderline_margin": BORDERLINE_MARGIN,
        },
        "fields_supplied": supplied,
        "fields_expected": len(artifacts.raw_input_columns),
    }
    result.update(borderline_assessment(probability))
    return result
