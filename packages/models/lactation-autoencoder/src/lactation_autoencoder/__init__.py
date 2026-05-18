"""Lactation autoencoder model module."""

from typing import Any

# Import transforms to trigger TransformRegistry registration
from lactation_autoencoder.dataloaders.transforms import (
    EventTokenizationTransform,
    HerdStatsEnrichmentTransform,
    HerdStatsNormalizationTransform,
    MilkNormalizationTransform,
)

__all__ = [
    "EventTokenizationTransform",
    "HerdStatsEnrichmentTransform",
    "MilkNormalizationTransform",
    "HerdStatsNormalizationTransform",
    "LactationAutoencoderModel",
    "LactationPredictor",
    "LactationPredictionResult",
]


def __getattr__(name: str) -> Any:
    if name == "LactationAutoencoderModel":
        from lactation_autoencoder.models import LactationAutoencoderModel

        return LactationAutoencoderModel
    if name in {"LactationPredictor", "LactationPredictionResult"}:
        from lactation_autoencoder.predictors import LactationPredictionResult, LactationPredictor

        return {
            "LactationPredictor": LactationPredictor,
            "LactationPredictionResult": LactationPredictionResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
