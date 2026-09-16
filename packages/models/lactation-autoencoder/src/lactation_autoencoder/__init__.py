"""Lactation autoencoder model module."""

from typing import TYPE_CHECKING, Any

# Import transforms to trigger TransformRegistry registration
from lactation_autoencoder.dataloaders import (
    LactationAutoencoderDataLoaderConfig,
    create_dataloader,
)
from lactation_autoencoder.dataloaders.transforms import (
    EventTokenizationTransform,
    HerdStatsEnrichmentTransform,
    HerdStatsNormalizationTransform,
    MilkNormalizationTransform,
)

if TYPE_CHECKING:
    from lactation_autoencoder.models import (
        LactationAutoencoderModel,
        LactationAutoencoderModelConfig,
        LactationAutoencoderModelProvider,
    )
    from lactation_autoencoder.predictors import LactationPredictionResult, LactationPredictor
    from lactation_autoencoder.trainers import (
        LactationAutoencoderEvaluationConfig,
        LactationAutoencoderEvaluator,
        LactationAutoencoderTrainer,
        LactationAutoencoderTrainingConfig,
    )

__all__ = [
    "EventTokenizationTransform",
    "HerdStatsEnrichmentTransform",
    "MilkNormalizationTransform",
    "HerdStatsNormalizationTransform",
    "create_dataloader",
    "LactationAutoencoderDataLoaderConfig",
    "LactationAutoencoderModel",
    "LactationAutoencoderModelConfig",
    "LactationAutoencoderModelProvider",
    "LactationPredictor",
    "LactationPredictionResult",
    "LactationAutoencoderTrainingConfig",
    "LactationAutoencoderEvaluationConfig",
    "LactationAutoencoderTrainer",
    "LactationAutoencoderEvaluator",
]


def __getattr__(name: str) -> Any:
    if name in {
        "LactationAutoencoderModel",
        "LactationAutoencoderModelConfig",
        "LactationAutoencoderModelProvider",
    }:
        from lactation_autoencoder.models import (
            LactationAutoencoderModel,
            LactationAutoencoderModelConfig,
            LactationAutoencoderModelProvider,
        )

        return {
            "LactationAutoencoderModel": LactationAutoencoderModel,
            "LactationAutoencoderModelConfig": LactationAutoencoderModelConfig,
            "LactationAutoencoderModelProvider": LactationAutoencoderModelProvider,
        }[name]
    if name in {"LactationPredictor", "LactationPredictionResult"}:
        from lactation_autoencoder.predictors import LactationPredictionResult, LactationPredictor

        return {
            "LactationPredictor": LactationPredictor,
            "LactationPredictionResult": LactationPredictionResult,
        }[name]
    if name in {
        "LactationAutoencoderTrainingConfig",
        "LactationAutoencoderEvaluationConfig",
        "LactationAutoencoderTrainer",
        "LactationAutoencoderEvaluator",
    }:
        from lactation_autoencoder.trainers import (
            LactationAutoencoderEvaluationConfig,
            LactationAutoencoderEvaluator,
            LactationAutoencoderTrainer,
            LactationAutoencoderTrainingConfig,
        )

        return {
            "LactationAutoencoderTrainingConfig": LactationAutoencoderTrainingConfig,
            "LactationAutoencoderEvaluationConfig": LactationAutoencoderEvaluationConfig,
            "LactationAutoencoderTrainer": LactationAutoencoderTrainer,
            "LactationAutoencoderEvaluator": LactationAutoencoderEvaluator,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
