"""Minimal scikit-learn SGD model and training implementation."""

from .dataloaders import (
    RegressionJSONSource,
    ScikitRegressionDataset,
    create_dataloader,
)
from .models import ScikitSGDModel, ScikitSGDModelConfig, ScikitSGDModelProvider
from .trainers import (
    ScikitSGDEvaluationConfig,
    ScikitSGDEvaluator,
    ScikitSGDTrainer,
    ScikitSGDTrainingConfig,
)

__all__ = [
    "RegressionJSONSource",
    "ScikitRegressionDataset",
    "create_dataloader",
    "ScikitSGDEvaluationConfig",
    "ScikitSGDEvaluator",
    "ScikitSGDModel",
    "ScikitSGDModelConfig",
    "ScikitSGDModelProvider",
    "ScikitSGDTrainer",
    "ScikitSGDTrainingConfig",
]
