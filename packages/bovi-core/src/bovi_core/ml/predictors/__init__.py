"""Base predictor classes and prediction result types."""

from .prediction_interface import CallableModel, PredictionInterface
from .predictor import Predictor
from .results import (
    BasePredictionResult,
    GenericPredictionResult,
    HumanReadablePredictionResult,
    SamPredictionResult,
    SamuraiPredictionResult,
)

__all__ = [
    # Interfaces
    "CallableModel",
    "PredictionInterface",
    "Predictor",
    # Result classes
    "BasePredictionResult",
    "HumanReadablePredictionResult",
    "GenericPredictionResult",
    "SamPredictionResult",
    "SamuraiPredictionResult",
]
