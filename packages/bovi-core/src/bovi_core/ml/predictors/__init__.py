"""Base predictor classes and prediction result types."""

from .prediction_interface import CallableModel, PredictionInterface, PredictorProtocol
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
    "PredictorProtocol",
    # Result classes
    "BasePredictionResult",
    "HumanReadablePredictionResult",
    "GenericPredictionResult",
    "SamPredictionResult",
    "SamuraiPredictionResult",
]
