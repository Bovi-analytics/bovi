"""Machine learning components."""

# DataLoader system
from .dataloaders import (
    AbstractDataLoader,
    Dataset,
    DataSource,
    FrameworkAdapter,
    TransformRegistry,
    UniversalTransform,
)
from .models import Model
from .predictors import (
    BasePredictionResult,
    CallableModel,
    GenericPredictionResult,
    HumanReadablePredictionResult,
    PredictionInterface,
    Predictor,
    SamPredictionResult,
    SamuraiPredictionResult,
)
from .registry import ModelRegistry, PredictorRegistry, create_model

__all__ = [
    "ModelRegistry",
    "PredictorRegistry",
    "create_model",
    "Model",
    "Predictor",
    "PredictionInterface",
    "CallableModel",
    # Result classes
    "BasePredictionResult",
    "HumanReadablePredictionResult",
    "GenericPredictionResult",
    "SamPredictionResult",
    "SamuraiPredictionResult",
    # DataLoader system
    "DataSource",
    "Dataset",
    "AbstractDataLoader",
    "UniversalTransform",
    "TransformRegistry",
    "FrameworkAdapter",
]
