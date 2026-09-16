"""YOLO cow detection model module."""

# Import transforms to trigger TransformRegistry registration
from bovi_yolo.dataloaders import YOLODataLoaderConfig, create_dataloader
from bovi_yolo.dataloaders.transforms import (
    ImageResizeTransform,
    ImageValidationTransform,
)

# Import model provider to trigger ModelProviderRegistry registration
from bovi_yolo.models import YOLOModel, YOLOModelConfig, YOLOModelProvider

# Import predictor and result
from bovi_yolo.predictors import YoloPredictionResult, YOLOPredictor

__all__ = [
    "ImageResizeTransform",
    "ImageValidationTransform",
    "create_dataloader",
    "YOLODataLoaderConfig",
    "YOLOModel",
    "YOLOModelConfig",
    "YOLOModelProvider",
    "YOLOPredictor",
    "YoloPredictionResult",
]
