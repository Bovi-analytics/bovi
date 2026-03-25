"""
Transform system for dataloaders.

NumPy-First Architecture:
- Vision transforms: Use Albumentations directly via TransformRegistry
- Tabular transforms: Use UniversalTransform base class (stateless)

Provides:
- TransformRegistry: Plugin system for transforms with auto-registration
- build_vision_pipeline: Build Albumentations Compose from config
- Time-series transforms: Imputation, Normalization, Padding, Windowing
"""

# Import timeseries to trigger registration
from bovi_core.ml.dataloaders.transforms import timeseries  # noqa: F401
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry
from bovi_core.ml.dataloaders.transforms.timeseries import (
    ImputationTransform,
    SequenceNormalizationTransform,
    SequencePaddingTransform,
    WindowingTransform,
)

# Convenience alias
build_vision_pipeline = TransformRegistry.build_vision_pipeline

__all__ = [
    "TransformRegistry",
    "build_vision_pipeline",
    # Time-series transforms
    "ImputationTransform",
    "SequenceNormalizationTransform",
    "SequencePaddingTransform",
    "WindowingTransform",
]
