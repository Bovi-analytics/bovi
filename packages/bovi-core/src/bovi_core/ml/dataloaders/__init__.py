"""
DataLoader system for bovi-core.

NumPy-First Architecture:
- Datasets return raw NumPy arrays/dicts
- Transforms are applied in DataLoaders via FrameworkAdapter
- Vision transforms use Albumentations directly
- Tabular transforms use UniversalTransform base class
"""

from .adapters import FrameworkAdapter
from .base import AbstractDataLoader, Dataset, DataSource, UniversalTransform
from .datasets import FeatureVectorDataset, ImageDataset, VideoDataset
from .loaders import PyTorchDataLoader, SklearnDataLoader, TensorFlowDataLoader
from .sources import BlobImageSource, LocalFileSource, TransformedSource
from .transforms import TransformRegistry, build_vision_pipeline

__all__ = [
    # Base abstractions
    "DataSource",
    "Dataset",
    "AbstractDataLoader",
    "UniversalTransform",
    # Adapters
    "FrameworkAdapter",
    # Datasets
    "ImageDataset",
    "VideoDataset",
    "FeatureVectorDataset",
    # Loaders
    "PyTorchDataLoader",
    "TensorFlowDataLoader",
    "SklearnDataLoader",
    # Sources
    "LocalFileSource",
    "BlobImageSource",
    "TransformedSource",
    # Transforms
    "TransformRegistry",
    "build_vision_pipeline",
]
