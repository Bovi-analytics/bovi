"""
DataLoader system for bovi-core.

NumPy-First Architecture:
- Datasets return raw NumPy arrays/dicts
- Sample transforms are explicit on TransformedDataset, before batching
- Vision transforms use Albumentations directly
- Tabular transforms use UniversalTransform base class
"""

from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset
from bovi_core.ml.dataloaders.loaders.base_loader import AbstractDataLoader
from bovi_core.ml.dataloaders.sources.base_source import DataSource
from bovi_core.ml.dataloaders.transforms.base_transform import UniversalTransform

from .datasets import (
    FeatureVectorDataset,
    ImageDataset,
    TabularDataset,
    TransformedDataset,
    VideoDataset,
)
from .loaders import PyTorchDataLoader, SklearnDataLoader, TensorFlowDataLoader
from .sources import BlobImageSource, JSONRecordsSource, LocalFileSource, TransformedSource
from .transforms import TransformRegistry, build_vision_pipeline

__all__ = [
    # Base abstractions
    "DataSource",
    "Dataset",
    "AbstractDataLoader",
    "UniversalTransform",
    # Datasets
    "ImageDataset",
    "VideoDataset",
    "FeatureVectorDataset",
    "TabularDataset",
    "TransformedDataset",
    # Loaders
    "PyTorchDataLoader",
    "TensorFlowDataLoader",
    "SklearnDataLoader",
    # Sources
    "LocalFileSource",
    "JSONRecordsSource",
    "BlobImageSource",
    "TransformedSource",
    # Transforms
    "TransformRegistry",
    "build_vision_pipeline",
]
