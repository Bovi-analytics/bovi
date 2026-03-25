"""
DataLoader system for bovi-core.

NumPy-First Architecture:
- Datasets return raw NumPy arrays/dicts
- Transforms are applied in DataLoaders via FrameworkAdapter
- Vision transforms use Albumentations directly
- Tabular transforms use UniversalTransform base class
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .adapters import FrameworkAdapter
from .base import AbstractDataLoader, Dataset, DataSource, UniversalTransform
from .datasets import FeatureVectorDataset, ImageDataset, VideoDataset
from .loaders import PyTorchDataLoader, SklearnDataLoader, TensorFlowDataLoader
from .sources import BlobImageSource, LocalFileSource, TransformedSource
from .transforms import TransformRegistry, build_vision_pipeline

if TYPE_CHECKING:
    from bovi_core.config import Config


def create_dataloader(
    config: Config,
    model_name: str,
    split: str = "train",
    framework: str = "pytorch",
    **override_params: object,
) -> AbstractDataLoader:
    """
    Create dataloader from config for specific model and split.

    This factory function builds a complete dataloader pipeline from config:
    1. Creates data source (blob, local, unity_catalog_table, etc.)
    2. Builds transform pipeline from config (via Registry)
    3. Creates dataset (WITHOUT transforms - returns raw NumPy)
    4. Wraps in framework-specific loader (transforms applied here)

    Args:
        config: Config instance.
        model_name: Model name (e.g., 'yolo', 'snn').
        split: Dataset split ('train', 'val', 'test').
        framework: Framework to use ('pytorch', 'tensorflow', 'sklearn').
        **override_params: Override config parameters (batch_size, num_workers, etc.).

    Returns:
        Configured dataloader instance.

    Raises:
        AttributeError: If model or dataloader config not found.
        ValueError: If unknown source type or framework.

    Example:
        ```python
        from bovi_core.config import Config
        from bovi_core.ml.dataloaders import create_dataloader

        config = Config(experiment_name="first_yolo_experiment")

        # Create YOLO dataloaders (640x640 images)
        yolo_train = create_dataloader(config, "yolo", "train")
        yolo_val = create_dataloader(config, "yolo", "val")

        # Override config parameters
        yolo_test = create_dataloader(
            config, "yolo", "test",
            batch_size=128,  # Override
            num_workers=16
        )

        # Use in training loop
        for batch in yolo_train:
            images = batch["image"]  # (B, C, H, W) float32 tensor
            labels = batch["label"]
            # ... training code
        ```
    """
    # Get model-specific dataloader config
    model_config = getattr(config.experiment.models, model_name)
    if not hasattr(model_config, "dataloaders"):
        raise AttributeError(
            f"Model '{model_name}' has no dataloaders config. "
            f"Add 'dataloaders' section to models.{model_name} in config.yaml"
        )

    dataloader_config = getattr(model_config.dataloaders, split)

    # Create source
    source = _create_source(dataloader_config.source, config)

    # Create dataset (NO transforms - returns raw NumPy)
    dataset = _create_dataset(source, config, dataloader_config)

    # Build transform pipeline if configured
    transform: Callable[..., dict[str, object]] | None = None
    if hasattr(dataloader_config, "transforms") and dataloader_config.transforms:
        transform = build_vision_pipeline(dataloader_config.transforms)

    # Create loader WITH transform
    if framework == "pytorch":
        return PyTorchDataLoader(
            dataset, config, split, model_name, transform=transform, **override_params
        )
    elif framework == "tensorflow":
        return TensorFlowDataLoader(
            dataset, config, split, model_name, transform=transform, **override_params
        )
    elif framework == "sklearn":
        return SklearnDataLoader(dataset, config, split, model_name, **override_params)
    else:
        raise ValueError(f"Unknown framework: {framework}. Supported: pytorch, tensorflow, sklearn")


def _create_source(source_config: object, config: Config) -> DataSource:
    """
    Create data source from config.

    Args:
        source_config: Source configuration from YAML.
        config: Config instance.

    Returns:
        DataSource instance.

    Raises:
        ValueError: If unknown source type.
    """
    source_type = getattr(source_config, "type", None)

    if source_type == "local":
        # Local filesystem source
        return LocalFileSource(
            root_dir=getattr(source_config, "root_dir", ""),
            file_pattern=getattr(source_config, "file_pattern", "*.jpg"),
            recursive=getattr(source_config, "recursive", True),
        )

    elif source_type == "blob":
        # Azure Blob Storage source
        return BlobImageSource(
            config=config,
            container=getattr(source_config, "container", ""),
            prefix=getattr(source_config, "prefix", ""),
            file_pattern=getattr(source_config, "file_pattern", "*.jpg"),
        )

    elif source_type == "unity_catalog_table":
        # Unity Catalog Parquet source
        from .sources.unity_catalog_source import UnityCatalogParquetSource

        return UnityCatalogParquetSource(
            config=config,
            catalog=getattr(source_config, "catalog", ""),
            schema=getattr(source_config, "schema", ""),
            table=getattr(source_config, "table", ""),
        )

    else:
        raise ValueError(
            f"Unknown source type: {source_type}. Supported: local, blob, unity_catalog_table"
        )


def _create_dataset(source: DataSource, config: Config, dataloader_config: object) -> Dataset:
    """
    Create dataset from source (NO transforms).

    Args:
        source: Data source.
        config: Config instance.
        dataloader_config: Dataloader configuration.

    Returns:
        Dataset instance.
    """
    # Determine dataset type from config or auto-detect
    dataset_type = getattr(dataloader_config, "dataset_type", "image")

    if dataset_type == "image":
        return ImageDataset(source=source, config=config)
    elif dataset_type == "video":
        frame_size_value = getattr(dataloader_config, "frame_size", (224, 224))
        frame_size: tuple[int, int] = (
            tuple(frame_size_value) if isinstance(frame_size_value, (list, tuple)) else (224, 224)
        )  # type: ignore[assignment]
        num_frames = int(getattr(dataloader_config, "num_frames", 16))
        return VideoDataset(
            source=source, config=config, frame_size=frame_size, num_frames=num_frames
        )
    else:
        # Default to ImageDataset
        return ImageDataset(source=source, config=config)


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
    # Factory function
    "create_dataloader",
]
