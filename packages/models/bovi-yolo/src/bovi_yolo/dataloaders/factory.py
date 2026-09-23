"""Config-driven construction of the YOLO data pipeline."""

from __future__ import annotations

from bovi_core.ml.dataloaders import PyTorchDataLoader, build_vision_pipeline
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.transforms import AlbumentationsTransform, ImagePreprocessing

from bovi_yolo.models import YOLOModelConfig

from .config import YOLODataLoaderConfig
from .dataset import YOLODataset
from .source import create_source


def create_dataloader(
    data_config: YOLODataLoaderConfig,
    model_config: YOLOModelConfig,
) -> PyTorchDataLoader:
    """Build the source, transforms, dataset, and loader for one split."""
    source = create_source(data_config.source)
    dataset = YOLODataset(source=source)
    transform = build_vision_pipeline(
        [transform.model_dump() for transform in data_config.transforms]
    )
    prepared_dataset = TransformedDataset(
        dataset,
        [
            AlbumentationsTransform(transform),
            ImagePreprocessing(normalize=True, channels_first=True),
        ],
    )

    return PyTorchDataLoader(
        dataset=prepared_dataset,
        split=data_config.split,
        batch_size=data_config.dataloader.batch_size,
        shuffle=data_config.dataloader.shuffle,
        seed=data_config.dataloader.seed,
        num_workers=data_config.dataloader.num_workers,
        pin_memory=data_config.dataloader.pin_memory,
        drop_last=data_config.dataloader.drop_last,
        persistent_workers=data_config.dataloader.persistent_workers,
        prefetch_factor=data_config.dataloader.prefetch_factor,
    )
