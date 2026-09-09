"""Config-driven construction of the YOLO data pipeline."""

from __future__ import annotations

from typing import Any

from bovi_core.config import Config
from bovi_core.ml.dataloaders import PyTorchDataLoader, build_vision_pipeline
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.transforms import AlbumentationsTransform, ImagePreprocessing

from .dataset import YOLODataset
from .source import create_source


def create_dataloader(
    config: Config,
    split: str,
    **override_params: Any,
) -> PyTorchDataLoader:
    """Build the source, transforms, dataset, and loader for one split."""
    split_config = getattr(config.experiment.models.yolo.dataloaders, split)
    source = create_source(config, split)
    dataset = YOLODataset(source=source, config=config)
    transform = build_vision_pipeline(split_config.transforms)
    prepared_dataset = TransformedDataset(
        dataset,
        [
            AlbumentationsTransform(transform),
            ImagePreprocessing(normalize=True, channels_first=True),
        ],
    )

    return PyTorchDataLoader(
        dataset=prepared_dataset,
        config=config,
        split=split,
        model_name="yolo",
        **override_params,
    )
