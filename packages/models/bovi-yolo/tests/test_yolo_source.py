"""Tests for YOLO source and dataloader construction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from bovi_core.ml.dataloaders import PyTorchDataLoader
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.sources import LocalFileSource
from bovi_yolo.dataloaders import create_dataloader, create_source


def test_create_source_from_config(yolo_config: object) -> None:
    source = create_source(yolo_config, split="inference")  # type: ignore[arg-type]

    assert isinstance(source, LocalFileSource)
    assert len(source) >= 1


def test_create_dataloader_composes_pipeline(yolo_config: object) -> None:
    loader = create_dataloader(
        yolo_config,  # type: ignore[arg-type]
        split="inference",
        num_workers=0,
    )

    batch = next(iter(loader))

    assert isinstance(loader, PyTorchDataLoader)
    assert isinstance(loader.dataset, TransformedDataset)
    assert tuple(batch["image"].shape) == (1, 3, 640, 640)
    assert batch["image"].is_floating_point()
    assert batch["image"].min() >= 0
    assert batch["image"].max() <= 1


def test_create_source_rejects_unsupported_type() -> None:
    config = MagicMock()
    config.experiment.models.yolo.dataloaders.train.source.type = "s3"

    with pytest.raises(ValueError, match="Unsupported YOLO source type"):
        create_source(config, split="train")
