"""Tests for YOLO source and dataloader construction."""

from __future__ import annotations

from pathlib import Path

import pytest
from bovi_core.config import Config
from bovi_core.ml import DataLoaderFactoryRegistry
from bovi_core.ml import create_dataloader as dispatch_dataloader
from bovi_core.ml.dataloaders import PyTorchDataLoader
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.sources import LocalFileSource
from bovi_yolo.dataloaders import (
    YOLODataLoaderConfig,
    YOLODatasetSettings,
    YOLOLocalSourceSettings,
    create_dataloader,
    create_source,
)
from bovi_yolo.models import YOLOModelConfig
from pydantic import ValidationError


def test_dataloader_factory_is_discovered_from_package_entry_point() -> None:
    DataLoaderFactoryRegistry.clear()

    assert DataLoaderFactoryRegistry.get("yolo") is create_dataloader


def test_create_source_from_config(yolo_config: Config) -> None:
    data_config = YOLODataLoaderConfig.from_config(yolo_config, split="inference")
    source = create_source(data_config.source)

    assert isinstance(source, LocalFileSource)
    assert len(source) >= 1


def test_create_dataloader_composes_pipeline(yolo_config: Config) -> None:
    data_config = YOLODataLoaderConfig.from_config(yolo_config, split="inference")
    model_config = YOLOModelConfig.from_config(yolo_config)
    loader = dispatch_dataloader(
        "yolo",
        data_config,
        model_config,
    )

    batch = next(iter(loader))

    assert isinstance(loader, PyTorchDataLoader)
    assert isinstance(loader.dataset, TransformedDataset)
    assert tuple(batch["image"].shape) == (1, 3, 640, 640)
    assert batch["image"].is_floating_point()
    assert batch["image"].min() >= 0
    assert batch["image"].max() <= 1


def test_data_config_supports_direct_construction(tmp_path: Path) -> None:
    data_config = YOLODataLoaderConfig(
        split="calibration",
        dataset=YOLODatasetSettings(),
        source=YOLOLocalSourceSettings(type="local", root_dir=tmp_path),
    )

    assert data_config.split == "calibration"
    assert data_config.source.root_dir == tmp_path


def test_data_config_rejects_unsupported_source_type() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        YOLODataLoaderConfig.model_validate(
            {
                "split": "train",
                "dataset": {"return_metadata": True},
                "source": {"type": "s3", "root_dir": "/tmp"},
            }
        )
