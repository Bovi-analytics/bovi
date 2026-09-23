"""Tests for config-driven lactation dataloader construction."""

from pathlib import Path

from bovi_core.config import Config
from bovi_core.ml import DataLoaderFactoryRegistry
from bovi_core.ml import create_dataloader as dispatch_dataloader
from bovi_core.ml.dataloaders import SklearnDataLoader
from lactation_autoencoder.dataloaders import (
    LactationAutoencoderDataLoaderConfig,
    LactationDatasetSettings,
    LactationJSONSourceSettings,
    create_dataloader,
)
from lactation_autoencoder.models import LactationAutoencoderModelConfig


def test_dataloader_factory_is_discovered_from_package_entry_point() -> None:
    DataLoaderFactoryRegistry.clear()

    assert DataLoaderFactoryRegistry.get("autoencoder") is create_dataloader


def test_create_dataloader_composes_inference_pipeline() -> None:
    Config.reset()
    config = Config(
        experiment_name="lactation_autoencoder",
        project_name="lactation-autoencoder",
    )

    data_config = LactationAutoencoderDataLoaderConfig.from_config(config, split="inference")
    model_config = LactationAutoencoderModelConfig.from_config(config)
    loader = dispatch_dataloader("autoencoder", data_config, model_config)
    batch = next(iter(loader))

    assert isinstance(loader, SklearnDataLoader)
    assert len(batch["features"]["milk"]) == 1
    assert batch["labels"].shape == (1, 304)


def test_data_config_supports_direct_construction(tmp_path: Path) -> None:
    data_config = LactationAutoencoderDataLoaderConfig(
        split="calibration",
        dataset=LactationDatasetSettings(max_days=120, keep_in_memory=False),
        source=LactationJSONSourceSettings(
            type="lactation_json",
            json_root_dir=tmp_path,
        ),
    )

    assert data_config.split == "calibration"
    assert data_config.dataset.max_days == 120
    assert data_config.source.json_root_dir == tmp_path
