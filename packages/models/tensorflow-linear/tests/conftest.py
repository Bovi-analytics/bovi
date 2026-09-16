"""Shared native model pipeline fixture."""

from pathlib import Path

import pytest
from bovi_core.config import Config
from bovi_core.ml import create_dataloader
from tensorflow_linear import (
    TensorFlowLinearDataLoaderConfig,
    TensorFlowLinearModelConfig,
)


@pytest.fixture
def pipeline():
    Config.reset()
    root = Path(__file__).resolve().parents[1]
    config = Config(
        experiment_name="tensorflow_linear",
        project_file_path=str(root / "pyproject.toml"),
        config_file_path=str(
            root / "data/experiments/tensorflow_linear/versions/v1/config/config.yaml"
        ),
    )
    model_config = TensorFlowLinearModelConfig.from_config(config)
    loaders = {
        split: create_dataloader(
            "tensorflow_linear",
            TensorFlowLinearDataLoaderConfig.from_config(config, split),
            model_config,
        )
        for split in ("train", "validation")
    }
    yield config, model_config, loaders
    Config.reset()
