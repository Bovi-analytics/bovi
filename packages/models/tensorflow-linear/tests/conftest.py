"""Shared native model pipeline fixture."""

from pathlib import Path

import pytest
from bovi_core.config import Config
from tensorflow_linear import TensorFlowLinearModelConfig, create_dataloader


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
        split: create_dataloader(config, model_config, split) for split in ("train", "validation")
    }
    yield config, model_config, loaders
    Config.reset()
