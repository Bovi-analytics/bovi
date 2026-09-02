"""Shared fixtures for the scikit SGD example package."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from bovi_core.config import Config
from scikit_sgd import (
    ScikitSGDModel,
    ScikitSGDModelConfig,
    ScikitSGDModelProvider,
    create_regression_dataloader,
)


@pytest.fixture
def experiment_config() -> Iterator[Config]:
    Config.reset()
    yield Config(experiment_name="scikit_sgd", project_name="scikit-sgd")
    Config.reset()


@pytest.fixture
def model_config(experiment_config: Config) -> ScikitSGDModelConfig:
    return ScikitSGDModelConfig.from_config(experiment_config)


@pytest.fixture
def model(model_config: ScikitSGDModelConfig) -> ScikitSGDModel:
    return ScikitSGDModelProvider().create(model_config)


@pytest.fixture
def dataloaders(experiment_config: Config, model_config: ScikitSGDModelConfig):
    return {
        split: create_regression_dataloader(experiment_config, model_config, split)
        for split in ("train", "validation")
    }


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "training-output"
