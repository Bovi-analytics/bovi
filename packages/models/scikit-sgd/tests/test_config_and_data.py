"""Tests for typed configuration and the config-driven data pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from bovi_core.config import Config
from scikit_sgd import (
    ScikitSGDEvaluationConfig,
    ScikitSGDModelConfig,
    ScikitSGDTrainingConfig,
    create_dataloader,
)


def test_typed_configs_are_built_from_the_model_node(experiment_config: Config) -> None:
    model_config = ScikitSGDModelConfig.from_config(experiment_config)
    training_config = ScikitSGDTrainingConfig.from_config(experiment_config)
    evaluation_config = ScikitSGDEvaluationConfig.from_config(experiment_config)

    assert model_config.feature_names == ("days_in_milk", "parity", "previous_yield")
    assert training_config.epochs == 40
    assert training_config.learning_rate == 0.05
    assert evaluation_config.metrics == ("mse", "mae", "r2")

    with pytest.raises(Exception):
        training_config.epochs = 1  # type: ignore[misc]


def test_dataloader_applies_transforms_and_collates_nested_features(
    experiment_config: Config,
    model_config: ScikitSGDModelConfig,
) -> None:
    loader = create_dataloader(experiment_config, model_config, "train")

    batch = next(iter(loader))

    assert set(batch) == {"features", "labels", "metadata"}
    assert set(batch["features"]) == set(model_config.feature_names)
    assert batch["labels"].shape == (4,)
    assert len(batch["metadata"]) == 4
    for values in batch["features"].values():
        assert isinstance(values, np.ndarray)
        assert values.shape == (4,)
        assert np.all((0.0 <= values) & (values <= 1.0))
