"""CPU integration for lactation training, evaluation, and weights restoration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from bovi_core.config import Config
from bovi_core.ml import EvaluationContext, TrainingContext
from bovi_core.ml.dataloaders import SklearnDataLoader
from bovi_core.ml.dataloaders.datasets import Dataset
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from lactation_autoencoder import (
    LactationAutoencoderEvaluationConfig,
    LactationAutoencoderEvaluator,
    LactationAutoencoderModelConfig,
    LactationAutoencoderModelProvider,
    LactationAutoencoderTrainer,
    LactationAutoencoderTrainingConfig,
)
from pydantic import ValidationError


class SyntheticLactationDataset(Dataset):
    """Small deterministic reconstruction dataset without storage dependencies."""

    def __init__(self, *, samples: int, days: int, herd_stats: int) -> None:
        self._samples = samples
        self._days = days
        self._herd_stats = herd_stats

    def __len__(self) -> int:
        return self._samples

    def __getitem__(self, index: int) -> dict[str, object]:
        milk = np.linspace(0.1, 0.9, self._days, dtype=np.float32)
        milk = np.clip(milk + index * 0.01, 0.0, 1.0)
        return {
            "features": {
                "milk": milk,
                "events": np.full(self._days, index % 3, dtype=np.int32),
                "parity": np.array([1 + index % 2], dtype=np.float32),
                "herd_stats": np.linspace(0.0, 1.0, self._herd_stats, dtype=np.float32),
            },
            "labels": milk.copy(),
            "metadata": {"index": index},
        }


def test_training_evaluation_and_weights_restore_on_cpu(tmp_path) -> None:
    config = LactationAutoencoderModelConfig(
        framework="tensorflow",
        input_dim=8,
        latent_dim=4,
        num_events=3,
        num_herd_stats=2,
    )
    dataset = SyntheticLactationDataset(samples=6, days=8, herd_stats=2)
    train = SklearnDataLoader(dataset, split="train", batch_size=3, shuffle=False)
    validation = SklearnDataLoader(dataset, split="validation", batch_size=3, shuffle=False)
    provider = LactationAutoencoderModelProvider()
    model = provider.create(config)
    context = TrainingContext(run_id=uuid4(), output_dir=tmp_path / "training")

    result = LactationAutoencoderTrainer(
        model,
        {"train": train, "validation": validation},
        LactationAutoencoderTrainingConfig(epochs=2, learning_rate=0.01),
        context,
    ).train()

    assert result.status == "completed", result.issues
    assert len(result.epochs) == 2
    assert result.num_examples == 6
    assert result.num_examples_processed == 12
    assert result.last_checkpoint is not None
    assert result.best_checkpoint is not None
    resolved = LocalCheckpointResolver().resolve(result.last_checkpoint)
    assert resolved.metadata["resume_scope"] == "weights_only"
    restored = provider.restore_checkpoint(config, resolved)

    batch = next(iter(validation))
    original = model.trainable_model(
        {
            "input_11": np.expand_dims(batch["features"]["milk"], -1),
            "input_12": batch["features"]["parity"],
            "input_13": batch["features"]["events"],
            "input_15": batch["features"]["herd_stats"],
        },
        training=False,
    )
    restored_output = restored.trainable_model(
        {
            "input_11": np.expand_dims(batch["features"]["milk"], -1),
            "input_12": batch["features"]["parity"],
            "input_13": batch["features"]["events"],
            "input_15": batch["features"]["herd_stats"],
        },
        training=False,
    )
    np.testing.assert_allclose(restored_output, original, atol=1e-6)

    evaluation = LactationAutoencoderEvaluator(
        restored,
        LactationAutoencoderEvaluationConfig(metrics=("mse", "mae", "rmse")),
    ).evaluate(
        validation,
        EvaluationContext(
            evaluation_id=uuid4(),
            output_dir=tmp_path / "evaluation",
            split="validation",
            model_version="last",
            training_run_id=context.run_id,
        ),
    )
    assert evaluation.status == "completed", evaluation.issues
    assert evaluation.num_examples == 6
    assert set(evaluation.metrics) == {"mse", "mae", "rmse"}
    assert all(np.isfinite(value) for value in evaluation.metrics.values())


def test_training_configs_are_immutable_and_validate_yaml() -> None:
    Config.reset()
    package_root = Path(__file__).resolve().parents[1]
    config = Config(
        experiment_name="lactation_autoencoder",
        project_file_path=str(package_root / "pyproject.toml"),
        config_file_path=str(
            package_root / "data/experiments/lactation_autoencoder/versions/v15/config/config.yaml"
        ),
    )
    training = LactationAutoencoderTrainingConfig.from_config(config)
    evaluation = LactationAutoencoderEvaluationConfig.from_config(config)

    assert training.optimizer == "adam"
    assert training.loss == "mse"
    assert evaluation.metrics == ("mse", "mae", "rmse")
    with pytest.raises(ValidationError):
        training.epochs = 1
    Config.reset()
