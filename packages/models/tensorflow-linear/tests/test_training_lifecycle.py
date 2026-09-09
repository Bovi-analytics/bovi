"""Native trainer integration with shared lifecycle counters and best selection."""

import pytest
from tensorflow_linear import (
    TensorFlowLinearModelProvider,
    TensorFlowLinearTrainer,
    TensorFlowLinearTrainingConfig,
)
from tensorflow_linear.trainers import trainer as trainer_module

pytestmark = pytest.mark.tensorflow


def test_native_steps_count_exposures_and_select_raw_best(pipeline, monkeypatch):
    _, definition, loaders = pipeline
    model = TensorFlowLinearModelProvider().create(definition)
    scores = iter([10.0, 9.6, 9.2])

    def measure(model, batches):
        list(batches)
        return len(loaders["train"].dataset), {"mse": next(scores), "mae": 1.0}

    monkeypatch.setattr(trainer_module, "measure", measure)
    result = TensorFlowLinearTrainer(
        model,
        {"train": loaders["train"]},
        TensorFlowLinearTrainingConfig(epochs=3, min_delta=1.0, early_stopping_patience=2),
    ).train()
    assert result.status == "completed", result.issues
    assert result.stop_reason == "early_stopping"
    assert result.best_epoch == 3
    assert result.num_examples is not None
    assert result.num_examples == len(loaders["train"].dataset)
    assert result.num_examples_processed == 3 * result.num_examples
