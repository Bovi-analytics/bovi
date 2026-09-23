"""Scikit uses the shared atomic bundle store with native joblib serialization."""

from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pytest
from bovi_core.ml import TrainingContext
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from scikit_sgd import ScikitSGDModelProvider, ScikitSGDTrainer, ScikitSGDTrainingConfig


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_incomplete_save_keeps_completed_metrics(
    model, dataloaders, tmp_path, monkeypatch, failure_at
):
    original = joblib.dump
    calls = 0

    def fail(payload, path):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            Path(path).write_bytes(b"partial")
            raise RuntimeError()
        return original(payload, path)

    monkeypatch.setattr(joblib, "dump", fail)
    result = ScikitSGDTrainer(
        model,
        dataloaders,
        ScikitSGDTrainingConfig(epochs=2, learning_rate=0.01),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    ).train()
    assert result.status == "failed"
    assert len(result.epochs) == (1 if failure_at <= 2 else 2)
    assert result.issues[0].exception_type == "RuntimeError"
    assert result.issues[0].message
    assert not list((tmp_path / "checkpoints").glob(".pending-*"))
    if failure_at == 1:
        assert result.last_checkpoint is None
    else:
        assert result.last_checkpoint is not None
        assert (
            LocalCheckpointResolver().resolve(result.last_checkpoint).metadata["completed_epoch"]
            == 1
        )
        assert result.best_epoch == (None if failure_at == 2 else 1)


def test_output_reuse_preserves_native_state(model, model_config, dataloaders, tmp_path):
    trainer = ScikitSGDTrainer(
        model,
        dataloaders,
        ScikitSGDTrainingConfig(epochs=1, learning_rate=0.01),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    )
    first = trainer.train()
    assert first.status == "completed", first.issues
    assert first.last_checkpoint is not None
    inputs = np.full((1, len(model_config.feature_names)), 0.5)
    expected = model(inputs)
    second = trainer.train()
    assert second.status == "completed", second.issues
    assert second.last_checkpoint is not None
    assert first.last_checkpoint.uri != second.last_checkpoint.uri
    resource = LocalCheckpointResolver().resolve(first.last_checkpoint)
    assert resource.metadata["resume_scope"] == "weights_only"
    restored = ScikitSGDModelProvider().restore_checkpoint(model_config, resource)
    np.testing.assert_allclose(restored(inputs), expected)
    with pytest.raises(ValueError, match="feature order"):
        ScikitSGDModelProvider().restore_checkpoint(
            model_config.model_copy(update={"feature_names": model_config.feature_names[::-1]}),
            resource,
        )
