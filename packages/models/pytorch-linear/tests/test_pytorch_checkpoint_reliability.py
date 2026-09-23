"""PyTorch checkpoint failures and epoch-boundary recovery contracts."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from bovi_core.ml import TrainingContext
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from pytorch_linear import (
    PyTorchLinearModelProvider,
    PyTorchLinearTrainer,
    PyTorchLinearTrainingConfig,
)

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_incomplete_native_save_retains_metrics(pipeline, tmp_path, monkeypatch, failure_at):
    _, definition, loaders = pipeline
    model = PyTorchLinearModelProvider().create(definition)
    calls = 0
    import torch

    original = torch.save

    def fail(payload, path):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            Path(path).write_bytes(b"partial")
            raise RuntimeError()
        return original(payload, path)

    monkeypatch.setattr(torch, "save", fail)
    result = PyTorchLinearTrainer(
        model,
        loaders,
        PyTorchLinearTrainingConfig(epochs=2),
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
        resource = LocalCheckpointResolver().resolve(result.last_checkpoint)
        assert resource.metadata["completed_epoch"] == 1
        assert result.best_epoch == (None if failure_at == 2 else 1)
        assert resource.metadata["resume_scope"] == "weights_only"


def test_output_reuse_does_not_change_old_reference(pipeline, tmp_path):
    _, definition, loaders = pipeline
    provider = PyTorchLinearModelProvider()
    model = provider.create(definition)
    trainer = PyTorchLinearTrainer(
        model,
        loaders,
        PyTorchLinearTrainingConfig(epochs=1),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    )
    first = trainer.train()
    assert first.status == "completed", first.issues
    assert first.last_checkpoint is not None
    expected = model([[0.5]])
    second = trainer.train()
    assert second.status == "completed", second.issues
    assert second.last_checkpoint is not None
    assert first.last_checkpoint.uri != second.last_checkpoint.uri
    resource = LocalCheckpointResolver().resolve(first.last_checkpoint)
    restored = provider.restore_checkpoint(definition, resource)
    np.testing.assert_allclose(restored([[0.5]]), expected, atol=1e-6)


def test_partial_epoch_cancellation_identifies_prior_checkpoint(pipeline, tmp_path, monkeypatch):
    _, definition, loaders = pipeline
    provider = PyTorchLinearModelProvider()
    model = provider.create(definition)
    trainer = PyTorchLinearTrainer(
        model,
        loaders,
        PyTorchLinearTrainingConfig(epochs=3),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    )
    from bovi_core.ml.trainers import lifecycle
    from pytorch_linear.trainers import trainer as trainer_module

    calls = 0
    convert = trainer_module.batch_to_arrays

    def counted_batch(*args):
        nonlocal calls
        calls += 1
        return convert(*args)

    def check_deadline(context):
        if calls > len(loaders["train"]):
            raise lifecycle.DeadlineReached

    monkeypatch.setattr(trainer_module, "batch_to_arrays", counted_batch)
    monkeypatch.setattr(lifecycle, "check_deadline", check_deadline)
    result = trainer.train()
    assert result.status == "cancelled"
    assert len(result.epochs) == 1
    assert result.last_checkpoint is not None
    resource = LocalCheckpointResolver().resolve(result.last_checkpoint)
    assert resource.metadata["completed_epoch"] == 1
    assert result.issues[0].code == "checkpoint_not_current_model"
    restored = provider.restore_checkpoint(definition, resource)
    assert not np.allclose(restored([[0.5]]), model([[0.5]]))


def test_missing_required_bundle_file_is_rejected(pipeline, tmp_path):
    _, definition, loaders = pipeline
    result = PyTorchLinearTrainer(
        PyTorchLinearModelProvider().create(definition),
        loaders,
        PyTorchLinearTrainingConfig(epochs=1),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    ).train()
    assert result.last_checkpoint is not None
    resource = LocalCheckpointResolver().resolve(result.last_checkpoint)
    assert resource.local_path is not None
    path = resource.local_path
    path.unlink()
    with pytest.raises(ValueError, match="files"):
        LocalCheckpointResolver().resolve(result.last_checkpoint)
