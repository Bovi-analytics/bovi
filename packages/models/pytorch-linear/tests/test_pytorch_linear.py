"""pytorch-linear CPU integration: config, data, training, checkpoints and evaluation."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from bovi_core.ml import (
    EvaluationContext,
    ModelProviderRegistry,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
    TrainingContext,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from pytorch_linear import (
    PyTorchLinearEvaluationConfig,
    PyTorchLinearEvaluator,
    PyTorchLinearModelProvider,
    PyTorchLinearTrainer,
    PyTorchLinearTrainingConfig,
)

pytestmark = pytest.mark.torch


def test_training_evaluation_and_resume(pipeline, tmp_path):
    config, definition, loaders = pipeline
    provider = PyTorchLinearModelProvider()
    assert ModelProviderRegistry.get("pytorch_linear") is PyTorchLinearModelProvider
    model = provider.create(definition)
    run = TrainingContext(run_id=uuid4(), output_dir=tmp_path / "first")
    trainer = PyTorchLinearTrainer(
        model, loaders, PyTorchLinearTrainingConfig.from_config(config), run
    )
    result = trainer.train()
    assert result.status == "completed", result.issues
    assert len(result.epochs) == 40
    assert result.epochs[-1].metrics["train_mse"] < result.epochs[0].metrics["train_mse"] / 100
    assert result.best_checkpoint is not None
    assert result.last_checkpoint is not None
    resource = LocalCheckpointResolver().resolve(result.last_checkpoint)
    assert resource.local_path is not None
    path = resource.local_path
    restored = provider.restore_checkpoint(definition, resource)
    np.testing.assert_allclose(restored([[0.5]]), model([[0.5]]), atol=1e-6)
    loaded = provider.load_artifact(
        definition,
        ResolvedModelArtifact(
            format=resource.format, source_uri=resource.source_uri, local_path=path
        ),
    )
    np.testing.assert_allclose(loaded([[0.5]]), model([[0.5]]), atol=1e-6)
    evaluation = PyTorchLinearEvaluator(
        restored, PyTorchLinearEvaluationConfig.from_config(config)
    ).evaluate(
        loaders["validation"],
        EvaluationContext(
            evaluation_id=uuid4(),
            output_dir=tmp_path / "evaluation",
            split="validation",
            model_version="last",
            training_run_id=run.run_id,
        ),
    )
    assert evaluation.status == "completed", evaluation.issues
    assert evaluation.num_examples == 4
    assert evaluation.metrics["mse"] < 0.001
    resumed = PyTorchLinearTrainer(
        restored,
        loaders,
        PyTorchLinearTrainingConfig(epochs=2),
        TrainingContext(
            run_id=uuid4(), resumed_from_run_id=run.run_id, output_dir=tmp_path / "resume"
        ),
    ).train()
    assert resumed.status == "completed"
    assert resumed.epochs[0].epoch == 1


def test_split_run_matches_continuous_training(pipeline, tmp_path):
    _, definition, loaders = pipeline
    provider = PyTorchLinearModelProvider()
    continuous = provider.create(definition)
    assert (
        PyTorchLinearTrainer(continuous, loaders, PyTorchLinearTrainingConfig(epochs=10))
        .train()
        .status
        == "completed"
    )
    first = provider.create(definition)
    result = PyTorchLinearTrainer(
        first,
        loaders,
        PyTorchLinearTrainingConfig(epochs=5),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    ).train()
    assert result.last_checkpoint is not None
    resumed = provider.restore_checkpoint(
        definition,
        LocalCheckpointResolver().resolve(result.last_checkpoint),
    )
    assert (
        PyTorchLinearTrainer(resumed, loaders, PyTorchLinearTrainingConfig(epochs=5)).train().status
        == "completed"
    )
    np.testing.assert_allclose(resumed([[-0.5], [0.5]]), continuous([[-0.5], [0.5]]), atol=1e-6)


def test_stopping_and_failed_data(pipeline, tmp_path):
    _, definition, loaders = pipeline
    model = PyTorchLinearModelProvider().create(definition)
    deadline = TrainingContext(
        run_id=uuid4(), output_dir=tmp_path, deadline=datetime.now(UTC) - timedelta(seconds=1)
    )
    cancelled = PyTorchLinearTrainer(
        model, loaders, PyTorchLinearTrainingConfig(), deadline
    ).train()
    assert cancelled.status == "cancelled"
    assert cancelled.stop_reason == "deadline_reached"
    assert not cancelled.epochs and cancelled.last_checkpoint is None
    failed = PyTorchLinearTrainer(model, {}, PyTorchLinearTrainingConfig()).train()
    assert failed.status == "failed"
    assert failed.issues[0].exception_type == "KeyError"
    early = PyTorchLinearTrainer(
        model,
        loaders,
        PyTorchLinearTrainingConfig(epochs=10, min_delta=100, early_stopping_patience=1),
    ).train()
    assert early.stop_reason == "early_stopping"
    assert len(early.epochs) == 2
    target = PyTorchLinearTrainer(
        model, loaders, PyTorchLinearTrainingConfig(target_mse=100)
    ).train()
    assert target.stop_reason == "target_metric_reached"
    assert target.last_checkpoint is None


def test_bad_resource_and_config(pipeline, tmp_path):
    _, definition, _ = pipeline
    with pytest.raises(ValueError, match="Expected a local"):
        PyTorchLinearModelProvider().restore_checkpoint(
            definition,
            ResolvedCheckpoint(format="wrong", source_uri="file:///missing", local_path=tmp_path),
        )
    with pytest.raises(ValueError):
        PyTorchLinearTrainingConfig(epochs=0)


def test_native_loader_and_validation_keep_torch_tensors(pipeline):
    import torch
    from bovi_core.ml.dataloaders import PyTorchDataLoader
    from pytorch_linear.trainers.arrays import batch_to_arrays

    _, definition, loaders = pipeline
    assert isinstance(loaders["train"], PyTorchDataLoader)
    batch = next(iter(loaders["train"]))
    assert all(isinstance(column, torch.Tensor) for column in batch["features"].values())
    assert isinstance(batch["labels"], torch.Tensor)
    x, y = batch_to_arrays(batch, definition.feature_names)
    assert isinstance(x, torch.Tensor)
    torch.testing.assert_close(
        x[:, 0], batch["features"][definition.feature_names[0]].to(torch.float32)
    )
    assert isinstance(y, torch.Tensor)
    torch.testing.assert_close(y, batch["labels"].to(torch.float32))


def test_evaluator_reports_an_exception_without_a_message(pipeline, monkeypatch, tmp_path):
    import pytorch_linear.trainers.evaluator as evaluator_module

    def fail(*args):
        raise RuntimeError()

    monkeypatch.setattr(evaluator_module, "measure", fail)
    _, definition, loaders = pipeline
    result = PyTorchLinearEvaluator(
        PyTorchLinearModelProvider().create(definition), PyTorchLinearEvaluationConfig()
    ).evaluate(
        loaders["validation"],
        EvaluationContext(
            evaluation_id=uuid4(), output_dir=tmp_path, split="validation", model_version="test"
        ),
    )
    assert result.status == "failed"
    assert result.issues[0].message == "RuntimeError"
    assert result.issues[0].exception_type == "RuntimeError"
