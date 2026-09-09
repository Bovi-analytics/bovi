"""tensorflow-linear CPU integration: config, data, training, checkpoints and evaluation."""

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
from tensorflow_linear import (
    TensorFlowLinearEvaluationConfig,
    TensorFlowLinearEvaluator,
    TensorFlowLinearModelProvider,
    TensorFlowLinearTrainer,
    TensorFlowLinearTrainingConfig,
)

pytestmark = pytest.mark.tensorflow


def test_training_evaluation_and_resume(pipeline, tmp_path):
    config, definition, loaders = pipeline
    provider = TensorFlowLinearModelProvider()
    assert ModelProviderRegistry.get("tensorflow_linear") is TensorFlowLinearModelProvider
    model = provider.create(definition)
    run = TrainingContext(run_id=uuid4(), output_dir=tmp_path / "first")
    trainer = TensorFlowLinearTrainer(
        model, loaders, TensorFlowLinearTrainingConfig.from_config(config), run
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
    evaluation = TensorFlowLinearEvaluator(
        restored, TensorFlowLinearEvaluationConfig.from_config(config)
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
    resumed = TensorFlowLinearTrainer(
        restored,
        loaders,
        TensorFlowLinearTrainingConfig(epochs=2),
        TrainingContext(
            run_id=uuid4(), resumed_from_run_id=run.run_id, output_dir=tmp_path / "resume"
        ),
    ).train()
    assert resumed.status == "completed"
    assert resumed.epochs[0].epoch == 1


def test_split_run_matches_continuous_training(pipeline, tmp_path):
    _, definition, loaders = pipeline
    provider = TensorFlowLinearModelProvider()
    continuous = provider.create(definition)
    assert (
        TensorFlowLinearTrainer(continuous, loaders, TensorFlowLinearTrainingConfig(epochs=10))
        .train()
        .status
        == "completed"
    )
    first = provider.create(definition)
    result = TensorFlowLinearTrainer(
        first,
        loaders,
        TensorFlowLinearTrainingConfig(epochs=5),
        TrainingContext(run_id=uuid4(), output_dir=tmp_path),
    ).train()
    assert result.last_checkpoint is not None
    resumed = provider.restore_checkpoint(
        definition,
        LocalCheckpointResolver().resolve(result.last_checkpoint),
    )
    assert (
        TensorFlowLinearTrainer(resumed, loaders, TensorFlowLinearTrainingConfig(epochs=5))
        .train()
        .status
        == "completed"
    )
    np.testing.assert_allclose(resumed([[-0.5], [0.5]]), continuous([[-0.5], [0.5]]), atol=1e-6)


def test_stopping_and_failed_data(pipeline, tmp_path):
    _, definition, loaders = pipeline
    model = TensorFlowLinearModelProvider().create(definition)
    deadline = TrainingContext(
        run_id=uuid4(), output_dir=tmp_path, deadline=datetime.now(UTC) - timedelta(seconds=1)
    )
    cancelled = TensorFlowLinearTrainer(
        model, loaders, TensorFlowLinearTrainingConfig(), deadline
    ).train()
    assert cancelled.status == "cancelled"
    assert cancelled.stop_reason == "deadline_reached"
    assert not cancelled.epochs and cancelled.last_checkpoint is None
    failed = TensorFlowLinearTrainer(model, {}, TensorFlowLinearTrainingConfig()).train()
    assert failed.status == "failed"
    assert failed.issues[0].exception_type == "KeyError"
    early = TensorFlowLinearTrainer(
        model,
        loaders,
        TensorFlowLinearTrainingConfig(epochs=10, min_delta=100, early_stopping_patience=1),
    ).train()
    assert early.stop_reason == "early_stopping"
    assert len(early.epochs) == 2
    target = TensorFlowLinearTrainer(
        model, loaders, TensorFlowLinearTrainingConfig(target_mse=100)
    ).train()
    assert target.stop_reason == "target_metric_reached"
    assert target.last_checkpoint is None


def test_bad_resource_and_config(pipeline, tmp_path):
    _, definition, _ = pipeline
    with pytest.raises(ValueError, match="Expected a local"):
        TensorFlowLinearModelProvider().restore_checkpoint(
            definition,
            ResolvedCheckpoint(format="wrong", source_uri="file:///missing", local_path=tmp_path),
        )
    with pytest.raises(ValueError):
        TensorFlowLinearTrainingConfig(epochs=0)


def test_native_loader_and_validation_keep_tensorflow_tensors(pipeline):
    import tensorflow as tf
    from bovi_core.ml.dataloaders import TensorFlowDataLoader
    from tensorflow_linear.trainers.arrays import batch_to_arrays

    _, definition, loaders = pipeline
    assert isinstance(loaders["train"], TensorFlowDataLoader)
    batch = next(iter(loaders["train"]))
    assert all(isinstance(column, tf.Tensor) for column in batch["features"].values())
    assert isinstance(batch["labels"], tf.Tensor)
    x, y = batch_to_arrays(batch, definition.feature_names)
    assert isinstance(x, tf.Tensor)
    np.testing.assert_array_equal(x[:, 0].numpy(), batch["features"][definition.feature_names[0]])
    assert isinstance(y, tf.Tensor)
    np.testing.assert_array_equal(y.numpy(), batch["labels"].numpy())


def test_evaluator_reports_an_exception_without_a_message(pipeline, monkeypatch, tmp_path):
    import tensorflow_linear.trainers.evaluator as evaluator_module

    def fail(*args):
        raise RuntimeError()

    monkeypatch.setattr(evaluator_module, "measure", fail)
    _, definition, loaders = pipeline
    result = TensorFlowLinearEvaluator(
        TensorFlowLinearModelProvider().create(definition), TensorFlowLinearEvaluationConfig()
    ).evaluate(
        loaders["validation"],
        EvaluationContext(
            evaluation_id=uuid4(), output_dir=tmp_path, split="validation", model_version="test"
        ),
    )
    assert result.status == "failed"
    assert result.issues[0].message == "RuntimeError"
    assert result.issues[0].exception_type == "RuntimeError"
