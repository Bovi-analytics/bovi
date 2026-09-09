"""End-to-end contract tests for training, checkpointing, resume, and evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from bovi_core.config import Config
from bovi_core.ml import (
    EvaluationContext,
    EvaluationStatus,
    TrainingContext,
    TrainingStatus,
    TrainingStopReason,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from scikit_sgd import (
    ScikitSGDEvaluationConfig,
    ScikitSGDEvaluator,
    ScikitSGDModel,
    ScikitSGDModelConfig,
    ScikitSGDModelProvider,
    ScikitSGDTrainer,
    ScikitSGDTrainingConfig,
)


def test_trainer_returns_epoch_history_and_checkpoint_references(
    experiment_config: Config,
    model: ScikitSGDModel,
    dataloaders: dict,
    output_dir: Path,
) -> None:
    context = TrainingContext(run_id=uuid4(), output_dir=output_dir)
    config = ScikitSGDTrainingConfig.from_config(experiment_config).model_copy(
        update={"epochs": 5, "early_stopping_patience": None}
    )

    result = ScikitSGDTrainer(model, dataloaders, config, context).train()

    assert result.status is TrainingStatus.COMPLETED
    assert result.stop_reason is TrainingStopReason.MAX_EPOCHS_REACHED
    assert len(result.epochs) == 5
    assert result.epochs[-1].metrics["validation_mse"] < result.epochs[0].metrics["validation_mse"]
    assert result.best_epoch is not None
    assert result.last_checkpoint is not None
    assert result.best_checkpoint is not None
    last_path = LocalCheckpointResolver().resolve(result.last_checkpoint).local_path
    best_path = LocalCheckpointResolver().resolve(result.best_checkpoint).local_path
    assert last_path is not None
    assert best_path is not None
    assert last_path.is_file()
    assert best_path.is_file()


def test_expired_deadline_cancels_before_first_epoch(
    experiment_config: Config,
    model: ScikitSGDModel,
    dataloaders: dict,
    output_dir: Path,
) -> None:
    context = TrainingContext(
        run_id=uuid4(),
        output_dir=output_dir,
        deadline=datetime.now(UTC) - timedelta(seconds=1),
    )

    result = ScikitSGDTrainer(
        model,
        dataloaders,
        ScikitSGDTrainingConfig.from_config(experiment_config),
        context,
    ).train()

    assert result.status is TrainingStatus.CANCELLED
    assert result.stop_reason is TrainingStopReason.DEADLINE_REACHED
    assert result.epochs == ()


def test_early_stopping_uses_the_monitored_validation_mse(
    experiment_config: Config,
    model: ScikitSGDModel,
    dataloaders: dict,
    output_dir: Path,
) -> None:
    config = ScikitSGDTrainingConfig.from_config(experiment_config).model_copy(
        update={"epochs": 10, "early_stopping_patience": 1, "min_delta": 1_000_000.0}
    )

    result = ScikitSGDTrainer(
        model,
        dataloaders,
        config,
        TrainingContext(run_id=uuid4(), output_dir=output_dir),
    ).train()

    assert result.status is TrainingStatus.COMPLETED
    assert result.stop_reason is TrainingStopReason.EARLY_STOPPING
    assert len(result.epochs) == 2


def test_checkpoint_can_be_restored_for_a_new_attempt(
    experiment_config: Config,
    model: ScikitSGDModel,
    model_config: ScikitSGDModelConfig,
    dataloaders: dict,
    output_dir: Path,
) -> None:
    first_run_id = uuid4()
    training_config = ScikitSGDTrainingConfig.from_config(experiment_config).model_copy(
        update={"epochs": 2, "early_stopping_patience": None}
    )
    first_result = ScikitSGDTrainer(
        model,
        dataloaders,
        training_config,
        TrainingContext(run_id=first_run_id, output_dir=output_dir / "first"),
    ).train()
    assert first_result.last_checkpoint is not None

    restored = ScikitSGDModelProvider().restore_checkpoint(
        model_config,
        LocalCheckpointResolver().resolve(first_result.last_checkpoint),
    )
    second_result = ScikitSGDTrainer(
        restored,
        dataloaders,
        training_config,
        TrainingContext(
            run_id=uuid4(),
            resumed_from_run_id=first_run_id,
            output_dir=output_dir / "second",
        ),
    ).train()

    assert second_result.status is TrainingStatus.COMPLETED
    assert second_result.epochs[0].epoch == 1


def test_evaluator_returns_configured_metrics(
    experiment_config: Config,
    model: ScikitSGDModel,
    dataloaders: dict,
    output_dir: Path,
) -> None:
    run_id = uuid4()
    training_result = ScikitSGDTrainer(
        model,
        dataloaders,
        ScikitSGDTrainingConfig.from_config(experiment_config).model_copy(
            update={"epochs": 10, "early_stopping_patience": None}
        ),
        TrainingContext(run_id=run_id, output_dir=output_dir),
    ).train()
    assert training_result.status is TrainingStatus.COMPLETED

    evaluation = ScikitSGDEvaluator(
        model,
        ScikitSGDEvaluationConfig.from_config(experiment_config),
    ).evaluate(
        dataloaders["validation"],
        EvaluationContext(
            evaluation_id=uuid4(),
            split="validation",
            model_version="test",
            training_run_id=run_id,
            output_dir=output_dir / "evaluation",
        ),
    )

    assert evaluation.status is EvaluationStatus.COMPLETED
    assert evaluation.num_examples == 6
    assert set(evaluation.metrics) == {"mse", "mae", "r2"}
    assert evaluation.metrics["r2"] > 0.5


def test_evaluator_reports_an_exception_without_a_message(
    experiment_config, model, dataloaders, output_dir, monkeypatch
):
    import scikit_sgd.trainers.evaluator as evaluator_module

    def fail(*args):
        raise RuntimeError()

    monkeypatch.setattr(evaluator_module, "collect_predictions", fail)
    result = ScikitSGDEvaluator(
        model, ScikitSGDEvaluationConfig.from_config(experiment_config)
    ).evaluate(
        dataloaders["validation"],
        EvaluationContext(
            evaluation_id=uuid4(), output_dir=output_dir, split="validation", model_version="test"
        ),
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.issues[0].message == "RuntimeError"
    assert result.issues[0].exception_type == "RuntimeError"
