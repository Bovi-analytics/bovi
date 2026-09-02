"""Concrete Bovi Core trainer for scikit-learn's SGDRegressor."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from bovi_core.ml import (
    CheckpointReference,
    EpochResult,
    Issue,
    IssueSeverity,
    Trainer,
    TrainingResult,
    TrainingStatus,
    TrainingStopReason,
)

from scikit_sgd.models import ScikitSGDModel

from .arrays import batch_to_arrays, collect_predictions
from .config import ScikitSGDTrainingConfig

SCIKIT_JOBLIB_FORMAT = "scikit-joblib"


class ScikitSGDTrainer(Trainer[ScikitSGDModel, ScikitSGDTrainingConfig]):
    """Train one SGD regressor incrementally over configured batches and epochs."""

    def train(self) -> TrainingResult:
        started_at = datetime.now(UTC)
        run_id = self.context.run_id if self.context is not None else uuid4()
        epoch_results: list[EpochResult] = []
        best_epoch: int | None = None
        best_metric = float("inf")
        stale_epochs = 0
        last_checkpoint: CheckpointReference | None = None
        best_checkpoint: CheckpointReference | None = None

        try:
            train_loader = self.dataloaders["train"]
            validation_loader = self.dataloaders.get("validation")
            self._configure_estimator()

            status = TrainingStatus.COMPLETED
            stop_reason = TrainingStopReason.MAX_EPOCHS_REACHED

            for epoch in range(1, self.config.epochs + 1):
                if self._deadline_reached():
                    status = TrainingStatus.CANCELLED
                    stop_reason = TrainingStopReason.DEADLINE_REACHED
                    break

                for batch in train_loader:
                    x, y = batch_to_arrays(batch, self.model.config.feature_names)
                    self.model.native_model.partial_fit(x, y)

                train_y, train_predictions = collect_predictions(
                    self.model,
                    train_loader,
                    self.model.config.feature_names,
                )
                metrics = self._regression_metrics(train_y, train_predictions, prefix="train")

                monitored_mse = metrics["train_mse"]
                if validation_loader is not None:
                    validation_y, validation_predictions = collect_predictions(
                        self.model,
                        validation_loader,
                        self.model.config.feature_names,
                    )
                    validation_metrics = self._regression_metrics(
                        validation_y,
                        validation_predictions,
                        prefix="validation",
                    )
                    metrics.update(validation_metrics)
                    monitored_mse = validation_metrics["validation_mse"]

                epoch_results.append(EpochResult(epoch=epoch, metrics=metrics))
                last_checkpoint = self._save_checkpoint("last")

                if monitored_mse < best_metric - self.config.min_delta:
                    best_metric = monitored_mse
                    best_epoch = epoch
                    stale_epochs = 0
                    best_checkpoint = self._save_checkpoint("best")
                else:
                    stale_epochs += 1

                if self.config.target_mse is not None and monitored_mse <= self.config.target_mse:
                    stop_reason = TrainingStopReason.TARGET_METRIC_REACHED
                    break

                patience = self.config.early_stopping_patience
                if patience is not None and stale_epochs >= patience:
                    stop_reason = TrainingStopReason.EARLY_STOPPING
                    break

            return TrainingResult(
                run_id=run_id,
                status=status,
                stop_reason=stop_reason,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                epochs=tuple(epoch_results),
                best_epoch=best_epoch,
                last_checkpoint=last_checkpoint,
                best_checkpoint=best_checkpoint,
            )
        except Exception as exc:
            issue = Issue(
                severity=IssueSeverity.ERROR,
                code="scikit_sgd_training_failed",
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            return TrainingResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                stop_reason=TrainingStopReason.ERROR,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                epochs=tuple(epoch_results),
                issues=(issue,),
                best_epoch=best_epoch,
                last_checkpoint=last_checkpoint,
                best_checkpoint=best_checkpoint,
            )

    def _configure_estimator(self) -> None:
        self.model.native_model.set_params(
            loss=self.config.loss,
            penalty=self.config.penalty,
            alpha=self.config.alpha,
            learning_rate=self.config.learning_rate_schedule,
            eta0=self.config.learning_rate,
            average=self.config.average,
        )

    def _deadline_reached(self) -> bool:
        return (
            self.context is not None
            and self.context.deadline is not None
            and datetime.now(UTC) >= self.context.deadline
        )

    def _save_checkpoint(self, name: str) -> CheckpointReference | None:
        if self.context is None:
            return None

        checkpoint_dir = self.context.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{name}.joblib"
        joblib.dump(self.model.native_model, checkpoint_path)
        return self._checkpoint_reference(checkpoint_path)

    @staticmethod
    def _checkpoint_reference(path: Path) -> CheckpointReference:
        return CheckpointReference(uri=path.resolve().as_uri(), format=SCIKIT_JOBLIB_FORMAT)

    @staticmethod
    def _regression_metrics(
        expected: np.ndarray,
        predicted: np.ndarray,
        prefix: str,
    ) -> dict[str, float]:
        errors = predicted - expected
        return {
            f"{prefix}_mse": float(np.mean(np.square(errors))),
            f"{prefix}_mae": float(np.mean(np.abs(errors))),
        }
