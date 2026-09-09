"""Minimal native training loop using the Bovi training contract."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import tensorflow as tf
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

from ..models import TensorFlowLinearModel
from ..models.provider import FORMAT
from .arrays import batch_to_arrays, measure
from .config import TensorFlowLinearTrainingConfig


class TensorFlowLinearTrainer(Trainer[TensorFlowLinearModel, TensorFlowLinearTrainingConfig]):
    """Train on CPU; SGD has no momentum or schedule requiring separate resume state."""

    def train(self) -> TrainingResult:
        started = datetime.now(UTC)
        run_id = self.context.run_id if self.context else uuid4()
        epochs = []
        best_epoch = None
        best = float("inf")
        stale = 0
        last_checkpoint = best_checkpoint = None
        status = TrainingStatus.COMPLETED
        stop = TrainingStopReason.MAX_EPOCHS_REACHED
        issues = ()
        try:
            train = self.dataloaders["train"]
            validation = self.dataloaders.get("validation")
            optimizer = self.model.native_model.optimizer
            optimizer.learning_rate.assign(self.config.learning_rate)
            for epoch in range(1, self.config.epochs + 1):
                if self._expired():
                    status, stop = TrainingStatus.CANCELLED, TrainingStopReason.DEADLINE_REACHED
                    break
                for batch in train:
                    if self._expired():
                        break
                    x, y = batch_to_arrays(batch, len(self.model.config.feature_names))
                    with tf.device("/CPU:0"):
                        with tf.GradientTape() as tape:
                            predicted = tf.reshape(self.model.native_model(x, training=True), (-1,))
                            loss = tf.reduce_mean(tf.square(predicted - y))
                        gradients = tape.gradient(loss, self.model.native_model.trainable_variables)
                        optimizer.apply_gradients(
                            zip(gradients, self.model.native_model.trainable_variables)
                        )
                if self._expired():
                    status, stop = TrainingStatus.CANCELLED, TrainingStopReason.DEADLINE_REACHED
                    break
                _, metrics = measure(self.model, train)
                metrics = {"train_" + k: v for k, v in metrics.items()}
                monitored = metrics["train_mse"]
                if validation is not None:
                    _, val_metrics = measure(self.model, validation)
                    metrics.update({"validation_" + k: v for k, v in val_metrics.items()})
                    monitored = val_metrics["mse"]
                result = EpochResult(epoch=epoch, metrics=metrics)
                last_checkpoint = self._save("last")
                epochs.append(result)
                if monitored < best - self.config.min_delta:
                    best, best_epoch, stale = monitored, epoch, 0
                    best_checkpoint = self._save("best")
                else:
                    stale += 1
                if self.config.target_mse is not None and monitored <= self.config.target_mse:
                    stop = TrainingStopReason.TARGET_METRIC_REACHED
                    break
                if (
                    self.config.early_stopping_patience is not None
                    and stale >= self.config.early_stopping_patience
                ):
                    stop = TrainingStopReason.EARLY_STOPPING
                    break
        except Exception as exc:
            status, stop = TrainingStatus.FAILED, TrainingStopReason.ERROR
            issues = (
                Issue(
                    severity=IssueSeverity.ERROR,
                    code="tensorflow_linear_training_failed",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                ),
            )
        return TrainingResult(
            run_id=run_id,
            status=status,
            stop_reason=stop,
            started_at=started,
            completed_at=datetime.now(UTC),
            epochs=tuple(epochs),
            best_epoch=best_epoch,
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint,
            issues=issues,
        )

    def _expired(self) -> bool:
        return (
            self.context is not None
            and self.context.deadline is not None
            and datetime.now(UTC) >= self.context.deadline
        )

    def _save(self, name: str) -> CheckpointReference | None:
        if self.context is None:
            return None
        directory = self.context.output_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".keras")
        temporary = directory / (name + ".tmp.keras")
        self.model.native_model.save(temporary)
        metadata = path.with_suffix(".json")
        metadata.write_text(json.dumps(list(self.model.config.feature_names)))
        temporary.replace(path)
        return CheckpointReference(uri=path.resolve().as_uri(), format=FORMAT)
