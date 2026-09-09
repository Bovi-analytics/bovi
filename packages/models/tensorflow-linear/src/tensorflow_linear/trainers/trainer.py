"""Minimal native training loop using the Bovi training contract."""

import json
from pathlib import Path

import tensorflow as tf
from bovi_core.ml import (
    CheckpointReference,
    Trainer,
    TrainingResult,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointStore
from bovi_core.ml.trainers.lifecycle import run_epochs
from bovi_core.ml.trainers.monitoring import MetricMonitor

from ..models import TensorFlowLinearModel
from ..models.provider import FORMAT
from .arrays import batch_to_arrays, measure
from .config import TensorFlowLinearTrainingConfig


class TensorFlowLinearTrainer(Trainer[TensorFlowLinearModel, TensorFlowLinearTrainingConfig]):
    """Train on CPU; checkpoints support weights-only restart, not exact resume."""

    def train(self) -> TrainingResult:
        def prepare():
            optimizer = self.model.native_model.optimizer
            optimizer.learning_rate.assign(self.config.learning_rate)

            def step(batch):
                x, y = batch_to_arrays(batch, self.model.config.feature_names)
                with tf.device("/CPU:0"):
                    with tf.GradientTape() as tape:
                        predicted = tf.reshape(self.model.native_model(x, training=True), (-1,))
                        loss = tf.reduce_mean(tf.square(predicted - y))
                    gradients = tape.gradient(loss, self.model.native_model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, self.model.native_model.trainable_variables)
                    )
                return len(y)

            return step

        return run_epochs(
            context=self.context,
            dataloaders=self.dataloaders,
            epochs=self.config.epochs,
            monitor=MetricMonitor(
                min_delta=self.config.min_delta,
                patience=self.config.early_stopping_patience,
                target=self.config.target_mse,
            ),
            monitor_metric="mse",
            prepare=prepare,
            measure=lambda batches: measure(self.model, batches),
            save=self._save,
            error_code="tensorflow_linear_training_failed",
        )

    def _save(self, name: str, *, epoch: int | None = None) -> CheckpointReference | None:
        if self.context is None:
            return None

        def write(directory: Path) -> None:
            self.model.native_model.save(directory / "model.keras")
            (directory / "model.json").write_text(
                json.dumps(list(self.model.config.feature_names)), encoding="utf-8"
            )

        return LocalCheckpointStore(self.context.output_dir / "checkpoints").save(
            name,
            FORMAT,
            write,
            entrypoint="model.keras",
            metadata={
                "model_config": self.model.config.model_dump(mode="json"),
                "completed_epoch": epoch,
                "run_id": str(self.context.run_id),
            },
        )
