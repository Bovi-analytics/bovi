"""Native Keras trainer for lactation reconstruction."""

from pathlib import Path

import tensorflow as tf
from bovi_core.ml import CheckpointReference, Trainer, TrainingResult
from bovi_core.ml.models.checkpoints import LocalCheckpointStore
from bovi_core.ml.trainers.lifecycle import run_epochs
from bovi_core.ml.trainers.monitoring import MetricMonitor

from lactation_autoencoder.models import (
    LACTATION_WEIGHTS_FORMAT,
    LactationAutoencoderModel,
)

from .arrays import batch_to_tensors, measure
from .config import LactationAutoencoderTrainingConfig


class LactationAutoencoderTrainer(
    Trainer[LactationAutoencoderModel, LactationAutoencoderTrainingConfig]
):
    """Train a Keras model; checkpoints intentionally restore weights only."""

    def train(self) -> TrainingResult:
        def prepare():
            native = self.model.trainable_model
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

            def step(batch):
                inputs, expected = batch_to_tensors(batch)
                with tf.device("/CPU:0"):
                    with tf.GradientTape() as tape:
                        predicted = native(inputs, training=True)
                        loss = tf.reduce_mean(tf.square(predicted - expected))
                    gradients = tape.gradient(loss, native.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, native.trainable_variables))
                return len(expected)

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
            error_code="lactation_autoencoder_training_failed",
        )

    def _save(self, name: str, *, epoch: int | None = None) -> CheckpointReference | None:
        if self.context is None:
            return None

        def write(directory: Path) -> None:
            tf.train.Checkpoint(model=self.model.trainable_model).write(
                str(directory / "model.weights")
            )

        return LocalCheckpointStore(self.context.output_dir / "checkpoints").save(
            name,
            LACTATION_WEIGHTS_FORMAT,
            write,
            entrypoint="model.weights.index",
            metadata={
                "model_config": self.model.config.model_dump(mode="json"),
                "completed_epoch": epoch,
                "run_id": str(self.context.run_id),
            },
        )
