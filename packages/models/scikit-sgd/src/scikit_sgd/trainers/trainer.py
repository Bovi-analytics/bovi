"""Concrete Bovi Core trainer for scikit-learn's SGDRegressor."""

from __future__ import annotations

from pathlib import Path

import joblib
from bovi_core.ml import (
    CheckpointReference,
    Trainer,
    TrainingResult,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointStore
from bovi_core.ml.trainers.lifecycle import run_epochs
from bovi_core.ml.trainers.monitoring import MetricMonitor

from scikit_sgd.models import ScikitSGDModel

from .arrays import batch_to_arrays, measure
from .config import ScikitSGDTrainingConfig

SCIKIT_JOBLIB_FORMAT = "scikit-joblib"


class ScikitSGDTrainer(Trainer[ScikitSGDModel, ScikitSGDTrainingConfig]):
    """Train one SGD regressor incrementally over configured batches and epochs."""

    def train(self) -> TrainingResult:
        def prepare():
            self._configure_estimator()

            def step(batch):
                x, y = batch_to_arrays(batch, self.model.config.feature_names)
                self.model.native_model.partial_fit(x, y)
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
            save=self._save_checkpoint,
            error_code="scikit_sgd_training_failed",
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

    def _save_checkpoint(
        self, name: str, *, epoch: int | None = None
    ) -> CheckpointReference | None:
        if self.context is None:
            return None

        def write(directory: Path) -> None:
            joblib.dump(
                {
                    "estimator": self.model.native_model,
                    "feature_names": list(self.model.config.feature_names),
                },
                directory / "model.joblib",
            )

        return LocalCheckpointStore(self.context.output_dir / "checkpoints").save(
            name,
            SCIKIT_JOBLIB_FORMAT,
            write,
            entrypoint="model.joblib",
            metadata={
                "model_config": self.model.config.model_dump(mode="json"),
                "completed_epoch": epoch,
                "run_id": str(self.context.run_id),
            },
        )
