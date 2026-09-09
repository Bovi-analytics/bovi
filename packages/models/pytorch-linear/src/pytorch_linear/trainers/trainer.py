"""Minimal native training loop using the Bovi training contract."""

from pathlib import Path

import torch
from bovi_core.ml import (
    CheckpointReference,
    Trainer,
    TrainingResult,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointStore
from bovi_core.ml.trainers.lifecycle import run_epochs
from bovi_core.ml.trainers.monitoring import MetricMonitor

from ..models import PyTorchLinearModel
from ..models.provider import FORMAT
from .arrays import batch_to_arrays, measure
from .config import PyTorchLinearTrainingConfig


class PyTorchLinearTrainer(Trainer[PyTorchLinearModel, PyTorchLinearTrainingConfig]):
    """Train on CPU; checkpoints support weights-only restart, not exact resume."""

    def train(self) -> TrainingResult:
        def prepare():
            optimizer = torch.optim.SGD(
                self.model.native_model.parameters(), lr=self.config.learning_rate
            )

            def step(batch):
                x, y = batch_to_arrays(batch, self.model.config.feature_names)
                self.model.native_model.train()
                optimizer.zero_grad()
                predicted = self.model.native_model(x).reshape(-1)
                loss = torch.nn.functional.mse_loss(predicted, y)
                loss.backward()
                optimizer.step()
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
            error_code="pytorch_linear_training_failed",
        )

    def _save(self, name: str, *, epoch: int | None = None) -> CheckpointReference | None:
        if self.context is None:
            return None

        def write(directory: Path) -> None:
            torch.save(
                {
                    "state_dict": self.model.native_model.state_dict(),
                    "feature_names": list(self.model.config.feature_names),
                },
                directory / "model.pt",
            )

        return LocalCheckpointStore(self.context.output_dir / "checkpoints").save(
            name,
            FORMAT,
            write,
            entrypoint="model.pt",
            metadata={
                "model_config": self.model.config.model_dump(mode="json"),
                "completed_epoch": epoch,
                "run_id": str(self.context.run_id),
            },
        )
