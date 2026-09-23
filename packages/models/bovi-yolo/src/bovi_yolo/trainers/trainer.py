"""Bovi trainer adapter around the native Ultralytics training lifecycle."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from bovi_core.ml import (
    CheckpointReference,
    Issue,
    Trainer,
    TrainingResult,
    TrainingStatus,
    TrainingStopReason,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointStore
from bovi_core.ml.trainers.lifecycle import DeadlineReached, check_deadline

from ..models import YOLOModel
from ..models.provider import ULTRALYTICS_PT_FORMAT
from .config import YOLOTrainingConfig
from .native import best_epoch, dataset_size, native_save_dir, read_epoch_results


class YOLOTrainer(Trainer[YOLOModel, YOLOTrainingConfig]):
    """Train detection models through Ultralytics' labelled-dataset API.

    The Bovi YOLO dataloader is intentionally inference-oriented and has no
    bounding-box targets. Ultralytics therefore remains responsible for reading
    the detection dataset YAML, labels, augmentation, batching, and native loss.
    """

    def train(self) -> TrainingResult:
        started_at = datetime.now(UTC)
        run_id = self.context.run_id if self.context else uuid4()
        try:
            check_deadline(self.context)
            with self._native_output_dir() as output_dir:
                metrics = self.model.native_model.train(
                    data=str(self.config.dataset_yaml_path),
                    epochs=self.config.epochs,
                    imgsz=self.config.image_size,
                    batch=self.config.batch_size,
                    device=self.config.device,
                    workers=self.config.workers,
                    patience=self.config.patience,
                    optimizer=self.config.optimizer,
                    lr0=self.config.initial_learning_rate,
                    seed=self.config.seed,
                    deterministic=self.config.deterministic,
                    pretrained=self.config.reuse_model_weights,
                    resume=self.config.resume,
                    plots=self.config.plots,
                    verbose=self.config.verbose,
                    project=str(output_dir),
                    name=str(run_id),
                    exist_ok=True,
                    save=True,
                )
                check_deadline(self.context)
                trainer = cast(Any, self.model.native_model.trainer)
                save_dir = native_save_dir(self.model.native_model)
                history = read_epoch_results(save_dir / "results.csv", metrics)
                selected_best_epoch = best_epoch(history)
                last_epoch = history[-1].epoch if history else None
                last = self._store_checkpoint(Path(trainer.last), "last", last_epoch)
                best = self._store_checkpoint(Path(trainer.best), "best", selected_best_epoch)
                stopped_early = len(history) < self.config.epochs
                num_examples = dataset_size(trainer, "train_loader")
                return TrainingResult(
                    run_id=run_id,
                    status=TrainingStatus.COMPLETED,
                    stop_reason=(
                        TrainingStopReason.EARLY_STOPPING
                        if stopped_early
                        else TrainingStopReason.MAX_EPOCHS_REACHED
                    ),
                    started_at=started_at,
                    completed_at=datetime.now(UTC),
                    num_examples=num_examples,
                    num_examples_processed=(
                        num_examples * len(history) if num_examples is not None else None
                    ),
                    epochs=history,
                    best_epoch=selected_best_epoch,
                    last_checkpoint=last,
                    best_checkpoint=best,
                )
        except DeadlineReached:
            status = TrainingStatus.CANCELLED
            stop_reason = TrainingStopReason.DEADLINE_REACHED
            issues: tuple[Issue, ...] = ()
        except Exception as exception:
            status = TrainingStatus.FAILED
            stop_reason = TrainingStopReason.ERROR
            issues = (Issue.from_exception(exception, code="yolo_training_failed"),)

        return TrainingResult(
            run_id=run_id,
            status=status,
            stop_reason=stop_reason,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            issues=issues,
        )

    def _store_checkpoint(
        self,
        source: Path,
        name: str,
        completed_epoch: int | None,
    ) -> CheckpointReference | None:
        if self.context is None or not source.is_file():
            return None

        def write(directory: Path) -> None:
            shutil.copy2(source, directory / "model.pt")

        return LocalCheckpointStore(self.context.output_dir / "checkpoints").save(
            name,
            ULTRALYTICS_PT_FORMAT,
            write,
            entrypoint="model.pt",
            metadata={
                "model_config": self.model.config.model_dump(mode="json"),
                "completed_epoch": completed_epoch,
                "run_id": str(self.context.run_id),
                "native_checkpoint": name,
            },
        )

    def _native_output_dir(self):
        if self.context is not None:
            output_dir = self.context.output_dir / "ultralytics"
            output_dir.mkdir(parents=True, exist_ok=True)
            return _ExistingDirectory(output_dir)
        return tempfile.TemporaryDirectory(prefix="bovi-yolo-")


class _ExistingDirectory:
    """Context manager matching TemporaryDirectory without deleting output."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *args: object) -> None:
        return None
