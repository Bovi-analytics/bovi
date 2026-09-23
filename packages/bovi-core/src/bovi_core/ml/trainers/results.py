from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from bovi_core.ml.models import CheckpointReference

from .issues import Issue


class TrainingStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TrainingStopReason(StrEnum):
    MAX_EPOCHS_REACHED = "max_epochs_reached"
    EARLY_STOPPING = "early_stopping"
    TARGET_METRIC_REACHED = "target_metric_reached"
    DEADLINE_REACHED = "deadline_reached"
    CANCELLED = "cancelled"
    ERROR = "error"


class EpochResult(BaseModel):
    """Metrics recorded after one completed local training epoch."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    epoch: int = Field(
        ge=1,
        description="One-based epoch number within this training attempt.",
    )
    metrics: dict[str, FiniteFloat] = Field(
        min_length=1,
        description="Model-specific scalar metrics recorded for this epoch.",
    )


class TrainingResult(BaseModel):
    """Immutable manifest describing one local training attempt."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    run_id: UUID
    status: TrainingStatus
    stop_reason: TrainingStopReason
    started_at: AwareDatetime
    completed_at: AwareDatetime
    num_examples: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Number of records in the training split, not batches or epoch exposures. "
        "None means unknown; zero means a known empty training split.",
    )
    num_examples_processed: int | None = Field(
        default=None,
        ge=0,
        strict=True,
        description="Training examples consumed by successful training steps in this attempt, "
        "including partial epochs and repeated exposures. Excludes evaluation, metrics-only "
        "passes, prefetched unused records, and previous attempts. None means unknown.",
    )
    epochs: tuple[EpochResult, ...] = ()
    issues: tuple[Issue, ...] = ()
    best_epoch: int | None = Field(default=None, ge=1)
    last_checkpoint: CheckpointReference | None = None
    best_checkpoint: CheckpointReference | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")

        epoch_numbers = [epoch.epoch for epoch in self.epochs]
        if epoch_numbers != sorted(set(epoch_numbers)):
            raise ValueError("epoch numbers must be unique and strictly increasing")

        if self.best_epoch is not None and self.best_epoch not in epoch_numbers:
            raise ValueError("best_epoch must refer to an epoch in epochs")
        if self.best_checkpoint is not None and self.best_epoch is None:
            raise ValueError("best_checkpoint requires best_epoch")

        allowed_stop_reasons = {
            TrainingStatus.COMPLETED: {
                TrainingStopReason.MAX_EPOCHS_REACHED,
                TrainingStopReason.EARLY_STOPPING,
                TrainingStopReason.TARGET_METRIC_REACHED,
            },
            TrainingStatus.CANCELLED: {
                TrainingStopReason.CANCELLED,
                TrainingStopReason.DEADLINE_REACHED,
            },
            TrainingStatus.FAILED: {TrainingStopReason.ERROR},
        }
        if self.stop_reason not in allowed_stop_reasons[self.status]:
            raise ValueError(
                f"stop_reason {self.stop_reason!s} is incompatible with status {self.status!s}"
            )

        return self
