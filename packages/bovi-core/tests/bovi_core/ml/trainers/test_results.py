import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from bovi_core.ml.trainers.issues import Issue, IssueSeverity
from bovi_core.ml.trainers.results import (
    CheckpointReference,
    EpochResult,
    TrainingResult,
    TrainingStatus,
    TrainingStopReason,
)
from pydantic import ValidationError


def test_epoch_result_accepts_model_specific_metrics() -> None:
    result = EpochResult(
        epoch=3,
        metrics={
            "train/loss": 0.18,
            "validation/loss": 0.21,
            "validation/mae": 0.09,
        },
    )

    assert result.epoch == 3
    assert result.metrics == {
        "train/loss": 0.18,
        "validation/loss": 0.21,
        "validation/mae": 0.09,
    }


@pytest.mark.parametrize(
    ("epoch", "metrics"),
    [
        (0, {"train/loss": 0.18}),
        (1, {}),
        (1, {"train/loss": math.nan}),
        (1, {"train/loss": math.inf}),
    ],
)
def test_epoch_result_rejects_invalid_values(
    epoch: int,
    metrics: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        EpochResult(epoch=epoch, metrics=metrics)


def test_issue_defaults_to_aware_timestamp_and_serializable_details() -> None:
    issue = Issue(
        severity=IssueSeverity.WARNING,
        code="training.slow_epoch",
        message="The epoch exceeded its expected duration.",
        details={"epoch": 2, "duration_seconds": 14.5},
    )

    assert issue.occurred_at.tzinfo is not None
    assert issue.details["epoch"] == 2
    assert '"severity":"warning"' in issue.model_dump_json()


def make_training_result(**overrides: object) -> TrainingResult:
    started_at = datetime(2026, 9, 2, 12, tzinfo=UTC)
    values: dict[str, object] = {
        "run_id": uuid4(),
        "status": TrainingStatus.COMPLETED,
        "stop_reason": TrainingStopReason.EARLY_STOPPING,
        "started_at": started_at,
        "completed_at": started_at + timedelta(minutes=5),
        "epochs": (
            EpochResult(epoch=1, metrics={"train/loss": 0.3}),
            EpochResult(epoch=2, metrics={"train/loss": 0.2}),
        ),
        "best_epoch": 2,
        "best_checkpoint": CheckpointReference(
            uri="file:///tmp/best.ckpt",
            format="pytorch-state-dict",
        ),
    }
    values.update(overrides)
    return TrainingResult.model_validate(values)


def test_training_result_records_complete_epoch_history() -> None:
    result = make_training_result()

    assert [epoch.epoch for epoch in result.epochs] == [1, 2]
    assert result.best_epoch == 2
    assert result.best_checkpoint is not None
    assert result.model_dump(mode="json")["status"] == "completed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"completed_at": datetime(2026, 9, 2, 11, tzinfo=UTC)},
        {
            "epochs": (
                EpochResult(epoch=2, metrics={"train/loss": 0.2}),
                EpochResult(epoch=1, metrics={"train/loss": 0.3}),
            )
        },
        {"best_epoch": 3},
        {"best_epoch": None},
        {
            "status": TrainingStatus.FAILED,
            "stop_reason": TrainingStopReason.EARLY_STOPPING,
        },
    ],
)
def test_training_result_rejects_inconsistent_manifest(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_training_result(**overrides)
