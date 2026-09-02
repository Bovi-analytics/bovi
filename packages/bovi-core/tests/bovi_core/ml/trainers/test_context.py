from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from bovi_core.ml.trainers.context import (
    EvaluationContext,
    FederatedEvaluationContext,
    FederatedTrainingContext,
    TrainingContext,
)
from pydantic import ValidationError


def make_context(**overrides: object) -> FederatedTrainingContext:
    values: dict[str, object] = {
        "run_id": uuid4(),
        "experiment_id": 1,
        "farm_id": 2,
        "round_id": 3,
        "attempt": 1,
        "base_global_model_version": "global-v3",
        "output_dir": Path("/tmp/training"),
        "deadline": datetime(2026, 9, 2, tzinfo=UTC),
    }
    values.update(overrides)
    return FederatedTrainingContext.model_validate(values)


def test_training_context_does_not_require_federated_metadata() -> None:
    context = TrainingContext(
        run_id=uuid4(),
        output_dir=Path("/tmp/training"),
    )

    assert context.deadline is None
    assert context.reason is None
    assert context.resumed_from_run_id is None


def test_training_context_records_global_model_and_resumed_run() -> None:
    resumed_from_run_id = uuid4()

    context = make_context(resumed_from_run_id=resumed_from_run_id)

    assert context.base_global_model_version == "global-v3"
    assert context.resumed_from_run_id == resumed_from_run_id


def test_training_context_defaults_resumed_run_to_none() -> None:
    context = make_context()

    assert context.resumed_from_run_id is None


@pytest.mark.parametrize(
    "base_global_model_version",
    [None, ""],
)
def test_training_context_requires_non_empty_global_model_version(
    base_global_model_version: str | None,
) -> None:
    overrides = (
        {}
        if base_global_model_version is None
        else {"base_global_model_version": base_global_model_version}
    )
    values: dict[str, object] = {
        "run_id": UUID("12345678-1234-5678-1234-567812345678"),
        "experiment_id": 1,
        "farm_id": 2,
        "round_id": 3,
        "attempt": 1,
        "output_dir": Path("/tmp/training"),
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        FederatedTrainingContext.model_validate(values)


def test_evaluation_context_does_not_require_federated_metadata() -> None:
    context = EvaluationContext(
        evaluation_id=uuid4(),
        split="test",
        model_version="model-v4",
        output_dir=Path("/tmp/evaluation"),
    )

    assert context.training_run_id is None
    assert context.deadline is None


def test_federated_evaluation_context_requires_complete_federated_metadata() -> None:
    values: dict[str, object] = {
        "evaluation_id": uuid4(),
        "split": "validation",
        "model_version": "global-v4",
        "output_dir": Path("/tmp/evaluation"),
        "experiment_id": 1,
        "farm_id": 2,
        "round_id": 4,
    }

    context = FederatedEvaluationContext.model_validate(values)
    assert context.farm_id == 2

    values.pop("round_id")
    with pytest.raises(ValidationError):
        FederatedEvaluationContext.model_validate(values)
