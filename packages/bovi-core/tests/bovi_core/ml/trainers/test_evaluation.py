from datetime import UTC, datetime, timedelta
from inspect import isabstract
from uuid import uuid4

import pytest
from bovi_core.ml.trainers.evaluation import (
    EvaluationArtifactReference,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
)
from pydantic import ValidationError


def make_evaluation_result(**overrides: object) -> EvaluationResult:
    started_at = datetime(2026, 9, 2, 12, tzinfo=UTC)
    values: dict[str, object] = {
        "evaluation_id": uuid4(),
        "status": EvaluationStatus.COMPLETED,
        "started_at": started_at,
        "completed_at": started_at + timedelta(minutes=2),
        "num_examples": 100,
        "metrics": {"loss": 0.2, "rmse": 0.4},
        "artifacts": (
            EvaluationArtifactReference(
                name="confusion_matrix",
                uri="file:///tmp/confusion-matrix.json",
                media_type="application/json",
            ),
        ),
    }
    values.update(overrides)
    return EvaluationResult.model_validate(values)


def test_evaluation_result_records_metrics_and_artifacts() -> None:
    result = make_evaluation_result()

    assert result.num_examples == 100
    assert result.metrics["rmse"] == 0.4
    assert result.artifacts[0].name == "confusion_matrix"
    assert result.model_dump(mode="json")["status"] == "completed"


def test_failed_evaluation_can_end_without_metrics() -> None:
    result = make_evaluation_result(
        status=EvaluationStatus.FAILED,
        metrics={},
        artifacts=(),
    )

    assert result.metrics == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics": {}},
        {"completed_at": datetime(2026, 9, 2, 11, tzinfo=UTC)},
        {"num_examples": -1},
    ],
)
def test_evaluation_result_rejects_inconsistent_manifest(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        make_evaluation_result(**overrides)


def test_evaluator_is_an_abstract_contract() -> None:
    assert isabstract(Evaluator)
    assert Evaluator.__abstractmethods__ == frozenset({"evaluate"})
