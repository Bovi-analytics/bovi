from uuid import uuid4

import pytest
from bovi_core.ml.trainers.issues import IssueSeverity
from bovi_core.ml.trainers.logging import (
    LogDestinationResult,
    LogDestinationStatus,
    LogIssue,
    ResultLogOutcome,
)
from pydantic import ValidationError


def test_result_log_outcome_records_results_per_destination() -> None:
    run_id = uuid4()
    outcome = ResultLogOutcome(
        run_id=run_id,
        destinations=(
            LogDestinationResult(
                destination="local",
                status=LogDestinationStatus.SUCCESS,
                location="file:///tmp/training_result.json",
                attempt_count=1,
            ),
            LogDestinationResult(
                destination="azure-blob",
                status=LogDestinationStatus.FAILED,
                attempt_count=3,
                issues=(
                    LogIssue(
                        severity=IssueSeverity.ERROR,
                        code="logging.cloud_timeout",
                        message="Uploading the manifest timed out.",
                        exception_type="TimeoutError",
                        write_attempt=3,
                    ),
                ),
            ),
        ),
    )

    assert outcome.run_id == run_id
    assert outcome.destinations[0].location == "file:///tmp/training_result.json"
    assert outcome.destinations[1].issues[0].write_attempt == 3


@pytest.mark.parametrize(
    "values",
    [
        {
            "destination": "local",
            "status": LogDestinationStatus.SUCCESS,
            "attempt_count": 1,
        },
        {
            "destination": "cloud",
            "status": LogDestinationStatus.FAILED,
            "attempt_count": 1,
        },
        {
            "destination": "cloud",
            "status": LogDestinationStatus.SKIPPED,
            "location": "azure://container/result.json",
            "attempt_count": 0,
        },
    ],
)
def test_log_destination_rejects_inconsistent_outcome(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LogDestinationResult.model_validate(values)
