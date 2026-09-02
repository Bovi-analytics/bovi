from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context import TrainingContext
from .issues import Issue, IssueSeverity
from .results import TrainingResult


class LogDestinationStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class LogIssue(Issue):
    """Diagnostic issue associated with one destination write attempt."""

    write_attempt: int = Field(ge=1)


class LogDestinationResult(BaseModel):
    """Outcome of persisting a training result to one destination."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    destination: str = Field(min_length=1)
    status: LogDestinationStatus
    location: str | None = Field(default=None, min_length=1)
    attempt_count: int = Field(ge=0)
    issues: tuple[LogIssue, ...] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> "LogDestinationResult":
        if self.status is LogDestinationStatus.SUCCESS:
            if self.location is None or self.attempt_count < 1:
                raise ValueError("successful logging requires a location and at least one attempt")
        elif self.status is LogDestinationStatus.FAILED:
            if self.attempt_count < 1:
                raise ValueError("failed logging requires at least one attempt")
            if not any(
                issue.severity in {IssueSeverity.ERROR, IssueSeverity.CRITICAL}
                for issue in self.issues
            ):
                raise ValueError("failed logging requires an error or critical issue")
        elif self.location is not None or self.attempt_count != 0:
            raise ValueError("skipped logging cannot have a location or attempts")

        return self


class ResultLogOutcome(BaseModel):
    """Combined logging outcome for one training run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    run_id: UUID
    destinations: tuple[LogDestinationResult, ...] = Field(min_length=1)


class TrainingResultLogger(Protocol):
    """Persist a training manifest without coupling storage to the trainer."""

    async def log(
        self,
        context: TrainingContext,
        result: TrainingResult,
    ) -> ResultLogOutcome: ...
