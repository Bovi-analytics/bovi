from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat

type IssueDetail = str | bool | int | FiniteFloat | None


class IssueSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Issue(BaseModel):
    """Serializable diagnostic information produced during training."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    severity: IssueSeverity
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    exception_type: str | None = Field(default=None, min_length=1)
    details: dict[str, IssueDetail] = Field(default_factory=dict)

    @staticmethod
    def from_exception(
        exception: Exception,
        *,
        code: str,
        severity: IssueSeverity = IssueSeverity.ERROR,
        trace_id: str | None = None,
    ) -> "Issue":
        """Describe an operational failure without relying on its message being usable.

        Tracebacks belong in local diagnostic logs; a trace ID can link them
        without including stack frames or runtime values in a remote result.
        """
        exception_type = type(exception).__name__
        try:
            message = str(exception).strip()
        except Exception:
            message = ""
        return Issue(
            severity=severity,
            code=code,
            message=message or exception_type,
            exception_type=exception_type,
            details={"trace_id": trace_id} if trace_id else {},
        )
