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
