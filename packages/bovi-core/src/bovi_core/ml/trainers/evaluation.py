from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Generic, Self, TypeVar
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from bovi_core.ml.dataloaders.base import AbstractDataLoader
from bovi_core.ml.models.model import Model

from .config import EvaluationConfig
from .context import EvaluationContext
from .issues import Issue

ModelT = TypeVar("ModelT", bound=Model[Any])
ConfigT = TypeVar("ConfigT", bound=EvaluationConfig)


class EvaluationStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EvaluationArtifactReference(BaseModel):
    """Reference to non-scalar output produced during evaluation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    name: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    checksum: str | None = Field(default=None, min_length=1)


class EvaluationResult(BaseModel):
    """Immutable manifest describing one model evaluation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    evaluation_id: UUID
    status: EvaluationStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime
    num_examples: int = Field(ge=0)
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    issues: tuple[Issue, ...] = ()
    artifacts: tuple[EvaluationArtifactReference, ...] = ()

    @model_validator(mode="after")
    def validate_result_consistency(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        if self.status is EvaluationStatus.COMPLETED and not self.metrics:
            raise ValueError("completed evaluation requires at least one metric")
        return self


class Evaluator(ABC, Generic[ModelT, ConfigT]):
    """Evaluate one Bovi model against one dataloader at a time."""

    def __init__(self, model: ModelT, config: ConfigT) -> None:
        self.model = model
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        dataloader: AbstractDataLoader,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate the configured model for the context's named data split."""
        raise NotImplementedError
