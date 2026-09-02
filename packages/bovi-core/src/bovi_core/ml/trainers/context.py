from pathlib import Path
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ExecutionContext(BaseModel):
    """Shared execution metadata for training and evaluation work."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    reason: str | None = Field(
        default=None,
        description="Optional explanation for why this execution was started.",
    )
    output_dir: Path = Field(
        description="Local directory for artifacts and metadata produced by this execution."
    )
    deadline: AwareDatetime | None = Field(
        default=None,
        description="Optional timezone-aware deadline for completing this execution.",
    )


class TrainingContext(ExecutionContext):
    """Identity and execution metadata for one training attempt."""

    run_id: UUID = Field(description="Unique identifier for this individual training attempt.")
    resumed_from_run_id: UUID | None = Field(
        default=None,
        description="Optional identifier of the earlier training run resumed by this attempt.",
    )


class FederatedTrainingContext(TrainingContext):
    """Federated metadata for one local training attempt on a farm."""

    experiment_id: int = Field(
        ge=1,
        description="Identifier of the federated experiment this attempt belongs to.",
    )
    farm_id: int = Field(
        ge=1,
        description="Identifier of the farm executing the local training attempt.",
    )
    round_id: int = Field(
        ge=1,
        description="One-based federated round number within the experiment.",
    )
    attempt: int = Field(
        ge=1,
        description="One-based attempt number for this farm and federated round.",
    )
    base_global_model_version: str = Field(
        min_length=1,
        description="Global model weight-state used to start this local training attempt.",
    )


class EvaluationContext(ExecutionContext):
    """Identity and execution metadata for one model evaluation."""

    evaluation_id: UUID = Field(description="Unique identifier for this evaluation.")
    split: str = Field(min_length=1, description="Named dataset split being evaluated.")
    model_version: str = Field(
        min_length=1,
        description="Identifier of the model state being evaluated.",
    )
    training_run_id: UUID | None = Field(
        default=None,
        description="Optional training run from which the evaluated model originated.",
    )


class FederatedEvaluationContext(EvaluationContext):
    """Federated metadata for an evaluation performed on a farm."""

    experiment_id: int = Field(
        ge=1,
        description="Identifier of the federated experiment being evaluated.",
    )
    farm_id: int = Field(
        ge=1,
        description="Identifier of the farm executing the evaluation.",
    )
    round_id: int = Field(
        ge=1,
        description="One-based federated round associated with the evaluated model.",
    )
