from pathlib import Path
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class TrainingContext(BaseModel):
    """Immutable identity and execution metadata for one local training attempt."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    run_id: UUID = Field(description="Unique identifier for this individual training attempt.")
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
    reason: str | None = Field(
        default=None,
        description="Optional explanation for why this training attempt was started.",
    )
    output_dir: Path = Field(
        description="Local directory for artifacts and metadata produced by this attempt."
    )
    deadline: AwareDatetime | None = Field(
        default=None,
        description="Optional timezone-aware deadline for completing this attempt.",
    )
