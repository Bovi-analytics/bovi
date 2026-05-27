"""SQLModel models - single source of truth for DB tables AND Pydantic schemas."""

from datetime import datetime
from typing import ClassVar

from sqlalchemy import JSON, Column, DateTime, UniqueConstraint
from sqlalchemy import func as sa_func
from sqlmodel import Field, SQLModel


class FittingResultBase(SQLModel):
    """Shared fields for fitting results (used in create requests and responses)."""

    model_type: str = Field(
        index=True, description="Model used (e.g. 'wood', 'milkbot', 'autoencoder')"
    )
    source_app: str = Field(index=True, description="Which backend app produced this result")
    input_data: dict = Field(
        default_factory=dict, sa_column=Column(JSON), description="The original request payload"
    )
    output_data: dict = Field(
        default_factory=dict, sa_column=Column(JSON), description="The prediction/fitting response"
    )
    metadata_extra: dict = Field(
        default_factory=dict, sa_column=Column(JSON), description="Additional metadata"
    )


class FittingResult(FittingResultBase, table=True):
    """Database table for fitting results."""

    __tablename__: ClassVar[str] = "fitting_results"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class FittingResultCreate(FittingResultBase):
    """Request body for storing a fitting result (no id or created_at)."""


class FittingResultRead(FittingResultBase):
    """Response body for a fitting result (includes id and created_at)."""

    id: int
    created_at: datetime


class HerdProfileBase(SQLModel):
    """Shared fields for herd profiles."""

    name: str = Field(max_length=100, description="User-given name for this profile")
    description: str = Field(default="", max_length=500)
    achieved_21_milk: float = Field(ge=0.0, le=1.0)
    achieved_305_milk: float = Field(ge=0.0, le=1.0)
    achieved_75_milk: float = Field(ge=0.0, le=1.0)
    achieved_milk: float = Field(ge=0.0, le=1.0)
    days_dry: float = Field(ge=0.0, le=1.0)
    days_in_milk: float = Field(ge=0.0, le=1.0)
    days_open: float = Field(ge=0.0, le=1.0)
    days_pregnant: float = Field(ge=0.0, le=1.0)
    historic_calving_interval: float = Field(ge=0.0, le=1.0)
    quality_sequence: float = Field(ge=0.0, le=1.0)


class HerdProfile(HerdProfileBase, table=True):
    """Database table for user-managed herd stat profiles."""

    __tablename__: ClassVar[str] = "herd_profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_herd_profile_name"),)

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=sa_func.now(),
            onupdate=sa_func.now(),
        ),
    )


class HerdProfileCreate(HerdProfileBase):
    """Request body for creating or updating a herd profile."""


class HerdProfileRead(HerdProfileBase):
    """Response body (includes auto-assigned fields; timestamps may be None in SQLite)."""

    id: int
    created_at: datetime | None  # None only when DB does not fill server default (e.g. SQLite)
    updated_at: datetime | None


# --- Benchmark models ---


class ChallengeBase(SQLModel):
    """Shared fields for benchmark challenges."""

    dataset: str = Field(description="'icar' or 'upload'")
    size: str = Field(default="full", description="'full' for ICAR, 'custom' for upload")
    period: str = Field(default="all", description="'all' for ICAR, 'custom' for upload")
    user_id: str | None = Field(
        default=None, description="Auth-ready; nullable until auth is added"
    )


class Challenge(ChallengeBase, table=True):
    """A benchmark challenge: a sampled set of cows with pre-computed reference yields."""

    __tablename__: ClassVar[str] = "challenges"

    id: int | None = Field(default=None, primary_key=True)
    cow_metadata: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: {parity, herd_id, dim[], milk_kg[]}} - test-day records per cow",
    )
    reference_yields: dict | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Legacy: TIM-calculated 305-day reference yields. Unused in v2 challenges.",
    )
    actual_yields: dict | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="{cow_id: float} - actual cumulative yield (ground truth).",
    )
    name: str | None = Field(
        default=None,
        description="Optional cohort name (used for upload-mode challenges).",
    )
    source: str | None = Field(
        default=None,
        description="'preset' or 'upload' - where the cohort came from.",
    )
    dataset_sources: list[dict] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Source files or source labels used to construct the challenge dataset.",
    )
    dataset_stats: dict | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Small dataset summary for challenge list/detail views.",
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class ChallengeRead(ChallengeBase):
    """API response for a challenge (excludes large internal blobs for list views)."""

    id: int
    created_at: datetime | None
    name: str | None = None
    source: str | None = None
    dataset_sources: list[dict] = Field(default_factory=list)
    dataset_stats: dict = Field(default_factory=dict)


class ChallengeDetail(ChallengeRead):
    """Full challenge response including cow data (used for export and submission)."""

    cow_metadata: dict
    reference_yields: dict | None = None
    actual_yields: dict | None = None


class SubmissionBase(SQLModel):
    """Shared fields for benchmark submissions."""

    submission_type: str = Field(description="'bovi_model' or 'own_method'")
    model_type: str | None = Field(default=None, description="Challenger model: 'tim', 'wood', ...")
    benchmark_model: str | None = Field(
        default=None, description="Server-run benchmark model the challenger is compared against."
    )
    organization: str | None = Field(default=None)
    country: str | None = Field(default=None)
    calculation_method: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    run_options: dict | None = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=True),
        description="Model run options, e.g. MilkBot fitting configuration.",
    )


class Submission(SubmissionBase, table=True):
    """A user's submission for a benchmark challenge."""

    __tablename__: ClassVar[str] = "submissions"

    id: int | None = Field(default=None, primary_key=True)
    challenge_id: int = Field(foreign_key="challenges.id")
    submitted_yields: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: float} - user-submitted or bovi-calculated yields",
    )
    bovi_yields: dict = Field(
        sa_column=Column(JSON),
        description="{cow_id: float} - server-run benchmark-model yields.",
    )
    stats: dict = Field(
        sa_column=Column(JSON),
        description="Comparison statistics (Pearson, RMSE, MAE, MAPE per parity)",
    )
    failed_cow_ids: list = Field(
        sa_column=Column(JSON),
        default_factory=list,
        description="Cow IDs excluded from stats due to parse/compute failure",
    )
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class SubmissionRead(SubmissionBase):
    """API response for a submission."""

    id: int
    challenge_id: int
    stats: dict
    failed_cow_ids: list
    created_at: datetime | None
