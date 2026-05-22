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
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_herd_profile_org_name"),)

    id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(default=None, foreign_key="users.id", index=True)
    organization_id: int | None = Field(default=None, foreign_key="organizations.id", index=True)
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

    organization_id: int


class HerdProfileRead(HerdProfileBase):
    """Response body (includes auto-assigned fields; timestamps may be None in SQLite)."""

    id: int
    user_id: int | None = None
    organization_id: int | None = None
    created_at: datetime | None  # None only when DB does not fill server default (e.g. SQLite)
    updated_at: datetime | None


# --- Auth and organization models ---


class UserBase(SQLModel):
    """Local Bovi user linked to a Microsoft Entra identity."""

    entra_tenant_id: str = Field(index=True)
    entra_oid: str = Field(index=True)
    account_type: str = Field(default="entra", index=True)
    email: str | None = Field(default=None, index=True)
    name: str | None = Field(default=None)
    role: str = Field(default="User", index=True)


class User(UserBase, table=True):
    """Application user used for ownership, organizations, and audit trails."""

    __tablename__: ClassVar[str] = "users"
    __table_args__ = (
        UniqueConstraint("entra_tenant_id", "entra_oid", name="uq_user_entra_identity"),
    )

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
    last_login_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))


class OrganizationBase(SQLModel):
    """Organization that owns shared Bovi records."""

    name: str = Field(max_length=200)


class Organization(OrganizationBase, table=True):
    """Organization/team boundary for shared benchmark records."""

    __tablename__: ClassVar[str] = "organizations"

    id: int | None = Field(default=None, primary_key=True)
    created_by_user_id: int | None = Field(default=None, foreign_key="users.id")
    source_entra_tenant_id: str | None = Field(default=None, index=True)
    source_domain: str | None = Field(default=None, max_length=320)
    source_display_name: str | None = Field(default=None, max_length=200)
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


class OrganizationMembershipBase(SQLModel):
    """Membership and organization-local role for a user."""

    user_id: int = Field(foreign_key="users.id", index=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True)
    role: str = Field(default="Member", index=True)


class OrganizationMembership(OrganizationMembershipBase, table=True):
    """Join table between users and organizations."""

    __tablename__: ClassVar[str] = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_organization_membership"),
    )

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )


class OrganizationInvite(SQLModel, table=True):
    """Reusable invite link for joining an organization."""

    __tablename__: ClassVar[str] = "organization_invites"

    id: int | None = Field(default=None, primary_key=True)
    organization_id: int = Field(foreign_key="organizations.id", index=True)
    token_hash: str = Field(index=True, unique=True, max_length=64)
    created_by_user_id: int | None = Field(default=None, foreign_key="users.id")
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    revoked_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    accepted_count: int = Field(default=0)
    last_accepted_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )


# --- Benchmark models ---


class ChallengeBase(SQLModel):
    """Shared fields for benchmark challenges."""

    dataset: str = Field(description="'icar' or 'upload'")
    size: str = Field(default="full", description="'full' for ICAR, 'custom' for upload")
    period: str = Field(default="all", description="'all' for ICAR, 'custom' for upload")
    user_id: int | None = Field(
        default=None, description="Auth-ready; nullable until auth is added"
    )
    organization_id: int | None = Field(default=None, foreign_key="organizations.id")


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
    user_id: int | None = Field(default=None)
    organization_id: int | None = Field(default=None, foreign_key="organizations.id")
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


class UploadAudit(SQLModel, table=True):
    """Raw CSV upload audit record persisted in the File Share-backed database."""

    __tablename__: ClassVar[str] = "upload_audits"

    id: int | None = Field(default=None, primary_key=True)
    action_type: str = Field(index=True, max_length=80)
    status: str = Field(index=True, max_length=20)
    user_id: int | None = Field(default=None, foreign_key="users.id")
    user_email: str | None = Field(default=None, max_length=320)
    user_name: str | None = Field(default=None, max_length=200)
    organization_id: int | None = Field(default=None, foreign_key="organizations.id")
    organization_name: str | None = Field(default=None, max_length=200)
    original_filename: str = Field(max_length=500)
    content_type: str | None = Field(default=None, max_length=200)
    size_bytes: int
    sha256: str = Field(max_length=64)
    blob_container: str = Field(max_length=200)
    blob_path: str = Field(max_length=1000)
    challenge_id: int | None = Field(default=None, foreign_key="challenges.id")
    submission_id: int | None = Field(default=None, foreign_key="submissions.id")
    error_detail: str | None = Field(default=None)
    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=sa_func.now()),
    )
