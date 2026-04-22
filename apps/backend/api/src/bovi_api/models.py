"""SQLModel models — single source of truth for DB tables AND Pydantic schemas."""

from datetime import datetime

from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlalchemy import func as sa_func
from sqlmodel import Field, SQLModel


class FittingResultBase(SQLModel):
    """Shared fields for fitting results (used in create requests and responses)."""

    model_type: str = Field(
        index=True, description="Model used (e.g. 'wood', 'milkbot', 'autoencoder')"
    )
    source_app: str = Field(index=True, description="Which backend app produced this result")
    input_data: dict = Field(
        default_factory=dict, sa_type=None, description="The original request payload"
    )
    output_data: dict = Field(
        default_factory=dict, sa_type=None, description="The prediction/fitting response"
    )
    metadata_extra: dict = Field(
        default_factory=dict, sa_type=None, description="Additional metadata"
    )


class FittingResult(FittingResultBase, table=True):
    """Database table for fitting results."""

    __tablename__ = "fitting_results"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime | None = Field(default=None, sa_column_kwargs={"server_default": "now()"})


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

    __tablename__ = "herd_profiles"
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
