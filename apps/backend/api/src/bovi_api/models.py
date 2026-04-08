"""SQLModel models — single source of truth for DB tables AND Pydantic schemas."""

from datetime import datetime

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
