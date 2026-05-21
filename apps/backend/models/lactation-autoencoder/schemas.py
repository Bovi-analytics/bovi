"""Request and response schemas for the lactation autoencoder API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VALID_IMPUTATION_METHODS = Literal["forward_fill", "backward_fill", "linear"]


class AutoencoderPredictRequest(BaseModel):
    """Request body for a single autoencoder prediction."""

    milk: list[float | None] = Field(
        ...,
        min_length=1,
        description="Daily milk yield (kg). Padded/truncated to 304.",
    )
    events: list[str] | None = Field(
        default=None,
        description="Daily events. Case-insensitive.",
    )
    parity: int = Field(default=1, ge=1, le=12)
    herd_id: int | None = Field(default=None)
    herd_stats: list[float] | None = Field(default=None, min_length=10, max_length=10)
    imputation_method: VALID_IMPUTATION_METHODS = Field(default="forward_fill")


class AutoencoderPredictResponse(BaseModel):
    """Response body for autoencoder prediction."""

    predictions: list[float] = Field(..., description="Predicted milk yields (304 days).")
    latent_vector: list[float] | None = Field(default=None)


class AutoencoderBatchRequest(BaseModel):
    """Request body for batch autoencoder prediction."""

    items: list[AutoencoderPredictRequest] = Field(..., min_length=1)
    imputation_method: VALID_IMPUTATION_METHODS = Field(default="forward_fill")


class AutoencoderBatchResponse(BaseModel):
    """Response body for batch autoencoder prediction."""

    results: list[AutoencoderPredictResponse]
