"""Request and response schemas for the lactation autoencoder API."""

from __future__ import annotations

import math
from typing import Literal, Self, cast

from lactation_autoencoder.dataloaders.datasets.lactation_dataset import (
    AUTOENCODER_INPUT_DAYS,
    periodic_records_in_horizon,
    project_periodic_records_to_daily,
)
from pydantic import BaseModel, Field, model_validator

VALID_IMPUTATION_METHODS = Literal["forward_fill", "backward_fill", "linear"]


class AutoencoderPredictRequest(BaseModel):
    """Request body for a single autoencoder prediction."""

    milk: list[float | None] | None = Field(
        default=None,
        min_length=1,
        description="Daily milk yield (kg). Padded/truncated to 304.",
    )
    dim: list[int] | None = Field(
        default=None,
        min_length=1,
        description="Days in milk for periodic observations. Alternative to milk.",
    )
    milkrecordings: list[float] | None = Field(
        default=None,
        min_length=1,
        description="Milk yield (kg) at each DIM. Alternative to milk.",
    )
    events: list[str] | None = Field(
        default=None,
        description="Daily events. Case-insensitive.",
    )
    parity: int = Field(default=1, ge=1, le=12)
    herd_id: int | None = Field(default=None)
    herd_stats: list[float] | None = Field(default=None, min_length=10, max_length=10)
    imputation_method: VALID_IMPUTATION_METHODS = Field(default="forward_fill")

    @model_validator(mode="after")
    def check_input_shape(self) -> Self:
        """Ensure callers provide exactly one supported observation shape."""
        has_daily = self.milk is not None
        has_dim = self.dim is not None
        has_periodic_milk = self.milkrecordings is not None
        has_periodic = has_dim or has_periodic_milk

        if has_daily and has_periodic:
            raise ValueError("Provide either milk or dim + milkrecordings, not both.")
        if not has_daily and not has_periodic:
            raise ValueError("Provide either milk or dim + milkrecordings.")
        if has_periodic and not (has_dim and has_periodic_milk):
            raise ValueError("Periodic input requires both dim and milkrecordings.")

        if self.milk is not None:
            invalid_milk = [
                value
                for value in self.milk
                if value is not None and (not math.isfinite(value) or value < 0)
            ]
            if invalid_milk:
                raise ValueError("milk values must be finite, non-negative numbers or null.")

        if self.dim is not None and self.milkrecordings is not None:
            if len(self.dim) != len(self.milkrecordings):
                raise ValueError(
                    "dim and milkrecordings must have the same length, "
                    f"got {len(self.dim)} and {len(self.milkrecordings)}"
                )
            self.dim, self.milkrecordings = periodic_records_in_horizon(
                self.dim,
                self.milkrecordings,
                AUTOENCODER_INPUT_DAYS,
            )

        return self

    def model_milk_input(self) -> list[float | None]:
        """Return the daily milk sequence consumed by the runtime pipeline."""
        if self.milk is not None:
            return self.milk
        if self.dim is None or self.milkrecordings is None:
            raise ValueError("Validated request has no usable milk input.")
        return cast(
            list[float | None],
            project_periodic_records_to_daily(self.dim, self.milkrecordings),
        )


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
