"""Request and response schemas for the lactation autoencoder API."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

VALID_IMPUTATION_METHODS = Literal["forward_fill", "backward_fill", "linear"]
AUTOENCODER_INPUT_DAYS = 304


def project_periodic_records_to_daily(
    dim: list[int],
    milkrecordings: list[float],
    max_days: int = AUTOENCODER_INPUT_DAYS,
) -> list[float | None]:
    """Project sparse DIM observations into a fixed daily sequence.

    Observed values are placed at ``DIM - 1``. All unobserved days remain
    ``0.0`` so periodic records can be sent through the existing autoencoder
    pipeline without inventing interpolated yields.
    """
    projected: list[float | None] = [0.0] * max_days
    for day, milk in zip(dim, milkrecordings, strict=True):
        projected[day - 1] = milk
    return projected


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

        if self.dim is not None and self.milkrecordings is not None:
            if len(self.dim) != len(self.milkrecordings):
                raise ValueError(
                    "dim and milkrecordings must have the same length, "
                    f"got {len(self.dim)} and {len(self.milkrecordings)}"
                )
            invalid_dims = [day for day in self.dim if day < 1 or day > AUTOENCODER_INPUT_DAYS]
            if invalid_dims:
                raise ValueError(
                    f"dim values must be between 1 and {AUTOENCODER_INPUT_DAYS} for "
                    "autoencoder projection."
                )
            if len(set(self.dim)) != len(self.dim):
                raise ValueError("dim values must be unique for autoencoder projection.")

        return self

    def model_milk_input(self) -> list[float | None]:
        """Return the daily milk sequence consumed by the runtime pipeline."""
        if self.milk is not None:
            return self.milk
        if self.dim is None or self.milkrecordings is None:
            raise ValueError("Validated request has no usable milk input.")
        return project_periodic_records_to_daily(self.dim, self.milkrecordings)


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
