"""Lactation curves Azure Function App — classical fitting and milkbot endpoints."""

import logging
import time
import uuid
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lactationcurve.characteristics import calculate_characteristic, test_interval_method
from lactationcurve.fitting import fit_lactation_curve, milkbot_model
from lactationcurve.preprocessing.validate_and_standardize import MilkBotPriors
from pydantic import BaseModel, Field, ValidationError, model_validator
from settings import Settings, get_settings
from starlette.middleware.base import BaseHTTPMiddleware

settings: Settings = get_settings()
logger = logging.getLogger("lactation_curves")

app = FastAPI(
    title="Lactation Curves",
    description="Classical lactation curve fitting and MilkBot prediction",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing and a unique request ID."""

    async def dispatch(self, request: Request, call_next):
        """Dispatch the request and log the result."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Unhandled error | %s %s | request_id=%s | %.0fms",
                request.method,
                request.url.path,
                request_id,
                duration_ms,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id

        if response.status_code >= 500:
            logger.error(
                "%s %s -> %d | request_id=%s | %.0fms",
                request.method,
                request.url.path,
                response.status_code,
                request_id,
                duration_ms,
            )
        else:
            logger.info(
                "%s %s -> %d | request_id=%s | %.0fms",
                request.method,
                request.url.path,
                response.status_code,
                request_id,
                duration_ms,
            )

        return response


app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _log_and_return_422(request: Request, errors: list | Sequence) -> JSONResponse:
    """Return a 422 response with structured validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "Validation error | %s %s | request_id=%s | errors=%s",
        request.method,
        request.url.path,
        request_id,
        errors,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(errors), "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle FastAPI request validation errors."""
    return _log_and_return_422(request, exc.errors())


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    return _log_and_return_422(request, exc.errors())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled exception | %s %s | request_id=%s | %s: %s",
        request.method,
        request.url.path,
        request_id,
        type(exc).__name__,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Field descriptions
# ---------------------------------------------------------------------------

DIM_DESC = (
    "Days in milk (DIM) for each test-day recording. Must have the same length as milkrecordings."
)
MILK_DESC = "Milk yield (kg) for each test-day recording. Must have the same length as dim."
MODEL_DESC = (
    "Lactation curve model to fit."
    " Wood (3-param), Wilmink (4-param), Ali-Schaeffer (5-param),"
    " Fischer (3-param), or MilkBot (4-param)."
)
FITTING_DESC = (
    "Fitting method. Currently only frequentist"
    " (scipy optimization) is supported via this endpoint."
)
CHARACTERISTIC_DESC = (
    "Which lactation characteristic to compute."
    " time_to_peak: DIM at peak yield."
    " peak_yield: maximum daily yield (kg)."
    " cumulative_milk_yield: total kg over the lactation."
    " persistency: rate of decline after peak."
)
PERSISTENCY_DESC = (
    "How to calculate persistency."
    " 'derived': average slope after peak (default)."
    " 'literature': analytical formula (Wood/MilkBot only)."
)
LACTATION_LENGTH_DESC = (
    "Lactation length in days for the calculation. 305 (default), or a custom integer."
)
TEST_IDS_DESC = (
    "Optional lactation/animal identifier per record."
    " When omitted, all records are treated as one lactation."
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class MilkBotPredictRequest(BaseModel):
    """Request body for direct MilkBot model prediction with known parameters."""

    t: list[int] = Field(
        ...,
        description="Days in milk (DIM) at which to evaluate the MilkBot model.",
        examples=[[1, 30, 60, 90, 120, 150, 200, 250, 305]],
    )
    a: float = Field(
        ...,
        description="Scale -- overall milk production level (kg).",
        examples=[40.0],
    )
    b: float = Field(
        ...,
        description="Ramp -- rate of rise in early lactation.",
        examples=[20.0],
    )
    c: float = Field(
        ...,
        description="Offset -- time correction for calving.",
        examples=[0.5],
    )
    d: float = Field(
        ...,
        description="Decay -- rate of exponential decline.",
        examples=[0.003],
    )


class FitRequest(BaseModel):
    """Request body for fitting a lactation curve model."""

    dim: list[int] = Field(
        ...,
        min_length=2,
        description=DIM_DESC,
        examples=[[10, 30, 60, 90, 120, 150, 200, 250, 305]],
    )
    milkrecordings: list[float] = Field(
        ...,
        min_length=2,
        description=MILK_DESC,
        examples=[[15.0, 25.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0]],
    )
    model: Literal["wood", "wilmink", "ali_schaeffer", "fischer", "milkbot"] = Field(
        default="wood",
        description=MODEL_DESC,
    )
    fitting: Literal["frequentist"] = Field(
        default="frequentist",
        description=FITTING_DESC,
    )
    breed: Literal["H", "J"] = Field(
        default="H",
        description="Breed: H = Holstein, J = Jersey.",
    )
    parity: int = Field(
        default=3,
        ge=1,
        description="Lactation number. Parities >= 3 are one group.",
    )
    continent: Literal["USA", "EU"] = Field(
        default="USA",
        description="Continent for priors: USA or EU",
    )
    custom_priors: MilkBotPriors | Literal["CHEN"] | None = Field(
        default=None,
        description=(
            "Custom prior distributions for Bayesian"
            " fitting. If a dict is provided, it must"
            " be a dictionary of prior distributions"
            " for each parameter in the model including"
            " mean and std values. If the string 'CHEN'"
            " is provided, the default Chen et al."
            " priors are used."
        ),
    )
    milk_unit: Literal["kg", "lbs"] = Field(
        default="kg",
        description="Unit of milk yield",
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Ensure dim and milkrecordings have the same length."""
        if len(self.dim) != len(self.milkrecordings):
            msg = (
                f"dim and milkrecordings must have the same length, "
                f"got {len(self.dim)} and {len(self.milkrecordings)}"
            )
            raise ValueError(msg)
        return self


class CharacteristicRequest(BaseModel):
    """Request body for computing a lactation characteristic."""

    dim: list[int] = Field(
        ...,
        min_length=2,
        description=DIM_DESC,
        examples=[[10, 30, 60, 90, 120, 150, 200, 250, 305]],
    )
    milkrecordings: list[float] = Field(
        ...,
        min_length=2,
        description=MILK_DESC,
        examples=[[15.0, 25.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0]],
    )
    model: Literal["wood", "wilmink", "ali_schaeffer", "fischer", "milkbot"] = Field(
        default="wood",
        description=MODEL_DESC,
    )
    characteristic: Literal[
        "time_to_peak",
        "peak_yield",
        "cumulative_milk_yield",
        "persistency",
    ] = Field(
        default="cumulative_milk_yield",
        description=CHARACTERISTIC_DESC,
    )
    fitting: Literal["frequentist"] = Field(
        default="frequentist",
        description=FITTING_DESC,
    )
    breed: Literal["H", "J"] = Field(
        default="H",
        description="Breed: H = Holstein, J = Jersey.",
    )
    parity: int = Field(
        default=3,
        ge=1,
        description="Lactation number. Parities >= 3 are one group.",
    )
    continent: Literal["USA", "EU"] = Field(
        default="USA",
        description="Continent for priors: USA or EU",
    )
    custom_priors: MilkBotPriors | Literal["CHEN"] | None = Field(
        default=None,
        description=(
            "Custom prior distributions for Bayesian"
            " fitting. If a dict is provided, it must"
            " be a dictionary of prior distributions"
            " for each parameter in the model including"
            " mean and std values. If the string 'CHEN'"
            " is provided, the default Chen et al."
            " priors are used."
        ),
    )
    milk_unit: Literal["kg", "lbs"] = Field(
        default="kg",
        description="Unit of milk yield",
    )
    persistency_method: Literal["derived", "literature"] = Field(
        default="derived",
        description=PERSISTENCY_DESC,
    )
    lactation_length: int = Field(
        default=305,
        ge=1,
        description=LACTATION_LENGTH_DESC,
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Ensure dim and milkrecordings have the same length."""
        if len(self.dim) != len(self.milkrecordings):
            msg = (
                f"dim and milkrecordings must have the same length, "
                f"got {len(self.dim)} and {len(self.milkrecordings)}"
            )
            raise ValueError(msg)
        return self


class TestIntervalRequest(BaseModel):
    """Request body for the ICAR Test Interval Method."""

    dim: list[int] = Field(
        ...,
        min_length=2,
        description=DIM_DESC,
        examples=[[10, 30, 60, 90, 120, 150, 200, 250, 305]],
    )
    milkrecordings: list[float] = Field(
        ...,
        min_length=2,
        description=MILK_DESC,
        examples=[[15.0, 25.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0]],
    )
    test_ids: list[int | str] | None = Field(
        default=None,
        description=TEST_IDS_DESC,
        examples=[[1, 1, 1, 1, 1, 1, 1, 1, 1]],
    )

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        """Ensure dim and milkrecordings have the same length."""
        if len(self.dim) != len(self.milkrecordings):
            msg = (
                f"dim and milkrecordings must have the same length, "
                f"got {len(self.dim)} and {len(self.milkrecordings)}"
            )
            raise ValueError(msg)
        if self.test_ids is not None and len(self.test_ids) != len(self.dim):
            msg = (
                f"test_ids must have the same length as dim, "
                f"got {len(self.test_ids)} and {len(self.dim)}"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: MilkBotPredictRequest) -> dict[str, list[float]]:
    """Evaluate the MilkBot model with known parameters.

    Use this when you already have the four MilkBot parameters
    (a, b, c, d) and want predicted milk yields at specific DIM.
    """
    t = np.array(request.t)
    predictions = milkbot_model(
        t,
        request.a,
        request.b,
        request.c,
        request.d,
    )
    return {"predictions": np.asarray(predictions).tolist()}


@app.post("/fit")
def fit(request: FitRequest) -> dict[str, list[float]]:
    """Fit a lactation curve model to test-day milk recordings.

    Takes observed test-day data (DIM + milk yields) and fits the
    specified model using scipy optimization. Returns predicted
    daily milk yields for DIM 1-305 (or up to max(dim) if > 305).

    The response contains 305+ predicted values, one per day.
    """
    predictions = fit_lactation_curve(
        dim=request.dim,
        milkrecordings=request.milkrecordings,
        model=request.model,
        fitting=request.fitting,
        breed=request.breed,
        parity=request.parity,
        continent=request.continent,
        custom_priors=request.custom_priors,
        milk_unit=request.milk_unit,
    )
    return {"predictions": predictions.tolist()}


@app.post("/characteristic")
def characteristic(
    request: CharacteristicRequest,
) -> dict[str, float]:
    """Compute a single lactation characteristic from milk recordings.

    Fits a lactation curve model to the data, then derives one of:
    - **time_to_peak**: DIM at which peak yield occurs.
    - **peak_yield**: maximum daily milk yield (kg).
    - **cumulative_milk_yield**: total kg over the lactation.
    - **persistency**: rate of decline after peak.

    Returns a single numeric value.
    """
    value = calculate_characteristic(
        dim=request.dim,
        milkrecordings=request.milkrecordings,
        model=request.model,
        characteristic=request.characteristic,
        fitting=request.fitting,
        parity=request.parity,
        breed=request.breed,
        continent=request.continent,
        custom_priors=request.custom_priors,
        milk_unit=request.milk_unit,
        persistency_method=request.persistency_method,
        lactation_length=request.lactation_length,
    )
    return {"value": float(value)}


@app.post("/test-interval")
def test_interval(
    request: TestIntervalRequest,
) -> dict[str, list[dict]]:
    """Calculate 305-day milk yield using the ICAR Test Interval Method.

    Uses the trapezoidal rule for interim test days and linear
    projection for the start/end of lactation. Records with
    DIM > 305 are excluded.

    Returns one result per unique test_id (or one result when
    test_ids is omitted).
    """
    data: dict[str, list] = {
        "DaysInMilk": request.dim,
        "MilkingYield": request.milkrecordings,
    }
    if request.test_ids is not None:
        data["TestId"] = request.test_ids
    df = pd.DataFrame(data)
    result_df = test_interval_method(df)
    return {
        "results": [
            {
                "test_id": row["TestId"],
                "total_305_yield": float(str(row["Total305Yield"])),
            }
            for _, row in result_df.iterrows()
        ],
    }
