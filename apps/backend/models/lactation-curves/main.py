"""Lactation curves Azure Function App - classical fitting and milkbot endpoints."""

import logging
import time
import uuid
from collections.abc import Sequence
from typing import Literal, Self, TypedDict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from lactationcurve.characteristics import (
    ISLC,
    best_predict_method,
    calculate_characteristic,
    test_interval_method,
)
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
    "Fitting method. Frequentist uses scipy optimization."
    " Bayesian is supported for MilkBot via the external MilkBot API."
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
    fitting: Literal["frequentist", "bayesian"] = Field(
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
    continent: Literal["USA", "EU", "CHEN"] = Field(
        default="USA",
        description="Prior source: USA, EU, or CHEN literature priors.",
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
    milk_unit: Literal["kg", "lb"] = Field(
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
        if self.fitting == "bayesian" and self.model != "milkbot":
            raise ValueError("Bayesian fitting is currently only implemented for MilkBot")
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
    fitting: Literal["frequentist", "bayesian"] = Field(
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
    continent: Literal["USA", "EU", "CHEN"] = Field(
        default="USA",
        description="Prior source: USA, EU, or CHEN literature priors.",
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
    milk_unit: Literal["kg", "lb"] = Field(
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
        if self.fitting == "bayesian" and self.model != "milkbot":
            raise ValueError("Bayesian fitting is currently only implemented for MilkBot")
        return self


class CharacteristicBatchItem(CharacteristicRequest):
    """One item in a batch characteristic request."""

    id: str | int | None = Field(
        default=None,
        description="Optional caller-provided identifier echoed in the batch response.",
    )


class CharacteristicBatchRequest(BaseModel):
    """Request body for computing lactation characteristics in batch."""

    items: list[CharacteristicBatchItem] = Field(
        ...,
        min_length=1,
        description="Characteristic requests to evaluate.",
    )


class CharacteristicBatchResult(BaseModel):
    """One result in a batch characteristic response."""

    id: str | int | None = None
    value: float | None = None


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


class YieldEstimateRequest(TestIntervalRequest):
    """Request body for 305-day yield estimators using test-day records."""


class _BayesianMilkBotKwargs(TypedDict):
    continent: str
    custom_priors: MilkBotPriors | Literal["CHEN"] | None
    key: str | None


def _bayesian_milkbot_kwargs(
    request: FitRequest | CharacteristicRequest,
) -> _BayesianMilkBotKwargs:
    """Return normalized fitting kwargs for package calls."""
    custom_priors: MilkBotPriors | Literal["CHEN"] | None = request.custom_priors
    continent = request.continent
    key: str | None = None

    if request.fitting == "bayesian":
        milkbot_key = getattr(settings, "milkbot_key", "")
        if not milkbot_key:
            raise HTTPException(
                status_code=503,
                detail="MILKBOT_KEY is required for Bayesian MilkBot fitting.",
            )
        key = milkbot_key
        if request.continent == "CHEN":
            continent = "USA"
            custom_priors = "CHEN"

    return {
        "continent": continent,
        "custom_priors": custom_priors,
        "key": key,
    }


def _calculate_characteristic_value(request: CharacteristicRequest) -> float | None:
    """Calculate a characteristic value for a validated request model."""
    bayesian_kwargs = _bayesian_milkbot_kwargs(request)
    value = calculate_characteristic(
        dim=request.dim,
        milkrecordings=request.milkrecordings,
        model=request.model,
        characteristic=request.characteristic,
        fitting=request.fitting,
        parity=request.parity,
        breed=request.breed,
        continent=bayesian_kwargs["continent"],
        custom_priors=bayesian_kwargs["custom_priors"],
        key=bayesian_kwargs["key"],
        milk_unit=request.milk_unit,
        persistency_method=request.persistency_method,
        lactation_length=request.lactation_length,
    )
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for platform probes."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: MilkBotPredictRequest) -> dict[str, list[float]]:
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
async def fit(request: FitRequest) -> dict[str, list[float]]:
    """Fit a lactation curve model to test-day milk recordings.

    Takes observed test-day data (DIM + milk yields) and fits the
    specified model using scipy optimization. Returns predicted
    daily milk yields for DIM 1-305 (or up to max(dim) if > 305).

    The response contains 305+ predicted values, one per day.
    """
    bayesian_kwargs = _bayesian_milkbot_kwargs(request)
    predictions = fit_lactation_curve(
        dim=request.dim,
        milkrecordings=request.milkrecordings,
        model=request.model,
        fitting=request.fitting,
        breed=request.breed,
        parity=request.parity,
        continent=bayesian_kwargs["continent"],
        custom_priors=bayesian_kwargs["custom_priors"],
        key=bayesian_kwargs["key"],
        milk_unit=request.milk_unit,
    )
    return {"predictions": predictions.tolist()}


@app.post("/characteristic")
async def characteristic(
    request: CharacteristicRequest,
) -> dict[str, float | None]:
    """Compute a single lactation characteristic from milk recordings.

    Fits a lactation curve model to the data, then derives one of:
    - **time_to_peak**: DIM at which peak yield occurs.
    - **peak_yield**: maximum daily milk yield (kg).
    - **cumulative_milk_yield**: total kg over the lactation.
    - **persistency**: rate of decline after peak.

    Returns a single numeric value, or null if no sensible value exists.
    """
    return {"value": _calculate_characteristic_value(request)}


@app.post("/characteristic/batch")
async def characteristic_batch(
    request: CharacteristicBatchRequest,
) -> dict[str, list[CharacteristicBatchResult]]:
    """Compute multiple lactation characteristics in a single request."""
    return {
        "results": [
            CharacteristicBatchResult(
                id=item.id,
                value=_calculate_characteristic_value(item),
            )
            for item in request.items
        ]
    }


@app.post("/test-interval")
async def test_interval(
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
    result_df = test_interval_method(df, default_test_id=1)
    return {
        "results": [
            {
                "test_id": row["TestId"],
                "total_305_yield": float(str(row["LactationMilkYield"])),
            }
            for _, row in result_df.iterrows()
        ],
    }


@app.post("/islc")
async def islc(
    request: YieldEstimateRequest,
) -> dict[str, list[dict]]:
    """Calculate lactation milk yield using the ISLC ICAR-style method."""
    data: dict[str, list] = {
        "DaysInMilk": request.dim,
        "MilkingYield": request.milkrecordings,
    }
    if request.test_ids is not None:
        data["TestId"] = request.test_ids
    result_df = ISLC(pd.DataFrame(data))
    return {
        "results": [
            {
                "test_id": row["TestId"],
                "total_305_yield": float(str(row["LactationMilkYield"])),
            }
            for _, row in result_df.iterrows()
        ],
    }


@app.post("/best-predict")
async def best_predict(
    request: YieldEstimateRequest,
) -> dict[str, list[dict]]:
    """Calculate 305-day milk yield using best prediction."""
    data: dict[str, list] = {
        "DaysInMilk": request.dim,
        "MilkingYield": request.milkrecordings,
    }
    if request.test_ids is not None:
        data["TestId"] = request.test_ids
    result_df = best_predict_method(pd.DataFrame(data))
    return {
        "results": [
            {
                "test_id": row["TestId"],
                "total_305_yield": float(str(row["LactationMilkYield"])),
            }
            for _, row in result_df.iterrows()
        ],
    }
