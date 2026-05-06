"""Lactation autoencoder Azure Function App -- TF model predictions."""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from typing import Literal

# MLflow emits a UserWarning about `Any` type hints from its OWN internal
# response schemas during import. There's no way for us to make those hints
# more specific - they're library-internal. A normal `filterwarnings` call
# won't survive: langchain_core (pulled in transitively) calls
# `warnings.resetwarnings()` during mlflow's import and wipes user filters.
# Monkey-patching `showwarning` is the only reliable way to silence it,
# since `showwarning` runs at display time after filter matching.
_orig_showwarning = warnings.showwarning


def _showwarning(message, category, filename, lineno, file=None, line=None):
    if "Any type hint is inferred as AnyType" in str(message):
        return
    _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _showwarning

from bovi_core.config import Config  # noqa: E402
from bovi_core.ml import create_model  # noqa: E402
from bovi_core.ml.dataloaders.sources import DictSource, TransformedSource  # noqa: E402
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry  # noqa: E402
from bovi_core.ml.dataloaders.transforms.timeseries import ImputationTransform  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from lactation_autoencoder.dataloaders import LactationDataset  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from settings import Settings, get_settings  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

settings: Settings = get_settings()
logger = logging.getLogger("lactation_autoencoder")

# ---------------------------------------------------------------------------
# Model startup (loaded once at module level)
# ---------------------------------------------------------------------------

config = Config(experiment_name="lactation_autoencoder", project_name="bovi")
model = create_model(config, "autoencoder")
transforms = TransformRegistry.from_config(config.experiment.dataloaders.inference.transforms)

app = FastAPI(
    title="Lactation Autoencoder",
    description="TensorFlow autoencoder for milk production prediction",
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
# Request / Response models
# ---------------------------------------------------------------------------

VALID_IMPUTATION_METHODS = Literal["forward_fill", "backward_fill", "linear", "zero", "mean"]


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


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def _build_transforms(
    imputation_method: str,
) -> list[object]:
    """Build transform list, swapping imputation method if needed.

    Args:
        imputation_method: Imputation strategy to use.

    Returns:
        List of transform callables.

    """
    default_method = "forward_fill"
    if imputation_method == default_method:
        return list(transforms.values())

    # Swap the imputation transform with one using the requested method
    swapped: list[object] = []
    for transform in transforms.values():
        if isinstance(transform, ImputationTransform):
            swapped.append(
                ImputationTransform(
                    method=imputation_method,
                    fields=transform.fields,
                )
            )
        else:
            swapped.append(transform)
    return swapped


def _predict_single(
    request: AutoencoderPredictRequest,
) -> AutoencoderPredictResponse:
    """Run prediction for a single request through the full pipeline.

    Args:
        request: Validated prediction request.

    Returns:
        Prediction response with 304-day milk yields.

    """
    data: dict[str, object] = request.model_dump()

    # Default events if not provided
    if data.get("events") is None:
        data["events"] = ["calving"] + ["pad"] * 303

    # Build transform list (swap imputation method if needed)
    transform_list = _build_transforms(request.imputation_method)

    # Same pipeline as the notebook: DictSource → TransformedSource → LactationDataset
    dict_source = DictSource([data])
    transformed = TransformedSource(dict_source, transform_list)
    dataset = LactationDataset(source=transformed, config=config)
    features = dataset[0]["features"]

    result = model.predict(features, return_format="rich")

    return AutoencoderPredictResponse(
        predictions=result.predictions.tolist(),
        latent_vector=None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=AutoencoderPredictResponse)
def predict(request: AutoencoderPredictRequest) -> AutoencoderPredictResponse:
    """Predict a full 304-day lactation curve from partial observations.

    Takes observed milk recordings and uses the trained autoencoder to
    reconstruct/predict the complete lactation curve.
    """
    return _predict_single(request)


@app.post("/predict/batch", response_model=AutoencoderBatchResponse)
def predict_batch(request: AutoencoderBatchRequest) -> AutoencoderBatchResponse:
    """Predict lactation curves for a batch of animals.

    Each item uses its own imputation_method if set, otherwise falls back
    to the batch-level imputation_method.
    """
    results: list[AutoencoderPredictResponse] = []
    for item in request.items:
        # Use batch-level imputation_method as fallback when item uses default
        if item.imputation_method == "forward_fill" and request.imputation_method != "forward_fill":
            item = item.model_copy(update={"imputation_method": request.imputation_method})
        results.append(_predict_single(item))

    return AutoencoderBatchResponse(results=results)
