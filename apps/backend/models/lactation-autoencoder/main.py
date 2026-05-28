"""Lactation autoencoder Azure Function App -- TF model predictions."""

from __future__ import annotations

import importlib
import logging
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

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
from bovi_core.ml import ModelRegistry, PredictorRegistry, create_model  # noqa: E402
from bovi_core.ml.dataloaders.sources import DictSource, TransformedSource  # noqa: E402
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry  # noqa: E402
from bovi_core.ml.dataloaders.transforms.timeseries import ImputationTransform  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from lactation_autoencoder.dataloaders import LactationDataset  # noqa: E402
from model_assets import ModelAssetError, ensure_model_assets  # noqa: E402
from schemas import (  # noqa: E402
    AutoencoderBatchRequest,
    AutoencoderBatchResponse,
    AutoencoderPredictRequest,
    AutoencoderPredictResponse,
)
from settings import Settings, get_settings  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

settings: Settings = get_settings()
logger = logging.getLogger("lactation_autoencoder")

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
# Prediction logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRuntime:
    """Loaded autoencoder runtime objects."""

    config: Config
    model: Any
    transforms: dict[str, object]


_model_runtime: ModelRuntime | None = None
_model_runtime_lock = threading.Lock()


def _ensure_autoencoder_registered() -> None:
    """Ensure autoencoder model and predictor decorators have populated registries."""
    model_module = importlib.import_module("lactation_autoencoder.models.lactation_model")
    predictor_module = importlib.import_module(
        "lactation_autoencoder.predictors.lactation_predictor"
    )

    if not ModelRegistry.is_registered("autoencoder"):
        importlib.reload(model_module)
    if not PredictorRegistry.is_registered("autoencoder"):
        importlib.reload(predictor_module)


@app.exception_handler(ModelAssetError)
async def model_asset_error_handler(request: Request, exc: ModelAssetError) -> JSONResponse:
    """Return a clear service-unavailable error when model assets are missing."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("Model asset error | request_id=%s | %s", request_id, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "request_id": request_id},
    )


def _get_model_runtime() -> ModelRuntime:
    """Load the model lazily so Azure Functions can index HTTP routes."""
    global _model_runtime
    if _model_runtime is None:
        with _model_runtime_lock:
            if _model_runtime is None:
                asset_paths = ensure_model_assets(settings)
                config = Config(
                    experiment_name="lactation_autoencoder",
                    config_file_path=str(asset_paths.config_path),
                    project_file_path=str(asset_paths.project_root / "pyproject.toml"),
                )
                _ensure_autoencoder_registered()
                model = create_model(config, "autoencoder")
                transforms = TransformRegistry.from_config(
                    config.experiment.dataloaders.inference.transforms
                )
                _model_runtime = ModelRuntime(config=config, model=model, transforms=transforms)
    return _model_runtime


def _build_transforms(
    imputation_method: str,
    transforms: dict[str, object],
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
    data: dict[str, object] = request.model_dump(exclude={"dim", "milkrecordings"})
    data["milk"] = request.model_milk_input()

    # Default events if not provided
    if data.get("events") is None:
        data["events"] = ["calving"] + ["pad"] * 303

    # Build transform list (swap imputation method if needed)
    runtime = _get_model_runtime()
    transform_list = _build_transforms(request.imputation_method, runtime.transforms)

    # Same pipeline as the notebook: DictSource → TransformedSource → LactationDataset
    dict_source = DictSource([data])
    transformed = TransformedSource(dict_source, transform_list)
    dataset = LactationDataset(source=transformed, config=runtime.config)
    features = dataset[0]["features"]

    result = runtime.model.predict(features, return_format="rich")

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


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for platform probes."""
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
