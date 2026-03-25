"""Lactation autoencoder Azure Function App -- TF model predictions."""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from settings import Settings, get_settings
from starlette.middleware.base import BaseHTTPMiddleware

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
# Request / Response models
# ---------------------------------------------------------------------------


class AutoencoderPredictRequest(BaseModel):
    """Request body for autoencoder prediction.

    The input is a partial lactation curve (observed milk yields at
    specific DIM) and the model reconstructs/predicts the full curve.
    """

    dim: list[int] = Field(
        ...,
        min_length=1,
        description="Days in milk (DIM) for each observed recording.",
        examples=[[10, 30, 60, 90, 120, 150, 200, 250, 305]],
    )
    milkrecordings: list[float] = Field(
        ...,
        min_length=1,
        description="Observed milk yield (kg) at each DIM.",
        examples=[[15.0, 25.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0]],
    )
    parity: int = Field(
        default=1,
        ge=1,
        description="Lactation number (parity).",
    )
    breed: str = Field(
        default="H",
        description="Breed code, e.g. H = Holstein, J = Jersey.",
    )


class AutoencoderPredictResponse(BaseModel):
    """Response body for autoencoder prediction."""

    predictions: list[float] = Field(
        ...,
        description="Predicted daily milk yields for DIM 1-305.",
    )
    latent_vector: list[float] | None = Field(
        default=None,
        description="Latent-space representation (if available).",
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
    """Predict a full 305-day lactation curve from partial observations.

    Takes observed test-day data and uses the trained autoencoder to
    reconstruct/predict the complete lactation curve.

    Note: This is a placeholder skeleton. The actual model loading and
    inference logic will be integrated in a future step when the
    lactation_autoencoder package predictor is wired in.
    """
    # TODO: Load the trained autoencoder model and run inference.
    #   from lactation_autoencoder.predictor import AutoencoderPredictor
    #   predictor = AutoencoderPredictor.load(model_path)
    #   result = predictor.predict(request.dim, request.milkrecordings, ...)

    # Placeholder: return zeros for 305 days to indicate skeleton status.
    placeholder_predictions = [0.0] * 305

    return AutoencoderPredictResponse(
        predictions=placeholder_predictions,
        latent_vector=None,
    )
