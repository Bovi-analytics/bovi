"""Proxy endpoints that forward requests to model Function Apps."""

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from bovi_api.settings import get_settings

logger = logging.getLogger("bovi_api.proxy")

router = APIRouter()

_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client, creating it lazily."""
    global _http_client  # noqa: PLW0603
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


async def close_client() -> None:
    """Close the shared HTTP client."""
    global _http_client  # noqa: PLW0603
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def _proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
    """Forward a POST request to a model Function App and return its response."""
    client = _get_client()
    body = await request.body()
    try:
        response = await client.post(
            f"{base_url}{path}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        return JSONResponse(status_code=response.status_code, content=response.json())
    except httpx.RequestError as exc:
        logger.exception("Proxy error forwarding to %s%s: %s", base_url, path, exc)
        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream service unavailable: {base_url}{path}"},
        )


# ---------------------------------------------------------------------------
# Lactation Curves
# ---------------------------------------------------------------------------

settings = get_settings()


@router.post("/curves/fit")
async def proxy_curves_fit(request: Request) -> JSONResponse:
    """Proxy: fit a lactation curve model."""
    return await _proxy_post(settings.lactation_curves_url, "/fit", request)


@router.post("/curves/predict")
async def proxy_curves_predict(request: Request) -> JSONResponse:
    """Proxy: evaluate the MilkBot model with known parameters."""
    return await _proxy_post(settings.lactation_curves_url, "/predict", request)


@router.post("/curves/characteristic")
async def proxy_curves_characteristic(request: Request) -> JSONResponse:
    """Proxy: compute a lactation characteristic."""
    return await _proxy_post(settings.lactation_curves_url, "/characteristic", request)


@router.post("/curves/test-interval")
async def proxy_curves_test_interval(request: Request) -> JSONResponse:
    """Proxy: calculate 305-day yield via ICAR Test Interval Method."""
    return await _proxy_post(settings.lactation_curves_url, "/test-interval", request)


# ---------------------------------------------------------------------------
# Lactation Autoencoder
# ---------------------------------------------------------------------------


@router.post("/autoencoder/predict")
async def proxy_autoencoder_predict(request: Request) -> JSONResponse:
    """Proxy: predict full lactation curve via autoencoder."""
    return await _proxy_post(settings.lactation_autoencoder_url, "/predict", request)
