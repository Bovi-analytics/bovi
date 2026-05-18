"""Health check endpoint."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from bovi_api.database import check_database_connection, get_session

router = APIRouter()


@router.get("/")
async def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}


@router.get("/health")
async def health_alias() -> dict[str, str]:
    """Liveness check endpoint with an explicit path."""
    return {"status": "ok"}


@router.get("/health/db")
async def database_health(session: AsyncSession = Depends(get_session)) -> JSONResponse:
    """Readiness check for the configured database connection."""
    result = await check_database_connection(session)
    http_status = (
        status.HTTP_200_OK if result["status"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=http_status, content=result)
