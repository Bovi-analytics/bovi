"""Authentication endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from bovi_api.auth import CurrentUser, current_user_payload, require_auth
from bovi_api.settings import Settings, get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def auth_status(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Return auth configuration status for the dashboard."""
    configured = bool(settings.azure_ad_client_id)
    return {
        "auth_configured": configured,
        "require_auth": not settings.auth_disabled,
        "auth_mode": "dev" if settings.dev_mode or settings.auth_disabled else "azure_ad",
    }


@router.get("/me")
async def me(current_user: Annotated[CurrentUser, Depends(require_auth)]) -> dict[str, Any]:
    """Return the current local Bovi user and organizations."""
    return await current_user_payload(current_user)


@router.get("/verify")
async def verify(current_user: Annotated[CurrentUser, Depends(require_auth)]) -> dict[str, Any]:
    """Verify the current token."""
    return {"valid": True, "user": await current_user_payload(current_user)}
