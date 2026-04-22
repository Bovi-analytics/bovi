"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bovi_api.database import dispose_engine
from bovi_api.routes import health, herd_profiles, proxy, results
from bovi_api.settings import get_settings


def _run_migrations() -> None:
    """Apply Alembic migrations up to head. Idempotent — safe to call on every startup."""
    api_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(api_root / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    yield
    await proxy.close_client()
    await dispose_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Run migrations synchronously at app-construction time, before uvicorn's
    # event loop starts. Alembic's env.py uses asyncio.run() internally, which
    # can't be called from inside an already-running loop.
    _run_migrations()

    app = FastAPI(
        title="Bovi API",
        description="Central gateway for the Bovi dairy analytics platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(proxy.router)
    app.include_router(results.router)
    app.include_router(herd_profiles.router, prefix="/herd-profiles")

    return app


app = create_app()
