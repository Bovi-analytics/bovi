"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError

from bovi_api.database import dispose_engine
from bovi_api.routes import benchmark, datasets, health, herd_profiles, proxy, results
from bovi_api.settings import get_settings

_ALEMBIC_DIR = Path(__file__).parent / "alembic"
_MIGRATION_LOCK_RETRIES = 5
_MIGRATION_LOCK_RETRY_SECONDS = 2.0


def _is_sqlite_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _run_migrations() -> None:
    """Apply Alembic migrations up to head. Idempotent - safe to call on every startup.

    Skips when no database is configured (e.g. test collection in CI before
    fixtures replace the engine).
    """
    if not get_settings().database_url:
        return
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    for attempt in range(_MIGRATION_LOCK_RETRIES):
        try:
            command.upgrade(cfg, "head")
            return
        except OperationalError as exc:
            if not _is_sqlite_locked(exc) or attempt == _MIGRATION_LOCK_RETRIES - 1:
                raise
            sleep(_MIGRATION_LOCK_RETRY_SECONDS)


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
    app.include_router(datasets.router)
    app.include_router(benchmark.router)

    return app


app = create_app()
