"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bovi_api.database import dispose_engine
from bovi_api.routes import (
    admin,
    auth,
    benchmark,
    datasets,
    health,
    herd_profiles,
    organizations,
    proxy,
    results,
    uploaded_datasets,
)
from bovi_api.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    yield
    await proxy.close_client()
    await dispose_engine()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

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
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(organizations.router)
    app.include_router(proxy.router)
    app.include_router(results.router)
    app.include_router(herd_profiles.router, prefix="/herd-profiles")
    app.include_router(datasets.router)
    app.include_router(uploaded_datasets.router)
    app.include_router(benchmark.router)

    return app


app = create_app()
