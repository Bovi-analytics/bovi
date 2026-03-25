"""PostgreSQL database connection and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from bovi_api.settings import get_settings

_engine = None
_session_factory = None


def _get_engine():
    """Return a shared async engine, creating it lazily."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            msg = "DATABASE_URL is not configured"
            raise RuntimeError(msg)
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a shared session factory."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (use as FastAPI dependency)."""
    factory = _get_session_factory()
    async with factory() as session:
        yield session


async def create_tables() -> None:
    """Create all tables (for development; use Alembic in production)."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose the engine and close all connections."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
