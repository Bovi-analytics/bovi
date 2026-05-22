"""Database connection and session management.

Production on Azure uses SQLite persisted on Azure Files. Local development
uses a SQLite file by default; SQLAlchemy can still use other configured URLs.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from bovi_api.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _is_sqlite_url(database_url: str) -> bool:
    """Return whether the configured SQLAlchemy URL points at SQLite."""
    return database_url.startswith("sqlite")


def _create_engine(database_url: str) -> AsyncEngine:
    """Create the async engine with backend-appropriate connection settings."""
    kwargs: dict[str, object] = {
        "echo": False,
        # Validate pooled connections before handing them out. This is mostly
        # useful for server databases, but harmless for SQLite and keeps the
        # engine policy consistent if another SQLAlchemy URL is configured.
        "pool_pre_ping": True,
    }
    if _is_sqlite_url(database_url):
        kwargs["connect_args"] = {"timeout": 30}

    engine = create_async_engine(database_url, **kwargs)
    if _is_sqlite_url(database_url):
        _configure_sqlite(engine, database_url)
    return engine


def _is_azure_files_sqlite(database_url: str) -> bool:
    """Return whether SQLite is backed by the Azure Files mount used in prod."""
    return database_url.startswith("sqlite+aiosqlite:////data/")


def _configure_sqlite(engine: AsyncEngine, database_url: str) -> None:
    """Apply SQLite pragmas to every new DB-API connection."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            if not _is_azure_files_sqlite(database_url):
                try:
                    cursor.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    # WAL is a performance preference, not a startup requirement.
                    # Some mounted filesystems reject or lock during journal changes.
                    pass
        finally:
            cursor.close()


def _get_engine():
    """Return a shared async engine, creating it lazily."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            msg = "DATABASE_URL is not configured"
            raise RuntimeError(msg)
        _engine = _create_engine(settings.database_url)
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
    """Create all tables for local development; Azure deployments use Alembic."""
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def check_database_connection(session: AsyncSession | None = None) -> dict[str, str]:
    """Run a lightweight database readiness check."""
    try:
        if session is not None:
            await session.execute(text("SELECT 1"))
            bind = session.get_bind()
            return {"status": "ok", "dialect": bind.dialect.name}

        engine = _get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "dialect": engine.dialect.name}
    except (RuntimeError, SQLAlchemyError) as exc:
        return {"status": "error", "detail": str(exc)}


async def dispose_engine() -> None:
    """Dispose the engine and close all connections."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
