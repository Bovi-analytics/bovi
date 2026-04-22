"""Alembic migration environment.

Migrations use a synchronous engine regardless of the runtime driver, so this
file can be invoked from any context (CLI, lifespan, uvicorn import) without
nesting ``asyncio.run`` inside an already-running event loop.
"""

from logging.config import fileConfig

from alembic import context
from bovi_api.models import FittingResult, HerdProfile  # noqa: F401 — registers tables
from bovi_api.settings import get_settings
from sqlalchemy import create_engine
from sqlmodel import SQLModel

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _sync_url(async_url: str) -> str:
    """Return a sync-driver equivalent of an async SQLAlchemy URL."""
    return async_url.replace("sqlite+aiosqlite://", "sqlite://").replace(
        "postgresql+asyncpg://", "postgresql+psycopg://"
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    settings = get_settings()
    context.configure(
        url=_sync_url(settings.database_url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using a synchronous engine."""
    settings = get_settings()
    engine = create_engine(_sync_url(settings.database_url))
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
