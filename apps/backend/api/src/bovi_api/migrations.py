"""Command entrypoint for applying API database migrations."""

import os
from pathlib import Path
from time import sleep

from alembic import command
from alembic.config import Config
from sqlalchemy.exc import OperationalError

from bovi_api.settings import Settings, get_settings

_ALEMBIC_DIR = Path(__file__).parent / "alembic"
_MIGRATION_LOCK_RETRIES = 5
_MIGRATION_LOCK_RETRY_SECONDS = 2.0


def _is_sqlite_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _migration_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    return cfg


def run_migrations() -> None:
    """Apply Alembic migrations up to head."""
    settings = get_settings()
    default_database_url = Settings.model_fields["database_url"].default
    if (
        not settings.database_url
        or settings.database_url == default_database_url
        and "DATABASE_URL" not in os.environ
    ):
        raise RuntimeError("DATABASE_URL is required to run migrations")

    for attempt in range(_MIGRATION_LOCK_RETRIES):
        try:
            command.upgrade(_migration_config(), "head")
            return
        except OperationalError as exc:
            if not _is_sqlite_locked(exc) or attempt == _MIGRATION_LOCK_RETRIES - 1:
                raise
            sleep(_MIGRATION_LOCK_RETRY_SECONDS)


if __name__ == "__main__":
    run_migrations()
