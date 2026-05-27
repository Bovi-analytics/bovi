"""Command entrypoint for applying API database migrations."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import sleep

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from bovi_api.settings import Settings, get_settings

_ALEMBIC_DIR = Path(__file__).parent / "alembic"
_MIGRATION_LOCK_RETRIES = 5
_MIGRATION_LOCK_RETRY_SECONDS = 2.0
_MIGRATION_FILE_LOCK_STALE_SECONDS = 15.0
_MIGRATION_FILE_LOCK_RETRY_SECONDS = 0.5


def _is_sqlite_locked(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _migration_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    return cfg


def _sqlite_migration_lock_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite"):
        return None
    database_path = make_url(database_url).database
    if not database_path or database_path == ":memory:":
        return None
    path = Path(database_path)
    return path.with_name(f"{path.name}.migration.lock")


@contextmanager
def _migration_lock(database_url: str) -> Iterator[None]:
    lock_path = _sqlite_migration_lock_path(database_url)
    if lock_path is None:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_seconds > _MIGRATION_FILE_LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            sleep(_MIGRATION_FILE_LOCK_RETRY_SECONDS)

    with os.fdopen(fd, "w") as lock_file:
        lock_file.write(f"{os.getpid()}\n")
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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

    with _migration_lock(settings.database_url):
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
