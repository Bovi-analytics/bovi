"""Smoke tests for Alembic migrations against a fresh SQLite database."""

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from bovi_api import migrations
from bovi_api.app import create_app
from bovi_api.settings import get_settings
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError


def test_alembic_upgrade_head_creates_all_runtime_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "bovi.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()

    cfg = Config()
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).parents[1] / "src" / "bovi_api" / "alembic"),
    )

    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert {
            "alembic_version",
            "fitting_results",
            "herd_profiles",
            "challenges",
            "submissions",
        }.issubset(set(inspector.get_table_names()))
        assert {column["name"] for column in inspector.get_columns("fitting_results")} == {
            "model_type",
            "source_app",
            "input_data",
            "output_data",
            "metadata_extra",
            "id",
            "created_at",
        }
        assert "run_options" in {column["name"] for column in inspector.get_columns("submissions")}
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_run_migrations_retries_sqlite_lock(monkeypatch, tmp_path):
    db_path = tmp_path / "locked.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    calls = 0
    lock_path = db_path.with_name("locked.db.migration.lock")
    sleeps: list[float] = []

    def upgrade(_cfg: Config, _revision: str) -> None:
        nonlocal calls
        calls += 1
        assert lock_path.exists()
        if calls < 3:
            raise OperationalError(
                "CREATE TABLE alembic_version",
                {},
                sqlite3.OperationalError("database is locked"),
            )

    monkeypatch.setattr(migrations.command, "upgrade", upgrade)
    monkeypatch.setattr(migrations, "sleep", sleeps.append)

    try:
        migrations.run_migrations()

        assert calls == 3
        assert not lock_path.exists()
        assert sleeps == [migrations._MIGRATION_LOCK_RETRY_SECONDS] * 2
    finally:
        get_settings.cache_clear()


def test_run_migrations_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()

    try:
        try:
            migrations.run_migrations()
        except RuntimeError as exc:
            assert str(exc) == "DATABASE_URL is required to run migrations"
        else:
            raise AssertionError("Expected run_migrations to require DATABASE_URL")
    finally:
        get_settings.cache_clear()


def test_create_app_does_not_run_migrations(monkeypatch, tmp_path):
    db_path = tmp_path / "startup.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()

    def fail_upgrade(_cfg: Config, _revision: str) -> None:
        raise AssertionError("create_app must not run migrations")

    monkeypatch.setattr(migrations.command, "upgrade", fail_upgrade)

    try:
        app = create_app()

        assert app.title == "Bovi API"
    finally:
        get_settings.cache_clear()
