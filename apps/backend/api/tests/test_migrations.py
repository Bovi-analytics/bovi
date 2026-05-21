"""Smoke tests for Alembic migrations against a fresh SQLite database."""

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from bovi_api import app as app_module
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
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_run_migrations_retries_sqlite_lock(monkeypatch, tmp_path):
    db_path = tmp_path / "locked.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    calls = 0
    sleeps: list[float] = []

    def upgrade(_cfg: Config, _revision: str) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OperationalError(
                "CREATE TABLE alembic_version",
                {},
                sqlite3.OperationalError("database is locked"),
            )

    monkeypatch.setattr(app_module.command, "upgrade", upgrade)
    monkeypatch.setattr(app_module, "sleep", sleeps.append)

    try:
        app_module._run_migrations()

        assert calls == 3
        assert sleeps == [app_module._MIGRATION_LOCK_RETRY_SECONDS] * 2
    finally:
        get_settings.cache_clear()
