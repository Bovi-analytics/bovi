"""Smoke tests for Alembic migrations against a fresh SQLite database."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from bovi_api.settings import get_settings
from sqlalchemy import create_engine, inspect


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
