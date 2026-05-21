"""Tests for API settings defaults and environment overrides."""

from bovi_api.settings import Settings


def test_database_url_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings.model_validate({})

    assert settings.database_url == "sqlite+aiosqlite:///./bovi.db"


def test_database_url_allows_explicit_empty_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")

    settings = Settings.model_validate({})

    assert settings.database_url == ""
