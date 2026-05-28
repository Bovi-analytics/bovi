"""Bovi Central API settings."""

from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central API configuration loaded from environment / .env file."""

    lactation_curves_url: str = "http://localhost:8001"
    lactation_autoencoder_url: str = "http://localhost:8002"
    database_url: str = "sqlite+aiosqlite:///./bovi.db"
    cors_origins: list[str] = ["http://localhost:3000"]
    connection_string: str | None = (
        None  # Azure Storage connection string (CONNECTION_STRING env var)
    )
    storage_account_name_icar: str | None = None
    storage_account_key_icar: str | None = None
    storage_account_container_icar: str | None = None
    bovi_env: str = "local"
    upload_blob_prefix: str = "bovi/uploads"
    upload_max_bytes: int = 500 * 1024 * 1024
    allow_dairy_comp_uploads: bool = False

    model_config = {"env_file": find_dotenv(), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
