"""Bovi Central API settings."""

from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central API configuration loaded from environment / .env file."""

    lactation_curves_url: str = "http://localhost:8001"
    lactation_autoencoder_url: str = "http://localhost:8002"
    database_url: str = ""
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": find_dotenv(), "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
