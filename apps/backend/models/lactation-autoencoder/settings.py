"""Lactation autoencoder Function App settings."""

from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Deployment-only configuration. ML config is handled by bovi-core Config."""

    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": find_dotenv(), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
