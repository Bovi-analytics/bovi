"""Lactation autoencoder Function App settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


def _default_model_cache_dir() -> str:
    """Return the repo root that contains local data/models assets."""
    return str(Path(__file__).resolve().parents[4])


class Settings(BaseSettings):
    """Deployment-only configuration. ML config is handled by bovi-core Config."""

    cors_origins: list[str] = ["http://localhost:3000"]
    azure_web_jobs_storage: str | None = Field(default=None, validation_alias="AzureWebJobsStorage")
    autoencoder_model_container: str = "model-assets"
    autoencoder_model_prefix: str = "data/models/lactation_autoencoder/versions/v15"
    autoencoder_model_cache_dir: str = Field(default_factory=_default_model_cache_dir)

    model_config = {
        "env_file": find_dotenv(),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
