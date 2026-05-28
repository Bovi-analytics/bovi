"""Blob-backed runtime assets for the lactation autoencoder Function App."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from azure.storage.blob import BlobServiceClient
from settings import Settings

logger = logging.getLogger("lactation_autoencoder.assets")

REQUIRED_ASSET_SUFFIXES = (
    "config/config.yaml",
    "inputs/inference/pkl/event_to_idx_dict.pkl",
    "weights/autoencoder/saved_model.pb",
    "weights/autoencoder/variables/variables.index",
)
_CACHE_MARKER = ".download-complete"
_PYPROJECT_TOML = """[project]
name = "bovi"
version = "0.1.0"
description = "Runtime cache root for Bovi autoencoder assets"
authors = [{ name = "Bovi", email = "noreply@bovi-analytics.local" }]
dependencies = []
"""


class ModelAssetError(RuntimeError):
    """Raised when model assets cannot be prepared for inference."""


@dataclass(frozen=True)
class ModelAssetPaths:
    """Local paths used to initialize the autoencoder runtime."""

    project_root: Path
    config_path: Path


BlobServiceFactory = Callable[[str], Any]


def ensure_model_assets(
    settings: Settings,
    *,
    blob_service_factory: BlobServiceFactory = BlobServiceClient.from_connection_string,
) -> ModelAssetPaths:
    """Ensure autoencoder assets are available in the local runtime cache."""
    cache_root = Path(settings.autoencoder_model_local_root)
    prefix = _clean_prefix(settings.autoencoder_model_prefix)
    config_path = cache_root / prefix / "config/config.yaml"
    marker_path = cache_root / prefix / _CACHE_MARKER

    if _required_assets_exist(cache_root, prefix):
        _ensure_runtime_project_file(cache_root)
        return ModelAssetPaths(project_root=cache_root, config_path=config_path)

    if not settings.azure_web_jobs_storage:
        missing = _missing_required_assets(cache_root, prefix)
        raise ModelAssetError(
            "AzureWebJobsStorage is required to load autoencoder model assets; "
            "local model asset root is incomplete; missing: " + ", ".join(missing)
        )

    logger.info(
        "Hydrating autoencoder model assets from blob container %s/%s",
        settings.autoencoder_model_container,
        prefix,
    )
    container_client = blob_service_factory(settings.azure_web_jobs_storage).get_container_client(
        settings.autoencoder_model_container
    )
    blobs = list(container_client.list_blobs(name_starts_with=f"{prefix}/"))
    if not blobs:
        raise ModelAssetError(
            f"No autoencoder model assets found in {settings.autoencoder_model_container}/{prefix}."
        )

    for blob in blobs:
        blob_name = str(blob.name)
        destination = _safe_cache_path(cache_root, blob_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            container_client.get_blob_client(blob_name).download_blob().readall()
        )

    missing = _missing_required_assets(cache_root, prefix)
    if missing:
        raise ModelAssetError(
            "Autoencoder model asset cache is incomplete; missing: " + ", ".join(missing)
        )

    _ensure_runtime_project_file(cache_root)
    marker_path.write_text(f"downloaded={len(blobs)}\n", encoding="utf-8")
    return ModelAssetPaths(project_root=cache_root, config_path=config_path)


def _clean_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    if not cleaned:
        raise ModelAssetError("AUTOENCODER_MODEL_PREFIX must not be empty.")
    return cleaned


def _required_assets_exist(cache_root: Path, prefix: str) -> bool:
    return not _missing_required_assets(cache_root, prefix)


def _missing_required_assets(cache_root: Path, prefix: str) -> list[str]:
    missing = []
    for suffix in REQUIRED_ASSET_SUFFIXES:
        asset = cache_root / prefix / suffix
        if not asset.exists():
            missing.append(f"{prefix}/{suffix}")
    variables_dir = cache_root / prefix / "weights/autoencoder/variables"
    if not any(variables_dir.glob("variables.data-*")):
        missing.append(f"{prefix}/weights/autoencoder/variables/variables.data-*")
    return missing


def _safe_cache_path(cache_root: Path, blob_name: str) -> Path:
    relative = PurePosixPath(blob_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelAssetError(f"Unsafe blob name for model asset cache: {blob_name}")
    return cache_root / Path(*relative.parts)


def _ensure_runtime_project_file(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    project_file = cache_root / "pyproject.toml"
    if not project_file.exists():
        project_file.write_text(_PYPROJECT_TOML, encoding="utf-8")
