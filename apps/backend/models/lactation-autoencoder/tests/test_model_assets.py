"""Tests for blob-backed autoencoder model assets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from model_assets import ModelAssetError, ensure_model_assets
from settings import Settings


class _FakeDownload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _FakeBlobClient:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def download_blob(self) -> _FakeDownload:
        return _FakeDownload(self._data)


class _FakeContainerClient:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs
        self.downloaded: list[str] = []

    def list_blobs(self, name_starts_with: str):
        return [
            SimpleNamespace(name=name)
            for name in sorted(self.blobs)
            if name.startswith(name_starts_with)
        ]

    def get_blob_client(self, blob: str) -> _FakeBlobClient:
        self.downloaded.append(blob)
        return _FakeBlobClient(self.blobs[blob])


class _FakeBlobService:
    def __init__(self, container: _FakeContainerClient) -> None:
        self.container = container

    def get_container_client(self, container_name: str) -> _FakeContainerClient:
        assert container_name == "model-assets"
        return self.container


def _settings(tmp_path) -> Settings:
    return Settings(
        azure_web_jobs_storage="UseDevelopmentStorage=true",
        autoencoder_model_cache_dir=str(tmp_path),
    )


def _required_blobs(prefix: str = "data/models/lactation_autoencoder/versions/v15"):
    return {
        f"{prefix}/config/config.yaml": b"experiment_name: lactation_autoencoder\n",
        f"{prefix}/inputs/inference/pkl/event_to_idx_dict.pkl": b"pickle",
        f"{prefix}/weights/autoencoder/saved_model.pb": b"model",
        f"{prefix}/weights/autoencoder/variables/variables.index": b"index",
        f"{prefix}/weights/autoencoder/variables/variables.data-00000-of-00001": b"data",
    }


def test_settings_accepts_azure_web_jobs_storage_alias(tmp_path):
    settings = Settings.model_validate(
        {
            "AzureWebJobsStorage": "UseDevelopmentStorage=true",
            "autoencoder_model_cache_dir": str(tmp_path),
        }
    )

    assert settings.azure_web_jobs_storage == "UseDevelopmentStorage=true"
    assert settings.autoencoder_model_cache_dir == str(tmp_path)


def test_ensure_model_assets_downloads_prefix_to_cache(tmp_path):
    container = _FakeContainerClient(_required_blobs())

    paths = ensure_model_assets(
        _settings(tmp_path),
        blob_service_factory=lambda _: _FakeBlobService(container),
    )

    assert paths.project_root == tmp_path
    assert paths.config_path == (
        tmp_path / "data/models/lactation_autoencoder/versions/v15/config/config.yaml"
    )
    assert paths.config_path.exists()
    assert (tmp_path / "pyproject.toml").exists()
    assert sorted(container.downloaded) == sorted(container.blobs)


def test_ensure_model_assets_reuses_complete_cache(tmp_path):
    container = _FakeContainerClient(_required_blobs())
    settings = _settings(tmp_path)

    ensure_model_assets(settings, blob_service_factory=lambda _: _FakeBlobService(container))
    container.downloaded.clear()

    ensure_model_assets(settings, blob_service_factory=lambda _: _FakeBlobService(container))

    assert container.downloaded == []


def test_ensure_model_assets_requires_storage_connection(tmp_path):
    settings = Settings(
        azure_web_jobs_storage=None,
        autoencoder_model_cache_dir=str(tmp_path),
    )

    with pytest.raises(ModelAssetError, match="AzureWebJobsStorage"):
        ensure_model_assets(settings, blob_service_factory=lambda _: None)


def test_ensure_model_assets_reports_incomplete_source_prefix(tmp_path):
    blobs = _required_blobs()
    blobs.pop("data/models/lactation_autoencoder/versions/v15/weights/autoencoder/saved_model.pb")
    container = _FakeContainerClient(blobs)

    with pytest.raises(ModelAssetError, match="saved_model.pb"):
        ensure_model_assets(
            _settings(tmp_path),
            blob_service_factory=lambda _: _FakeBlobService(container),
        )
