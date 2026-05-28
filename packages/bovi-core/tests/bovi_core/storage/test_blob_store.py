"""Tests for the config-free BlobStore helper."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from bovi_core.storage import BlobStore


@dataclass
class _StoredBlob:
    data: bytes
    metadata: dict[str, str]
    content_type: str | None
    content_encoding: str | None
    etag: str


class _Download:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _BlobClient:
    def __init__(self, store: dict[str, _StoredBlob], path: str) -> None:
        self._store = store
        self._path = path

    def upload_blob(self, data, *, overwrite, metadata, content_settings) -> None:
        if not overwrite and self._path in self._store:
            raise ResourceExistsError(message="exists")
        payload = bytes(data)
        self._store[self._path] = _StoredBlob(
            data=payload,
            metadata=metadata,
            content_type=content_settings.content_type,
            content_encoding=content_settings.content_encoding,
            etag=f"etag-{len(self._store) + 1}",
        )

    def get_blob_properties(self) -> dict:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        stored = self._store[self._path]
        return {"etag": stored.etag}

    def download_blob(self, **kwargs) -> _Download:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        data = self._store[self._path].data
        offset = kwargs.get("offset")
        length = kwargs.get("length")
        if offset is not None:
            data = data[offset : offset + length if length is not None else None]
        return _Download(data)

    def exists(self) -> bool:
        return self._path in self._store

    def delete_blob(self, **_kwargs) -> bool:
        if self._path not in self._store:
            raise ResourceNotFoundError(message="missing")
        del self._store[self._path]
        return True


class _ContainerClient:
    def __init__(self) -> None:
        self.store: dict[str, _StoredBlob] = {}

    def get_blob_client(self, *, blob: str) -> _BlobClient:
        return _BlobClient(self.store, blob)


@pytest.fixture
def container() -> _ContainerClient:
    return _ContainerClient()


@pytest.fixture
def blob_store(container: _ContainerClient) -> BlobStore:
    return BlobStore(container, account_name="acct", container_name="container")


def test_upload_bytes_stores_content_settings_metadata_and_checksum(
    blob_store: BlobStore, container: _ContainerClient
) -> None:
    result = blob_store.upload_bytes(
        "path/file.csv",
        b"cow,yield\n1,42\n",
        content_type="text/csv",
        metadata={"artifact_kind": "submission_results_csv"},
    )

    stored = container.store["path/file.csv"]
    assert result.size_bytes == len(stored.data)
    assert result.sha256 == stored.metadata["sha256"]
    assert stored.metadata["artifact_kind"] == "submission_results_csv"
    assert stored.content_type == "text/csv"
    assert stored.content_encoding is None
    assert result.etag == "etag-1"


def test_upload_bytes_does_not_overwrite_by_default(blob_store: BlobStore) -> None:
    blob_store.upload_bytes("same", b"one", content_type="text/plain")

    with pytest.raises(ResourceExistsError):
        blob_store.upload_bytes("same", b"two", content_type="text/plain")


def test_upload_json_gzip_round_trips(blob_store: BlobStore, container: _ContainerClient) -> None:
    payload = {"cow": {"dim": [1, 2], "milk_kg": [30.5, 31.0]}}

    result = blob_store.upload_json_gzip("payload.json.gz", payload, metadata={"schema": "1"})

    stored = container.store["payload.json.gz"]
    assert stored.content_type == "application/json"
    assert stored.content_encoding == "gzip"
    assert stored.metadata["schema"] == "1"
    assert json.loads(gzip.decompress(stored.data)) == payload
    assert blob_store.download_json_gzip(result.blob_path) == payload


def test_download_json_gzip_accepts_transparently_decoded_payload(
    blob_store: BlobStore, container: _ContainerClient
) -> None:
    payload = {"cow": {"dim": [1, 2], "milk_kg": [30.5, 31.0]}}
    container.store["payload.json.gz"] = _StoredBlob(
        data=json.dumps(payload).encode("utf-8"),
        metadata={},
        content_type="application/json",
        content_encoding="gzip",
        etag="etag-1",
    )

    assert blob_store.download_json_gzip("payload.json.gz") == payload


def test_delete_if_exists(blob_store: BlobStore) -> None:
    blob_store.upload_bytes("delete-me", b"x", content_type="text/plain")

    assert blob_store.delete_if_exists("delete-me") is True
    assert blob_store.exists("delete-me") is False
    assert blob_store.delete_if_exists("delete-me") is False
