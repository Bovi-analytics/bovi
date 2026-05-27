"""Small Azure Blob Storage wrapper for service code.

This module intentionally does not depend on :class:`bovi_core.config.Config`.
The ML/Databricks-oriented helpers in ``bovi_core.utils.blob_utils`` still
exist for legacy callers; this class is the config-free primitive used by API
services that already have explicit settings.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings


@dataclass(frozen=True)
class BlobWriteResult:
    """Metadata returned after writing a blob."""

    blob_path: str
    size_bytes: int
    sha256: str
    etag: str | None
    content_type: str
    content_encoding: str | None = None


class BlobStore:
    """Config-free Azure Blob Storage helper.

    The implementation is synchronous because the repository currently depends
    on ``azure-storage-blob`` rather than ``azure-storage-blob.aio``. FastAPI
    callers should run operations through ``run_in_threadpool`` when called
    from async route handlers.
    """

    def __init__(self, container_client: Any, *, account_name: str, container_name: str) -> None:
        self.container_client = container_client
        self.account_name = account_name
        self.container_name = container_name

    @classmethod
    def from_connection_parts(
        cls,
        *,
        account_name: str,
        account_key: str,
        container_name: str,
    ) -> "BlobStore":
        """Create a store from explicit Azure Storage credentials."""
        connection_string = (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={account_name};"
            f"AccountKey={account_key};"
            "EndpointSuffix=core.windows.net"
        )
        return cls.from_connection_string(
            connection_string=connection_string,
            account_name=account_name,
            container_name=container_name,
        )

    @classmethod
    def from_connection_string(
        cls,
        *,
        connection_string: str,
        account_name: str,
        container_name: str,
    ) -> "BlobStore":
        """Create a store from an Azure Storage connection string."""
        service_client = BlobServiceClient.from_connection_string(connection_string)
        return cls(
            service_client.get_container_client(container_name),
            account_name=account_name,
            container_name=container_name,
        )

    def upload_bytes(
        self,
        blob_path: str,
        data: bytes,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
        content_encoding: str | None = None,
        overwrite: bool = False,
    ) -> BlobWriteResult:
        """Upload bytes and return persisted blob metadata."""
        sha256 = hashlib.sha256(data).hexdigest()
        clean_metadata = dict(metadata or {})
        clean_metadata.setdefault("sha256", sha256)

        blob_client = self.container_client.get_blob_client(blob=blob_path)
        blob_client.upload_blob(
            data,
            overwrite=overwrite,
            metadata=clean_metadata,
            content_settings=ContentSettings(
                content_type=content_type,
                content_encoding=content_encoding,
            ),
        )
        props = blob_client.get_blob_properties()
        etag = _extract_etag(props)
        return BlobWriteResult(
            blob_path=blob_path,
            size_bytes=len(data),
            sha256=sha256,
            etag=etag,
            content_type=content_type,
            content_encoding=content_encoding,
        )

    def upload_json_gzip(
        self,
        blob_path: str,
        payload: Any,
        *,
        metadata: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> BlobWriteResult:
        """Serialize a payload as JSON, gzip it, and upload it."""
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return self.upload_bytes(
            blob_path,
            gzip.compress(raw),
            content_type="application/json",
            content_encoding="gzip",
            metadata=metadata,
            overwrite=overwrite,
        )

    def download_bytes(
        self,
        blob_path: str,
        *,
        offset: int | None = None,
        length: int | None = None,
    ) -> bytes:
        """Download a blob as bytes."""
        blob_client = self.container_client.get_blob_client(blob=blob_path)
        kwargs = {}
        if offset is not None:
            kwargs["offset"] = offset
        if length is not None:
            kwargs["length"] = length
        return blob_client.download_blob(**kwargs).readall()

    def download_json_gzip(self, blob_path: str) -> Any:
        """Download a gzip-compressed JSON blob."""
        return json.loads(gzip.decompress(self.download_bytes(blob_path)))

    def exists(self, blob_path: str) -> bool:
        """Return whether a blob exists."""
        blob_client = self.container_client.get_blob_client(blob=blob_path)
        try:
            return bool(blob_client.exists())
        except AttributeError:
            try:
                blob_client.get_blob_properties()
            except ResourceNotFoundError:
                return False
            return True

    def delete_if_exists(self, blob_path: str) -> bool:
        """Delete a blob if present, returning whether it existed."""
        blob_client = self.container_client.get_blob_client(blob=blob_path)
        try:
            blob_client.delete_blob(delete_snapshots="include")
            return True
        except ResourceNotFoundError:
            return False


def _extract_etag(props: Any) -> str | None:
    etag = getattr(props, "etag", None)
    if etag is not None:
        return str(etag)
    if isinstance(props, dict) and props.get("etag") is not None:
        return str(props["etag"])
    return None
