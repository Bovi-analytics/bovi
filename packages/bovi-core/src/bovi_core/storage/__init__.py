"""Storage clients shared across Bovi services."""

from .blob_store import BlobStore, BlobWriteResult

__all__ = ["BlobStore", "BlobWriteResult"]
