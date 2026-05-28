"""Tests for upload size validation helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from bovi_api.settings import Settings
from bovi_api.upload_limits import ensure_upload_file_size
from fastapi import HTTPException, UploadFile


def test_default_upload_limit_rejects_file_larger_than_500_mb():
    settings = Settings()
    assert settings.upload_max_bytes == 500 * 1024 * 1024

    oversized_file = cast(
        UploadFile,
        SimpleNamespace(
            filename="oversized.csv",
            size=501 * 1024 * 1024,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        ensure_upload_file_size(oversized_file, max_size=settings.upload_max_bytes)

    assert exc_info.value.status_code == 413
    detail = exc_info.value.detail
    assert "oversized.csv" in detail
    assert "500 MB upload limit" in detail
    assert "Split the file into smaller CSV files" in detail
