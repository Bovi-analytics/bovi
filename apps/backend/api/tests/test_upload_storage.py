"""Tests for raw upload storage helpers."""

from datetime import datetime, timezone

import pytest
from bovi_api.settings import Settings
from bovi_api.upload_storage import (
    UploadStorageError,
    build_upload_blob_path,
    upload_csv_to_blob,
)


def test_build_upload_blob_path_is_canonical_and_sanitized():
    path = build_upload_blob_path(
        "benchmark_submission_results",
        "../My Results (final).csv",
        upload_id="abc123",
        now=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )

    assert path == "uploads/2026/05/21/abc123/benchmark_submission_results/My_Results_final_.csv"


def test_upload_csv_to_blob_requires_connection_string():
    with pytest.raises(UploadStorageError, match="CONNECTION_STRING"):
        upload_csv_to_blob(
            b"cow_id,yield_305day\n1,8000\n",
            filename="results.csv",
            content_type="text/csv",
            action_type="benchmark_submission_results",
            settings=Settings(connection_string=None),
        )
