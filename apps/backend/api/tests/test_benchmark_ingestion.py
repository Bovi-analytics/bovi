"""Tests for benchmark CSV ingestion."""

import pytest
from bovi_api.benchmark_ingestion import parse_submission_csv, parse_test_day_csv


def test_parse_valid_csv():
    csv_bytes = b"cow_id,yield_305day\ncow1,8500.0\ncow2,9200.5\n"
    result = parse_submission_csv(csv_bytes)
    assert result == {"cow1": 8500.0, "cow2": 9200.5}


def test_parse_skips_invalid_rows_and_returns_failed():
    csv_bytes = b"cow_id,yield_305day\ncow1,8500.0\ncow2,not_a_number\ncow3,7000.0\n"
    result, failed = parse_submission_csv(csv_bytes, return_failed=True)
    assert result == {"cow1": 8500.0, "cow3": 7000.0}
    assert failed == ["cow2"]


def test_parse_missing_required_column_raises():
    csv_bytes = b"cow_id,wrong_column\ncow1,8500.0\n"
    with pytest.raises(ValueError, match="yield_305day"):
        parse_submission_csv(csv_bytes)


def test_parse_empty_file_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_submission_csv(b"")


def test_parse_negative_yield_raises():
    csv_bytes = b"cow_id,yield_305day\ncow1,-100.0\n"
    with pytest.raises(ValueError, match="negative"):
        parse_submission_csv(csv_bytes)


def test_parse_test_day_csv_preserves_herd_id_and_parity():
    csv_bytes = (
        b"cow_id,herd_id,parity,dim,milk_kg\ncow1,2942694,2,10,25.5\ncow1,2942694,2,20,28.0\n"
    )

    result = parse_test_day_csv(csv_bytes)

    assert result == {
        "cow1": {
            "herd_id": 2942694,
            "parity": 2,
            "dim": [10, 20],
            "milk_kg": [25.5, 28.0],
        }
    }
