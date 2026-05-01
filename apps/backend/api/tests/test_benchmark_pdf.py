"""Tests for benchmark PDF report generation."""

from __future__ import annotations


def test_generate_report_pdf_returns_bytes():
    from bovi_api.benchmark_pdf import generate_report_pdf

    stats = {
        "overall": {"pearson": 0.97, "rmse": 45.0, "mae": 38.0, "mape": 4.3, "n": 100},
        "by_parity": {
            "1": {"pearson": 0.96, "rmse": 48.0, "mae": 40.0, "mape": 4.8, "n": 50},
            "3+": {"pearson": 0.98, "rmse": 42.0, "mae": 36.0, "mape": 3.9, "n": 50},
        },
        "failed_count": 0,
    }
    submitted = {str(i): 8000.0 + i * 10 for i in range(20)}
    reference = {str(i): 8000.0 + i * 12 for i in range(20)}
    bovi_y = {str(i): 8000.0 + i * 11 for i in range(20)}

    result = generate_report_pdf(
        stats=stats,
        submitted_yields=submitted,
        reference_yields=reference,
        bovi_yields=bovi_y,
        flavor="all",
        challenge_dataset="aurora",
        challenge_size="small",
    )
    assert isinstance(result, bytes)
    assert len(result) > 1000  # non-trivial PDF
    assert result[:4] == b"%PDF"
