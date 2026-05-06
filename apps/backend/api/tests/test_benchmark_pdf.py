"""Tests for benchmark PDF report generation (v2 three-section)."""

from __future__ import annotations


def _stats_v2() -> dict:
    block = {
        "overall": {"pearson": 0.97, "rmse": 45.0, "mae": 38.0, "mape": 4.3, "n": 20},
        "by_parity": {
            "1": {"pearson": 0.96, "rmse": 48.0, "mae": 40.0, "mape": 4.8, "n": 10},
            "3+": {"pearson": 0.98, "rmse": 42.0, "mae": 36.0, "mape": 3.9, "n": 10},
        },
    }
    return {
        "version": 2,
        "challenger_vs_aly": block,
        "benchmark_vs_aly": block,
        "challenger_vs_benchmark": block,
        "failed_count": 0,
    }


def _yields() -> tuple[dict, dict, dict, dict]:
    challenger = {str(i): 8000.0 + i * 10 for i in range(20)}
    benchmark = {str(i): 8000.0 + i * 12 for i in range(20)}
    actual = {str(i): 7900.0 + i * 11 for i in range(20)}
    parities = {str(i): (1 if i < 10 else 3) for i in range(20)}
    return challenger, benchmark, actual, parities


def test_generate_report_pdf_returns_pdf_bytes():
    from bovi_api.benchmark_pdf import generate_report_pdf

    challenger, benchmark, actual, parities = _yields()
    result = generate_report_pdf(
        stats=_stats_v2(),
        challenger_yields=challenger,
        benchmark_yields=benchmark,
        actual_yields=actual,
        parities=parities,
        challenge_name="Test cohort",
        challenge_source="preset",
        submission_type="bovi_model",
        challenger_label="wood",
        benchmark_label="tim",
    )
    assert isinstance(result, bytes)
    assert result[:4] == b"%PDF"
    assert len(result) > 5000


def test_generate_report_pdf_handles_missing_blocks():
    from bovi_api.benchmark_pdf import generate_report_pdf

    challenger, benchmark, actual, parities = _yields()
    stats = {"version": 2, "failed_count": 0}
    result = generate_report_pdf(
        stats=stats,
        challenger_yields=challenger,
        benchmark_yields=benchmark,
        actual_yields=actual,
        parities=parities,
        challenger_label="own method",
        benchmark_label="tim",
    )
    assert result[:4] == b"%PDF"
