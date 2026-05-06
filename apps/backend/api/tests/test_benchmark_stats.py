"""Tests for benchmark statistics calculation."""

import pytest
from bovi_api.benchmark_stats import calculate_comparison_stats


def _make_identical(n: int = 10) -> tuple[dict, dict, dict]:
    """Perfect submission: submitted == reference."""
    ids = [str(i) for i in range(n)]
    yields = {cid: 8000.0 + i * 100 for i, cid in enumerate(ids)}
    parities = {cid: (i % 3) + 1 for i, cid in enumerate(ids)}
    return yields, dict(yields), parities


def test_identical_submission_has_pearson_one():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["pearson"] == pytest.approx(1.0, abs=1e-6)


def test_identical_submission_has_zero_rmse():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_stats_contain_required_keys():
    submitted, reference, parities = _make_identical(20)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert "overall" in stats
    assert "by_parity" in stats
    assert "failed_count" in stats
    for key in ("pearson", "rmse", "mae", "mape", "n"):
        assert key in stats["overall"]


def test_parity_grouping_combines_3plus():
    ids = ["a", "b", "c", "d"]
    yields = {cid: 8000.0 for cid in ids}
    parities = {"a": 1, "b": 2, "c": 3, "d": 4}
    stats = calculate_comparison_stats(yields, dict(yields), parities)
    assert "3+" in stats["by_parity"]
    assert "3" not in stats["by_parity"]
    assert "4" not in stats["by_parity"]
    assert stats["by_parity"]["3+"]["n"] == 2


def test_missing_reference_cow_counted_as_failed():
    submitted = {"cow1": 8000.0, "cow2": 9000.0}
    reference = {"cow1": 8000.0}  # cow2 missing
    parities = {"cow1": 1, "cow2": 1}
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["failed_count"] == 1
    assert stats["overall"]["n"] == 1


def test_n_reflects_matched_cows():
    submitted, reference, parities = _make_identical(15)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert stats["overall"]["n"] == 15


def test_vs_aly_block_added_when_actuals_provided():
    submitted, reference, parities = _make_identical(20)
    actuals = {cid: v - 50 for cid, v in submitted.items()}
    stats = calculate_comparison_stats(submitted, reference, parities, actual_yields=actuals)
    assert "vs_aly" in stats
    assert "overall" in stats["vs_aly"]
    assert "by_parity" in stats["vs_aly"]
    assert stats["vs_aly"]["overall"]["n"] == 20
    # Top-level vs-TIM block remains unchanged
    assert stats["overall"]["pearson"] == pytest.approx(1.0, abs=1e-6)


def test_vs_aly_absent_when_no_actuals():
    submitted, reference, parities = _make_identical(10)
    stats = calculate_comparison_stats(submitted, reference, parities)
    assert "vs_aly" not in stats


def test_v2_three_blocks_present():
    from bovi_api.benchmark_stats import calculate_comparison_stats_v2

    challenger = {str(i): 8000.0 + i * 10 for i in range(10)}
    benchmark = {str(i): 8000.0 + i * 12 for i in range(10)}
    actual = {str(i): 7900.0 + i * 11 for i in range(10)}
    parities = {str(i): 1 for i in range(10)}
    stats = calculate_comparison_stats_v2(
        challenger_yields=challenger,
        benchmark_yields=benchmark,
        actual_yields=actual,
        parities=parities,
    )
    assert stats["version"] == 2
    assert "challenger_vs_aly" in stats
    assert "benchmark_vs_aly" in stats
    assert "challenger_vs_benchmark" in stats
    assert stats["challenger_vs_aly"]["overall"]["n"] == 10
