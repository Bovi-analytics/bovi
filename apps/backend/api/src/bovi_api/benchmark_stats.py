"""Statistics for comparing yields against ground-truth and benchmark values."""

from __future__ import annotations

import math
from typing import Any, cast

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _stats_for(pairs: list[tuple[float, float]]) -> dict:
    """Compute Pearson, RMSE, MAE, MAPE on (a, b) pairs."""
    if not pairs:
        return {"pearson": None, "rmse": None, "mae": None, "mape": None, "n": 0}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    if len(pairs) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        corr = None
    else:
        result = pearsonr(a, b)
        corr = round(float(cast(Any, result)[0]), 6)
    rmse = math.sqrt(mean_squared_error(b, a))
    mae = mean_absolute_error(b, a)
    mape = sum(abs((y - x) / y) for x, y in zip(a, b) if y != 0) / len(pairs) * 100
    return {
        "pearson": corr,
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 3),
        "n": len(pairs),
    }


def _block(
    a: dict[str, float],
    b: dict[str, float],
    parities: dict[str, int],
) -> dict:
    """Build {overall, by_parity} comparing a vs b on common cow_ids."""
    common = [cid for cid in a if cid in b]
    all_pairs = [(a[cid], b[cid]) for cid in common]
    parity_groups: dict[str, list[tuple[float, float]]] = {}
    for cid in common:
        p = parities.get(cid, 1)
        key = str(p) if p <= 2 else "3+"
        parity_groups.setdefault(key, []).append((a[cid], b[cid]))
    return {
        "overall": _stats_for(all_pairs),
        "by_parity": {k: _stats_for(v) for k, v in sorted(parity_groups.items())},
    }


def calculate_comparison_stats_v2(
    challenger_yields: dict[str, float],
    benchmark_yields: dict[str, float],
    actual_yields: dict[str, float],
    parities: dict[str, int],
) -> dict:
    """Three-axis benchmarking stats: challenger/benchmark vs ALY plus inter-model agreement.

    Args:
        challenger_yields: 305-day yields from the challenger model.
        benchmark_yields: 305-day yields from the benchmark model (server-run).
        actual_yields: ground-truth ALY (daily milk meter cumulative).
        parities: {cow_id: parity_int} for grouping.

    Returns:
        dict with version=2 and three blocks plus failed_count.

    """
    failed_count = sum(1 for cid in challenger_yields if cid not in actual_yields)
    return {
        "version": 2,
        "challenger_vs_aly": _block(challenger_yields, actual_yields, parities),
        "benchmark_vs_aly": _block(benchmark_yields, actual_yields, parities),
        "challenger_vs_benchmark": _block(challenger_yields, benchmark_yields, parities),
        "failed_count": failed_count,
    }


def calculate_comparison_stats(
    submitted: dict[str, float],
    reference: dict[str, float],
    parities: dict[str, int],
    actual_yields: dict[str, float] | None = None,
) -> dict:
    """Legacy two-block API kept for back-compat with old tests/old submissions."""
    vs_ref = _block(submitted, reference, parities)
    common = [cid for cid in submitted if cid in reference]
    failed_count = len(submitted) - len(common)
    out: dict = {
        "overall": vs_ref["overall"],
        "by_parity": vs_ref["by_parity"],
        "failed_count": failed_count,
    }
    if actual_yields:
        out["vs_aly"] = _block(submitted, actual_yields, parities)
    return out
