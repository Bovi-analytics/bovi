"""Statistics for comparing submitted 305-day yields against reference values."""

from __future__ import annotations

import math
from typing import Any, cast

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_comparison_stats(
    submitted: dict[str, float],
    reference: dict[str, float],
    parities: dict[str, int],
) -> dict:
    """Compute Pearson, RMSE, MAE, MAPE overall and per parity group (1, 2, 3+).

    Args:
        submitted: {cow_id: yield_305day} from the submission.
        reference: {cow_id: yield_305day} ICAR TIM reference values.
        parities: {cow_id: parity_int} from challenge.cow_metadata.

    Returns:
        dict with keys "overall", "by_parity", "failed_count".

    """
    common = [cid for cid in submitted if cid in reference]
    failed_count = len(submitted) - len(common)

    def _stats_for(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"pearson": None, "rmse": None, "mae": None, "mape": None, "n": 0}
        sub = [p[0] for p in pairs]
        ref = [p[1] for p in pairs]
        if len(pairs) < 2 or len(set(sub)) < 2 or len(set(ref)) < 2:
            corr = None
        else:
            result = pearsonr(sub, ref)
            corr = round(float(cast(Any, result)[0]), 6)
        rmse = math.sqrt(mean_squared_error(ref, sub))
        mae = mean_absolute_error(ref, sub)
        mape = sum(abs((r - s) / r) for s, r in zip(sub, ref) if r != 0) / len(pairs) * 100
        return {
            "pearson": corr,
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "mape": round(mape, 3),
            "n": len(pairs),
        }

    all_pairs = [(submitted[cid], reference[cid]) for cid in common]

    parity_groups: dict[str, list[tuple[float, float]]] = {}
    for cid in common:
        p = parities.get(cid, 1)
        key = str(p) if p <= 2 else "3+"
        parity_groups.setdefault(key, []).append((submitted[cid], reference[cid]))

    return {
        "overall": _stats_for(all_pairs),
        "by_parity": {k: _stats_for(v) for k, v in sorted(parity_groups.items())},
        "failed_count": failed_count,
    }
