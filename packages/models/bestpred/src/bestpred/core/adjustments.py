"""Fortran-compatible adjustment helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

NEW_3X: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.12, 0.09, 0.10, 0.00),
    (0.14, 0.10, 0.11, 0.00),
    (0.14, 0.10, 0.11, 0.00),
)
OLD_3X: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.20, 0.20, 0.20, 0.00),
    (0.17, 0.17, 0.17, 0.00),
    (0.15, 0.15, 0.15, 0.00),
)


@dataclass(frozen=True)
class ThreeXAdjustment:
    """Output of Fortran `adjust3X`."""

    test_factors: FloatArray
    lactation_factors: FloatArray
    partial_factors: FloatArray


def expected_daily_yield(
    *,
    trait: int,
    dim: int,
    mrd: int,
    daily_yield: FloatArray,
    herd_ratio: Sequence[float],
) -> float:
    """Port of Fortran `ymean()`.

    `daily_yield` is shaped `(4, maxlen)` and DIM is 1-based.
    """

    _validate_trait(trait)
    begin, end = _measurement_range(dim, trait, mrd)
    values = daily_yield[trait - 1, begin - 1 : end]
    return float(np.mean(values) * herd_ratio[trait - 1])


def adjust_3x(
    *,
    dims: Sequence[int],
    length: int,
    fresh_year: int,
    parity: int,
    milkings: Sequence[int],
    cumulative_yield: FloatArray,
    use_3x: int,
    maxlen: int,
) -> ThreeXAdjustment:
    """Port of the deterministic parts of Fortran `adjust3X`.

    `cumulative_yield` is shaped `(4, maxlen)` and stores `meanyld`.
    """

    if len(dims) != len(milkings):
        raise ValueError("dims and milkings must have the same length")
    if maxlen < 305:
        raise ValueError("maxlen must be at least 305 for 3X adjustment")
    if not 1 <= length <= maxlen:
        raise ValueError(f"length must be between 1 and {maxlen}")

    trait_count = 4
    test_factors = np.ones((trait_count, len(dims)), dtype=np.float64)
    lactation_factors = np.ones(trait_count, dtype=np.float64)
    partial_factors = np.ones(trait_count, dtype=np.float64)
    adjustment = np.array(
        _three_x_multipliers(use_3x=use_3x, fresh_year=fresh_year, parity=parity), dtype=np.float64
    )

    valid_pairs = [
        (dim, milking) for dim, milking in zip(dims, milkings, strict=True) if dim <= maxlen
    ]
    for index, (_dim, milking) in enumerate(zip(dims, milkings, strict=True)):
        if milking > 2:
            test_factors[:, index] = adjustment

    if not valid_pairs:
        return ThreeXAdjustment(
            test_factors=test_factors,
            lactation_factors=lactation_factors,
            partial_factors=partial_factors,
        )

    sorted_dims = sorted(dim for dim, _milking in valid_pairs)
    if sorted_dims[0] <= 0 or sorted_dims[0] > maxlen:
        return ThreeXAdjustment(
            test_factors=test_factors,
            lactation_factors=lactation_factors,
            partial_factors=partial_factors,
        )

    freq = {dim: milking for dim, milking in valid_pairs}
    lactation_3x = np.zeros(trait_count, dtype=np.float64)
    partial_3x = np.zeros(trait_count, dtype=np.float64)
    segment_adjustment = np.ones(trait_count, dtype=np.float64)
    begin_305 = 0
    begin_partial = 0
    end_305 = 0
    end_partial = 0

    for index, dim in enumerate(sorted_dims, start=1):
        end_305 = min(dim, 305)
        end_partial = min(dim, length)
        segment_adjustment = adjustment if freq[dim] > 2 else np.ones(trait_count, dtype=np.float64)
        if index > 1:
            lactation_3x -= cumulative_yield[:, begin_305 - 1] / segment_adjustment
            partial_3x -= cumulative_yield[:, begin_partial - 1] / segment_adjustment
        lactation_3x += cumulative_yield[:, end_305 - 1] / segment_adjustment
        partial_3x += cumulative_yield[:, end_partial - 1] / segment_adjustment
        begin_305 = end_305
        begin_partial = end_partial

    if end_305 < 305:
        lactation_3x += (
            cumulative_yield[:, 304] - cumulative_yield[:, end_305 - 1]
        ) / segment_adjustment
    if end_305 < length:
        partial_3x += (
            cumulative_yield[:, length - 1] - cumulative_yield[:, end_partial - 1]
        ) / segment_adjustment

    for trait_index in range(trait_count):
        if lactation_3x[trait_index] != 0.0:
            lactation_factors[trait_index] = (
                cumulative_yield[trait_index, 304] / lactation_3x[trait_index]
            )
        if partial_3x[trait_index] != 0.0:
            partial_factors[trait_index] = (
                cumulative_yield[trait_index, length - 1] / partial_3x[trait_index]
            )

    return ThreeXAdjustment(
        test_factors=test_factors,
        lactation_factors=lactation_factors,
        partial_factors=partial_factors,
    )


def _three_x_multipliers(
    *, use_3x: int, fresh_year: int, parity: int
) -> tuple[float, float, float, float]:
    parity_group = min(max(parity, 1), 3)
    if use_3x == 0:
        raw = (0.0, 0.0, 0.0, 0.0)
    elif use_3x == 1:
        raw = OLD_3X[parity_group - 1]
    elif use_3x == 2:
        raw = NEW_3X[parity_group - 1]
    elif use_3x == 3:
        phase_year = min(max(fresh_year, 1996), 1999)
        old_weight = (1999 - phase_year) / (1999 - 1996)
        new_weight = 1.0 - old_weight
        old = OLD_3X[parity_group - 1]
        new = NEW_3X[parity_group - 1]
        raw = (
            old_weight * old[0] + new_weight * new[0],
            old_weight * old[1] + new_weight * new[1],
            old_weight * old[2] + new_weight * new[2],
            old_weight * old[3] + new_weight * new[3],
        )
    else:
        raise ValueError(f"Unsupported use_3x value: {use_3x!r}")

    return (
        1.0 / (1.0 + raw[0]),
        1.0 / (1.0 + raw[1]),
        1.0 / (1.0 + raw[2]),
        1.0 / (1.0 + raw[3]),
    )


def _measurement_range(dim: int, trait: int, mrd: int) -> tuple[int, int]:
    begin = dim - mrd + 1
    end = dim
    if trait > 1:
        begin = dim - (mrd - 1) // 2
        end = begin
    return begin, end


def _validate_trait(trait: int) -> None:
    if not 1 <= trait <= 4:
        raise ValueError(f"Unsupported trait: {trait!r}")
