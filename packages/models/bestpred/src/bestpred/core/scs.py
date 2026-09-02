"""Somatic cell score age, stage, and month adjustments from `ageadjs.c`."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bestpred.core.age import perpetual_day
from bestpred.models import Format4Record

FloatArray = npt.NDArray[np.float64]

DEFAULT_ADJUST_SCS: Final[Path | None] = None
DATA_PACKAGE: Final[str] = "bestpred.data"
DEFAULT_SCS_305: Final = 329
DEFAULT_I305: Final = 305


@dataclass(frozen=True)
class ScsAdjustmentData:
    """Parsed adjustment tables from `adjust.scs`."""

    dim: FloatArray
    age: FloatArray
    month: FloatArray


def adjusted_scs(
    *,
    breed: str,
    parity: int,
    fresh_month: int,
    state: int,
    age_at_freshening_months: int,
    dim: int,
    scs: int,
    data: ScsAdjustmentData | None = None,
) -> int:
    """Pure Python port of the active `ageadjs.c::adjscs` routine.

    Inputs and outputs are scaled as `100 * SCS`, matching the C and Fortran
    boundary. The active Linux build links `ageadjs.c`, which uses the additive
    SCS adjustment formula and loads factors from `adjust.scs`.
    """

    tables = load_scs_adjustment_data() if data is None else data
    lactation_index = 0 if parity == 1 else 1
    breed_index = 1 if breed.strip().upper()[:1] in {"J", "G"} else 0
    age = min(max(age_at_freshening_months, 18), 120)
    clamped_dim = min(max(dim, 15), 305)
    region_index = _scs_region_index(state)
    adjusted = round(
        scs
        - tables.dim[clamped_dim, lactation_index, breed_index]
        - tables.month[fresh_month, region_index, breed_index]
        + (tables.age[age, breed_index] - 1.0) * 300.0
    )
    return min(max(int(adjusted), 1), 999)


def scs_age_factor(
    *,
    breed: str,
    parity: int,
    fresh_month: int,
    state: int,
    age_at_freshening_months: int,
    dim: int = DEFAULT_I305,
    scs305: int = DEFAULT_SCS_305,
    data: ScsAdjustmentData | None = None,
) -> float:
    """Return the SCS factor assigned to `agefac(4)` in `bestpred_fmt4.f90`."""

    return adjusted_scs(
        breed=breed,
        parity=parity,
        fresh_month=fresh_month,
        state=state,
        age_at_freshening_months=age_at_freshening_months,
        dim=dim,
        scs=scs305,
        data=data,
    ) / float(scs305)


def format4_scs_age_factor(
    record: Format4Record,
    *,
    dim: int = DEFAULT_I305,
    scs305: int = DEFAULT_SCS_305,
) -> float:
    """Compute the SCS age factor for a prepared Format 4 record."""

    fresh_month = int(record.fresh_date[4:6])
    state = int(record.herd_id[:2])
    age = int((perpetual_day(record.fresh_date) - perpetual_day(record.birth_date)) / 30.5)
    return scs_age_factor(
        breed=record.cow_id[:1],
        parity=record.parity,
        fresh_month=fresh_month,
        state=state,
        age_at_freshening_months=age,
        dim=dim,
        scs305=scs305,
    )


@lru_cache(maxsize=1)
def load_scs_adjustment_data(path: Path | None = DEFAULT_ADJUST_SCS) -> ScsAdjustmentData:
    """Load SCS DIM, age, and month factors in the same order as `ageadjs.c`."""

    source = (
        resources.files(DATA_PACKAGE).joinpath("adjust.scs").read_text(encoding="utf-8")
        if path is None
        else path.read_text(encoding="utf-8")
    )
    values = [float(token) for token in source.split()]
    dim_values = 291 * 2 * 2
    age_values = 103 * 2
    month_values = 4 * 12 * 2
    expected_minimum = dim_values + age_values + month_values
    if len(values) < expected_minimum:
        raise ValueError(
            f"adjust.scs has {len(values)} numeric values, expected at least {expected_minimum}"
        )

    offset = 0
    dim = np.zeros((306, 2, 2), dtype=np.float64)
    for lactation_index in range(2):
        for day in range(15, 306):
            dim[day, lactation_index, 0] = values[offset]
            dim[day, lactation_index, 1] = values[offset + 1]
            offset += 2

    age = np.zeros((121, 2), dtype=np.float64)
    for age_months in range(18, 121):
        age[age_months, 0] = values[offset]
        age[age_months, 1] = values[offset + 1]
        offset += 2

    month = np.zeros((13, 4, 2), dtype=np.float64)
    for region_index in range(4):
        for fresh_month in range(1, 13):
            month[fresh_month, region_index, 0] = values[offset]
            month[fresh_month, region_index, 1] = values[offset + 1]
            offset += 2

    return ScsAdjustmentData(dim=dim, age=age, month=month)


def _scs_region_index(state: int) -> int:
    if state <= 23:
        return 0
    if state <= 48:
        return 1
    if state <= 74:
        return 2
    return 3
