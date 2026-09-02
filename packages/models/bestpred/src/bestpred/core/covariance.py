"""Fortran-compatible covariance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DCR_BY_SUPERVISION: Final[tuple[float, ...]] = (
    0.0,
    1.0,
    0.77,
    0.97,
    1.0,
    1.0,
    0.77,
    0.97,
    1.0,
    1.0,
)
PHENOTYPIC_CORRELATION: Final[FloatArray] = np.array(
    [
        [1.0, 0.67, 0.85, -0.08],
        [0.67, 1.0, 0.77, -0.14],
        [0.85, 0.77, 1.0, -0.10],
        [-0.08, -0.14, -0.10, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CovarianceTables:
    """Cached `covari` and `covd` tables for one parity group."""

    covariance_to_yield: FloatArray
    covariance_to_persistency: FloatArray
    parity_group: int
    maxlen: int
    precise: int


def observation_covariance(
    *,
    dim1: int,
    trait1: int,
    supervision1: int,
    milkings1: int,
    samples1: int,
    mrd1: int,
    dim2: int,
    trait2: int,
    supervision2: int,
    milkings2: int,
    samples2: int,
    mrd2: int,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
) -> float:
    """Port of Fortran `vary()`.

    `daily_sd` is shaped `(4, maxlen)` and uses zero-based Python indexes.
    Public scalar arguments keep the Fortran conventions: traits and DIM start
    at 1.
    """

    _validate_trait(trait1)
    _validate_trait(trait2)
    _validate_parity_group(parity_group)

    ibegin, iend = _measurement_range(dim1, trait1, mrd1)
    jbegin, jend = _measurement_range(dim2, trait2, mrd2)
    covariance = 0.0

    for day_i in range(ibegin, iend + 1):
        for day_j in range(jbegin, jend + 1):
            dim_difference = abs(day_i - day_j)
            idiag = 1.0 if day_i == day_j else 0.0
            corr = _base_correlation(trait1, trait2, parity_group, dim_difference, idiag)

            if trait1 == trait2 and day_i == day_j:
                corr = 1.0

            if day_i == day_j:
                corr += (
                    0.3
                    * _trait_correlation(trait1, trait2)
                    * (float(milkings1) / max(samples1, samples2) - 1.0)
                )
                if trait1 == trait2 and _dcr(supervision1) == 0.0:
                    corr = 1_000_000.0

            err1 = _owner_sampler_error(supervision1)
            err2 = _owner_sampler_error(supervision2)
            corr += sqrt(err1 * err2) * _trait_correlation(trait1, trait2)

            covariance += (
                corr
                * _sd_at(daily_sd, trait1, day_i, maxlen)
                * _sd_at(daily_sd, trait2, day_j, maxlen)
            )

    return covariance / (iend - ibegin + 1) / (jend - jbegin + 1)


def build_covariance_tables(
    *,
    daily_sd: FloatArray,
    parity_group: int,
    maxlen: int,
    precise: int = 1,
) -> CovarianceTables:
    """Build the cached tables populated by Fortran `covary(..., trait=0)`."""

    if precise < 1:
        raise ValueError("precise must be at least 1")

    covariance_to_yield = np.zeros((4, 4, maxlen, maxlen), dtype=np.float64)
    covariance_to_persistency = np.zeros((4, 4, maxlen, maxlen), dtype=np.float64)
    iplus = (precise - 1) / 2.0

    for target_trait in range(1, 5):
        for observed_trait in range(1, 5):
            for observed_dim in range(1, maxlen + 1):
                cumulative = 0.0
                cumulative_dim = 0.0
                for lactation_dim in range(1, maxlen + 1, precise):
                    cov = observation_covariance(
                        dim1=lactation_dim,
                        trait1=target_trait,
                        supervision1=1,
                        milkings1=2,
                        samples1=2,
                        mrd1=1,
                        dim2=observed_dim,
                        trait2=observed_trait,
                        supervision2=1,
                        milkings2=2,
                        samples2=2,
                        mrd2=1,
                        daily_sd=daily_sd,
                        parity_group=parity_group,
                        maxlen=maxlen,
                    )
                    step = min(precise, maxlen + 1 - lactation_dim)
                    cumulative += cov * step
                    cumulative_dim += cov * (lactation_dim + iplus) * step
                    covariance_to_yield[
                        target_trait - 1, observed_trait - 1, observed_dim - 1, lactation_dim - 1
                    ] = cumulative
                    covariance_to_persistency[
                        target_trait - 1, observed_trait - 1, observed_dim - 1, lactation_dim - 1
                    ] = cumulative_dim

    return CovarianceTables(
        covariance_to_yield=covariance_to_yield,
        covariance_to_persistency=covariance_to_persistency,
        parity_group=parity_group,
        maxlen=maxlen,
        precise=precise,
    )


def lactation_covariance(
    *,
    tables: CovarianceTables,
    target_trait: int,
    observed_trait: int,
    dim: int,
    lactation_length: int,
    mrd: int,
    statistic: int = 1,
) -> float:
    """Lookup path of Fortran `covary()` for yield or persistency covariance."""

    _validate_trait(target_trait)
    _validate_trait(observed_trait)
    if not 1 <= lactation_length <= tables.maxlen:
        raise ValueError(f"lactation_length must be between 1 and {tables.maxlen}")

    mbegin, mend = _measurement_range(dim, observed_trait, mrd)
    table = tables.covariance_to_yield if statistic == 1 else tables.covariance_to_persistency
    covariance = 0.0

    for measurement_dim in range(mbegin, mend + 1):
        covariance += table[
            target_trait - 1, observed_trait - 1, measurement_dim - 1, lactation_length - 1
        ]

    return covariance / (mend - mbegin + 1)


def _base_correlation(
    trait1: int, trait2: int, parity_group: int, dim_difference: int, idiag: float
) -> float:
    trait_corr = _trait_correlation(trait1, trait2)
    if parity_group == 1:
        if trait1 < 4 and trait2 < 4:
            return (0.214 * idiag + 0.786 * 0.998**dim_difference) * trait_corr
        if trait1 == 4 and trait2 == 4:
            return (0.199 * idiag + 0.801 * 0.998**dim_difference) * trait_corr
        sqc1 = sqrt(0.214 * idiag + 0.786 * 0.998**dim_difference)
        sqc2 = sqrt(0.199 * idiag + 0.801 * 0.998**dim_difference)
        return sqc1 * sqc2 * trait_corr

    if trait1 < 4 and trait2 < 4:
        return (0.132 * idiag + 0.868 * 0.997**dim_difference) * trait_corr
    if trait1 == 4 and trait2 == 4:
        return (0.199 * idiag + 0.801 * 0.998**dim_difference) * trait_corr
    sqc1 = sqrt(0.132 * idiag + 0.868 * 0.997**dim_difference)
    sqc2 = sqrt(0.199 * idiag + 0.801 * 0.998**dim_difference)
    return sqc1 * sqc2 * trait_corr


def _measurement_range(dim: int, trait: int, mrd: int) -> tuple[int, int]:
    begin = dim - mrd + 1
    end = dim
    if trait > 1:
        begin = dim - (mrd - 1) // 2
        end = begin
    return begin, end


def _sd_at(daily_sd: FloatArray, trait: int, dim: int, maxlen: int) -> float:
    if not 1 <= dim <= maxlen:
        raise ValueError(f"DIM {dim} is outside 1..{maxlen}")
    return float(daily_sd[trait - 1, dim - 1])


def _trait_correlation(trait1: int, trait2: int) -> float:
    return float(PHENOTYPIC_CORRELATION[trait1 - 1, trait2 - 1])


def _dcr(supervision: int) -> float:
    if 0 <= supervision <= 9:
        return DCR_BY_SUPERVISION[supervision]
    raise ValueError(f"Unsupported supervision code: {supervision!r}")


def _owner_sampler_error(supervision: int) -> float:
    dcr = _dcr(supervision)
    if dcr <= 0.0:
        return 0.0
    return (1.0 / dcr - 1.0) * 0.62


def _validate_trait(trait: int) -> None:
    if not 1 <= trait <= 4:
        raise ValueError(f"Unsupported trait: {trait!r}")


def _validate_parity_group(parity_group: int) -> None:
    if parity_group not in {1, 2}:
        raise ValueError(f"Unsupported parity group: {parity_group!r}")
