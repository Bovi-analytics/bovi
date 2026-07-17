from __future__ import annotations

import numpy as np
import pytest

from bestpred.core.covariance import (
    build_covariance_tables,
    lactation_covariance,
    observation_covariance,
)
from bestpred.core.curves import interpolate_curve
from bestpred.models import Trait


def _source11_daily_sd(*, parity_group: int, breed: int = 4, maxlen: int = 365) -> np.ndarray:
    methods = (("W", Trait.MILK), ("W", Trait.FAT), ("W", Trait.PROTEIN), ("G", Trait.SCS))
    return np.vstack(
        [
            interpolate_curve(
                trait=trait, parity_group=parity_group, breed=breed, method=method, maxlen=maxlen
            ).daily_sd
            for method, trait in methods
        ]
    )


def test_same_day_same_trait_covariance_is_variance() -> None:
    daily_sd = _source11_daily_sd(parity_group=2, maxlen=20)

    covariance = observation_covariance(
        dim1=10,
        trait1=1,
        supervision1=1,
        milkings1=2,
        samples1=2,
        mrd1=1,
        dim2=10,
        trait2=1,
        supervision2=1,
        milkings2=2,
        samples2=2,
        mrd2=1,
        daily_sd=daily_sd,
        parity_group=2,
        maxlen=20,
    )

    assert covariance == pytest.approx(daily_sd[0, 9] ** 2)


def test_same_day_ampm_increases_covariance_like_fortran() -> None:
    daily_sd = _source11_daily_sd(parity_group=2, maxlen=20)

    covariance = observation_covariance(
        dim1=10,
        trait1=1,
        supervision1=1,
        milkings1=2,
        samples1=1,
        mrd1=1,
        dim2=10,
        trait2=1,
        supervision2=1,
        milkings2=2,
        samples2=1,
        mrd2=1,
        daily_sd=daily_sd,
        parity_group=2,
        maxlen=20,
    )

    assert covariance == pytest.approx(1.3 * daily_sd[0, 9] ** 2)


def test_same_day_cross_trait_uses_phenotypic_correlation() -> None:
    daily_sd = _source11_daily_sd(parity_group=2, maxlen=20)

    covariance = observation_covariance(
        dim1=10,
        trait1=1,
        supervision1=1,
        milkings1=2,
        samples1=2,
        mrd1=1,
        dim2=10,
        trait2=2,
        supervision2=1,
        milkings2=2,
        samples2=2,
        mrd2=1,
        daily_sd=daily_sd,
        parity_group=2,
        maxlen=20,
    )

    assert covariance == pytest.approx(0.67 * daily_sd[0, 9] * daily_sd[1, 9])


def test_covariance_tables_match_direct_sum_for_lactation_yield() -> None:
    daily_sd = _source11_daily_sd(parity_group=2, maxlen=5)
    tables = build_covariance_tables(daily_sd=daily_sd, parity_group=2, maxlen=5)

    covariance = lactation_covariance(
        tables=tables,
        target_trait=1,
        observed_trait=1,
        dim=3,
        lactation_length=5,
        mrd=1,
    )
    direct = sum(
        observation_covariance(
            dim1=day,
            trait1=1,
            supervision1=1,
            milkings1=2,
            samples1=2,
            mrd1=1,
            dim2=3,
            trait2=1,
            supervision2=1,
            milkings2=2,
            samples2=2,
            mrd2=1,
            daily_sd=daily_sd,
            parity_group=2,
            maxlen=5,
        )
        for day in range(1, 6)
    )

    assert covariance == pytest.approx(direct)
