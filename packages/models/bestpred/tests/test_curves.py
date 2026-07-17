from __future__ import annotations

from math import exp

import pytest

from bestpred.core.curves import interpolate_curve, load_regional_curve_tables
from bestpred.models import Trait


def test_linear_curve_matches_fortran_monthly_interpolation() -> None:
    curve = interpolate_curve(trait=Trait.MILK, parity_group=1, breed=4, method="L", maxlen=366)

    assert curve.method_used == "L"
    assert curve.daily_yield[0] == pytest.approx(48.773333333333326)
    assert curve.daily_yield[14] == pytest.approx(53.3)
    assert curve.daily_yield[44] == pytest.approx(63.0)
    assert curve.daily_yield[365] == pytest.approx(52.1)
    assert curve.daily_sd[14] == pytest.approx(10.8)
    assert curve.cumulative_yield[1] == pytest.approx(curve.daily_yield[0] + curve.daily_yield[1])


def test_wood_curve_matches_fortran_holstein_later_milk_formula() -> None:
    curve = interpolate_curve(trait=Trait.MILK, parity_group=2, breed=4, method="W", maxlen=3)

    expected_day_1 = 22.0087 * 1.0**0.2155 * exp(-1.0 * 0.00357)
    expected_day_3 = 22.0087 * 3.0**0.2155 * exp(-3.0 * 0.00357)
    expected_sd_day_1 = 8.7545 * 1.0**0.0282 * exp(-1.0 * 0.000439)

    assert curve.method_used == "W"
    assert curve.breed_used == 4
    assert curve.daily_yield[0] == pytest.approx(expected_day_1)
    assert curve.daily_yield[2] == pytest.approx(expected_day_3)
    assert curve.daily_sd[0] == pytest.approx(expected_sd_day_1)
    assert curve.mean_persistency_numerator == pytest.approx(
        curve.daily_yield[0] + curve.daily_yield[1] * 2.0 + curve.daily_yield[2] * 3.0
    )


def test_morant_gnanasakthy_curve_matches_fortran_holstein_later_scs_formula() -> None:
    curve = interpolate_curve(trait=Trait.SCS, parity_group=2, breed=4, method="G", maxlen=1)

    shifted_dim = 11.0
    expected_mean = (
        2.5072 - (-0.00431 * shifted_dim) + (-4.59e-06 * shifted_dim**2) + 8.9804 / shifted_dim
    )
    expected_sd = (
        2.4849
        - (0.00229 * shifted_dim)
        + (3.454e-06 * shifted_dim**2) / 2.0
        + (-6.3911 / shifted_dim)
    )

    assert curve.method_used == "G"
    assert curve.daily_yield[0] == pytest.approx(expected_mean)
    assert curve.daily_sd[0] == pytest.approx(expected_sd)


def test_invalid_breed_defaults_to_holstein_like_fortran() -> None:
    holstein = interpolate_curve(trait=Trait.PROTEIN, parity_group=1, breed=4, method="W", maxlen=5)
    invalid = interpolate_curve(
        trait=Trait.PROTEIN, parity_group=1, breed=999, method="W", maxlen=5
    )

    assert invalid.breed_used == 4
    assert invalid.daily_yield.tolist() == pytest.approx(holstein.daily_yield.tolist())
    assert invalid.daily_sd.tolist() == pytest.approx(holstein.daily_sd.tolist())


def test_invalid_method_falls_back_to_linear_like_fortran() -> None:
    curve = interpolate_curve(trait=Trait.FAT, parity_group=1, breed=4, method="X", maxlen=1)

    assert curve.method_used == "L"
    assert curve.daily_yield[0] == pytest.approx(1.8073333333333335)


def test_regional_wood_curve_matches_fortran_mideast_spring_milk_formula() -> None:
    curve = interpolate_curve(
        trait=Trait.MILK,
        parity_group=1,
        breed=4,
        method="R",
        maxlen=1,
        region=1,
    )

    assert curve.method_used == "R"
    assert curve.daily_yield[0] == pytest.approx(14.3926 * 1.0**0.2274 * exp(-0.00235))
    assert curve.daily_sd[0] == pytest.approx(5.9724 * 1.0**0.0232 * exp(0.00027))


def test_calving_wood_curve_matches_fortran_spring_milk_formula() -> None:
    curve = interpolate_curve(
        trait=Trait.MILK,
        parity_group=1,
        breed=4,
        method="C",
        maxlen=1,
        season=1,
    )

    assert curve.method_used == "C"
    assert curve.daily_yield[0] == pytest.approx(15.786 * 1.0**0.20756 * exp(-0.002180946))
    assert curve.daily_sd[0] == pytest.approx(5.8618 * 1.0**0.03394 * exp(0.00025351))


def test_seasonal_wood_curve_matches_fortran_mideast_spring_milk_formula() -> None:
    curve = interpolate_curve(
        trait=Trait.MILK,
        parity_group=1,
        breed=4,
        method="T",
        maxlen=1,
        region=1,
        season=1,
    )

    assert curve.method_used == "T"
    assert curve.daily_yield[0] == pytest.approx(17.0437 * 1.0**0.18043 * exp(-0.001958839))
    assert curve.daily_sd[0] == pytest.approx(6.5683 * 1.0 ** (-0.00245) * exp(0.000428644))


def test_regional_mandg_curve_matches_fortran_mideast_scs_formula() -> None:
    curve = interpolate_curve(
        trait=Trait.SCS,
        parity_group=1,
        breed=4,
        method="S",
        maxlen=1,
        region=1,
    )

    shifted_dim = 11.0
    expected_mean = (
        2.4457 - (-0.00058 * shifted_dim) + 7.297e-6 * shifted_dim**2 + (14.3740 / shifted_dim)
    )
    expected_sd = (
        1.8239
        - (-0.00018 * shifted_dim)
        + (-8.96e-7 * shifted_dim**2) / 2.0
        + (0.5750 / shifted_dim)
    )

    assert curve.method_used == "S"
    assert curve.daily_yield[0] == pytest.approx(expected_mean)
    assert curve.daily_sd[0] == pytest.approx(expected_sd)


def test_calving_mandg_curve_matches_fortran_spring_scs_formula() -> None:
    curve = interpolate_curve(
        trait=Trait.SCS,
        parity_group=1,
        breed=4,
        method="D",
        maxlen=1,
        season=1,
    )

    shifted_dim = 11.0
    expected_mean = (
        2.4468 - (0.00232 * shifted_dim) + 0.000002592 * shifted_dim**2 + (-4.7834 / shifted_dim)
    )
    expected_sd = (
        1.8669
        - (0.0003 * shifted_dim)
        + (0.000000128 * shifted_dim**2) / 2.0
        + (-0.486 / shifted_dim)
    )

    assert curve.method_used == "D"
    assert curve.daily_yield[0] == pytest.approx(expected_mean)
    assert curve.daily_sd[0] == pytest.approx(expected_sd)


def test_seasonal_mandg_curve_uses_fortran_table_shape() -> None:
    tables = load_regional_curve_tables()
    curve = interpolate_curve(
        trait=Trait.SCS,
        parity_group=1,
        breed=4,
        method="U",
        maxlen=1,
        region=1,
        season=1,
    )

    shifted_dim = 11.0
    mean_params = tables["seasonal_mandg_means"][:, 0, 0, 0, 0]
    sd_params = tables["seasonal_mandg_sd"][:, 0, 0, 0, 0]

    assert curve.method_used == "U"
    assert curve.daily_yield[0] == pytest.approx(
        mean_params[0]
        - mean_params[1] * shifted_dim
        + mean_params[2] * shifted_dim**2
        + mean_params[3] / shifted_dim
    )
    assert curve.daily_sd[0] == pytest.approx(
        sd_params[0]
        - sd_params[1] * shifted_dim
        + (sd_params[2] * shifted_dim**2) / 2.0
        + sd_params[3] / shifted_dim
    )
