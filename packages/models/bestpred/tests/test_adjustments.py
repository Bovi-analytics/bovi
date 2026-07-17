from __future__ import annotations

import numpy as np
import pytest

from bestpred.core.adjustments import adjust_3x, expected_daily_yield


def test_expected_daily_yield_averages_milk_mrd_range() -> None:
    daily_yield = np.array(
        [
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    value = expected_daily_yield(
        trait=1, dim=3, mrd=2, daily_yield=daily_yield, herd_ratio=(2.0, 1.0, 1.0, 1.0)
    )

    assert value == pytest.approx(50.0)


def test_expected_daily_yield_uses_center_day_for_component_traits() -> None:
    daily_yield = np.array(
        [
            [10.0, 20.0, 30.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    value = expected_daily_yield(
        trait=2, dim=3, mrd=3, daily_yield=daily_yield, herd_ratio=(1.0, 3.0, 1.0, 1.0)
    )

    assert value == pytest.approx(6.0)


def test_adjust_3x_no_adjustment_mode_returns_identity_factors() -> None:
    cumulative_yield = np.vstack([np.arange(1, 366, dtype=np.float64) for _ in range(4)])

    result = adjust_3x(
        dims=(100,),
        length=305,
        fresh_year=2000,
        parity=1,
        milkings=(3,),
        cumulative_yield=cumulative_yield,
        use_3x=0,
        maxlen=365,
    )

    np.testing.assert_allclose(result.test_factors, np.ones((4, 1)))
    assert result.lactation_factors.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert result.partial_factors.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_adjust_3x_new_factors_match_single_3x_lactation_case() -> None:
    cumulative_yield = np.vstack([np.arange(1, 366, dtype=np.float64) for _ in range(4)])

    result = adjust_3x(
        dims=(100,),
        length=305,
        fresh_year=2000,
        parity=1,
        milkings=(3,),
        cumulative_yield=cumulative_yield,
        use_3x=2,
        maxlen=365,
    )

    expected = [1.0 / 1.12, 1.0 / 1.09, 1.0 / 1.10, 1.0]
    assert result.test_factors[:, 0].tolist() == pytest.approx(expected)
    assert result.lactation_factors.tolist() == pytest.approx(expected)
    assert result.partial_factors.tolist() == pytest.approx(expected)
