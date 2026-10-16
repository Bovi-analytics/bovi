"""Tests for the centering date method."""

import numpy as np
import pandas as pd
import pytest

from lactationcurve.characteristics.centering_date_method import (
    centering_date_method,
    centering_date_method_single_lactation,
)


def _scenario_data(
    dates: list[str],
    days_in_milk: list[int],
) -> pd.DataFrame:
    """Build a monthly test-day scenario."""
    return pd.DataFrame(
        {
            "DaysInMilk": days_in_milk,
            "MilkingYield": [5.0, 15.0, 25.0, 35.0, 30.0, 28.0, 25.0, 24.0, 20.0, 18.0],
            "TestId": [1] * 10,
            "TestCalendarDate": pd.to_datetime(dates),
        }
    )


@pytest.mark.parametrize(
    ("scenario", "expected_yield"),
    [
        (
            "march_without_february",
            6933.5,
        ),
        (
            "january_with_february",
            6900.5,
        ),
        (
            "march_partial_month_without_february",
            6803.5,
        ),
        (
            "january_partial_month_with_february",
            6770.5,
        ),
    ],
)
def test_monthly_test_day_scenarios(
    scenario: str,
    expected_yield: float,
) -> None:
    """Validate monthly testing with different start months and DIM offsets."""
    dates_by_scenario = {
        "march_without_february": [
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
            "2026-06-01",
            "2026-07-01",
            "2026-08-01",
            "2026-09-01",
            "2026-10-01",
            "2026-11-01",
            "2026-12-01",
        ],
        "january_with_february": [
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
            "2026-06-01",
            "2026-07-01",
            "2026-08-01",
            "2026-09-01",
            "2026-10-01",
        ],
        "march_partial_month_without_february": [
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
            "2026-06-01",
            "2026-07-01",
            "2026-08-01",
            "2026-09-01",
            "2026-10-01",
            "2026-11-01",
            "2026-12-01",
        ],
        "january_partial_month_with_february": [
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
            "2026-06-01",
            "2026-07-01",
            "2026-08-01",
            "2026-09-01",
            "2026-10-01",
        ],
    }
    days_by_scenario = {
        "march_without_february": [10, 41, 71, 102, 132, 163, 194, 224, 255, 285],
        "january_with_february": [10, 41, 71, 102, 132, 163, 194, 224, 255, 285],
        "march_partial_month_without_february": [20, 51, 81, 112, 142, 173, 204, 234, 265, 295],
        "january_partial_month_with_february": [20, 51, 81, 112, 142, 173, 204, 234, 265, 295],
    }

    result = centering_date_method(
        _scenario_data(dates_by_scenario[scenario], days_by_scenario[scenario])
    )

    assert result.loc[0, "TestId"] == 1
    assert result.loc[0, "LactationMilkYield"] == pytest.approx(expected_yield)


def test_single_lactation_uses_centered_month_weights() -> None:
    """Weight the first and last test months by their centered intervals."""
    df = pd.DataFrame(
        {
            "TestCalendarDate": pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10"]),
            "DaysInMilk": [10, 41, 69],
            "MilkingYield": [20.0, 30.0, 40.0],
        }
    )

    result = centering_date_method_single_lactation(df, lactation_length=100)

    assert result == pytest.approx(3240.0)


def test_single_lactation_interpolates_missing_months() -> None:
    """Estimate a missing month from the nearest recorded months."""
    df = pd.DataFrame(
        {
            "TestCalendarDate": pd.to_datetime(["2024-01-10", "2024-03-10"]),
            "DaysInMilk": [10, 69],
            "MilkingYield": [20.0, 40.0],
        }
    )

    result = centering_date_method_single_lactation(df, lactation_length=100)

    assert result == pytest.approx(3240.0)


def test_single_lactation_accounts_for_leap_year() -> None:
    """Use 29 days for February when the test dates are in a leap year."""
    df = pd.DataFrame(
        {
            "TestCalendarDate": pd.to_datetime(["2024-01-10", "2024-03-10"]),
            "DaysInMilk": [10, 70],
            "MilkingYield": [20.0, 40.0],
        }
    )

    result = centering_date_method_single_lactation(df, lactation_length=100)

    assert result == pytest.approx(3200.0)


def test_centering_date_method_groups_lactations_and_filters_dim() -> None:
    """Process each TestId independently and exclude records beyond the horizon."""
    df = pd.DataFrame(
        {
            "TestCalendarDate": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-02-10",
                    "2024-01-10",
                    "2024-02-10",
                    "2024-04-10",
                ]
            ),
            "DaysInMilk": [10, 41, 10, 41, 120],
            "MilkingYield": [20.0, 30.0, 10.0, 10.0, 999.0],
            "TestId": [1, 1, 2, 2, 2],
        }
    )

    result = centering_date_method(df, lactation_length=100)

    result = result.sort_values("TestId").reset_index(drop=True)
    assert result["TestId"].tolist() == [1, 2]
    assert np.allclose(result["LactationMilkYield"], [2715.0, 990.0])


def test_centering_date_method_supports_aliases_and_default_test_id() -> None:
    """Resolve common aliases and create the requested default identifier."""
    df = pd.DataFrame(
        {
            "DIM": [10, 41],
            "Yield": [20.0, 30.0],
            "TestCalendarDate": ["2024-01-10", "2024-02-10"],
        }
    )

    result = centering_date_method(df, default_test_id=7, lactation_length=100)

    assert list(result.columns) == ["TestId", "LactationMilkYield"]
    assert result["TestId"].tolist() == [7]
    assert result["LactationMilkYield"].iloc[0] == pytest.approx(2715.0)


def test_centering_date_method_rejects_multiple_tests_in_one_month() -> None:
    """The centering date method requires at most one test per month."""
    df = pd.DataFrame(
        {
            "TestCalendarDate": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            "DaysInMilk": [10, 20],
            "MilkingYield": [20.0, 30.0],
            "TestId": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="multiple measurements"):
        centering_date_method(df)
