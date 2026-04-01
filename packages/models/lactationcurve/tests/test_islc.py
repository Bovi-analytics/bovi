"""
Test suite for ICAR ISLC interpolation and 305-day yield estimation.

This module provides comprehensive tests for the ISLC implementation,
including helper utilities, interpolation methods, 305-day yield
calculation, and standard-curve representation generation.

Test Categories:
    - Curve helpers: Conversion and lookup helper behavior
    - ISLC method core: Validation and branch coverage for ``ISLC_method``
    - Interpolation methods: ``interpolation_standard_lc`` and linear helpers
    - ISLC orchestration: Grouping, filtering, and per-lactation output
    - ISLC ICAR orchestration: Grouping, filtering, and per-lactation output
    - Representation builder: Validation and output-shape checks for
      ``create_standard_lc_representation``

Usage:
    Run all tests::

        pytest tests/lactationcurve/test_islc.py -v

    Run specific marker::

        pytest tests/lactationcurve/test_islc.py -m helpers -v
        pytest tests/lactationcurve/test_islc.py -m islc_method -v
        pytest tests/lactationcurve/test_islc.py -m interpolation -v
        pytest tests/lactationcurve/test_islc.py -m islc -v
        pytest tests/lactationcurve/test_islc.py -m islc_icar -v
        pytest tests/lactationcurve/test_islc.py -m representation -v

Author:
    Meike van Leerdam
"""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

islc_mod = import_module("lactationcurve.characteristics.ISLC")


def _standard_curve_series() -> pd.Series:
    """Create a deterministic standard curve indexed by DIM 0..305."""
    return pd.Series(np.linspace(40.0, 20.0, 306), index=range(306))


def _identity_corr_and_std() -> tuple[pd.DataFrame, np.ndarray]:
    """Create neutral correlation and std inputs for ISLC tests."""
    n = len(islc_mod.GRID_DAYS)
    corr = pd.DataFrame(np.eye(n), index=range(n), columns=range(n))
    std = np.ones(n, dtype=float)
    return corr, std


def _identity_corr_and_std_icar() -> tuple[pd.DataFrame, np.ndarray]:
    """Create neutral correlation and std inputs for ISLC_ICAR tests."""
    n = len(islc_mod.GRID_DAYS) + 2  # includes day 0 and day 305
    corr = pd.DataFrame(np.eye(n), index=range(n), columns=range(n))
    std = np.ones(n, dtype=float)
    return corr, std


@pytest.mark.helpers
def test_curve_to_series_passthrough_for_series() -> None:
    """_curve_to_series returns a Series unchanged when input is already a Series."""
    curve = _standard_curve_series()
    result = islc_mod._curve_to_series(curve)
    assert isinstance(result, pd.Series)
    assert result.equals(curve)


@pytest.mark.helpers
def test_curve_to_series_converts_numpy_array() -> None:
    """_curve_to_series converts ndarray to Series with positional index."""
    arr = np.array([1.0, 2.0, 3.0])
    result = islc_mod._curve_to_series(arr)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 1.0
    assert result.iloc[2] == 3.0


@pytest.mark.helpers
def test_curve_value_prefers_label_then_fallback_positional() -> None:
    """_curve_value supports both labeled and positional lookup paths."""
    labeled = pd.Series([10.0, 20.0, 30.0], index=[10, 20, 30])
    positional = pd.Series([11.0, 22.0, 33.0], index=[100, 200, 300])

    assert islc_mod._curve_value(labeled, 20) == 20.0
    assert islc_mod._curve_value(positional, 1) == 22.0


@pytest.mark.islc_method
def test_islc_method_raises_for_non_grid_day() -> None:
    """ISLC_method fails fast when input has DIM values outside GRID_DAYS."""
    corr, std = _identity_corr_and_std()
    lactation = pd.DataFrame({"GridDay": [11, 30], "MilkYieldInterp": [30.0, 28.0]})

    with pytest.raises(ValueError, match="not part of the grid"):
        islc_mod.ISLC_method(lactation, _standard_curve_series(), corr, std)


@pytest.mark.islc_method
def test_islc_method_returns_float_when_day_290_present() -> None:
    """ISLC_method produces finite float when day 290 branch is used."""
    corr, std = _identity_corr_and_std()
    lactation = pd.DataFrame(
        {
            "GridDay": [10, 30, 50, 290],
            "MilkYieldInterp": [30.0, 29.0, 28.0, 20.0],
        }
    )

    value = islc_mod.ISLC_method(lactation, _standard_curve_series(), corr, std)
    assert isinstance(value, float)
    assert np.isfinite(value)


@pytest.mark.interpolation
def test_interpolation_standard_lc_raises_without_standard_curve() -> None:
    """Interpolation requires a standard lactation curve input."""
    group = pd.DataFrame({"DaysInMilk": [10], "MilkingYield": [30.0], "TestId": [1]})
    with pytest.raises(ValueError, match="standard lactation curve"):
        islc_mod.interpolation_standard_lc(group, "DaysInMilk", "MilkingYield", None)


@pytest.mark.interpolation
def test_interpolation_standard_lc_preserves_testid() -> None:
    """Interpolation output keeps TestId for exact and interpolated grid days."""
    group = pd.DataFrame(
        {
            "DaysInMilk": [10, 50],
            "MilkingYield": [30.0, 20.0],
            "TestId": [42, 42],
        }
    )

    result = islc_mod.interpolation_standard_lc(
        group,
        column_name_dim="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc=_standard_curve_series(),
    )

    assert not result.empty
    assert "TestId" in result.columns
    assert set(result["TestId"].tolist()) == {42}
    assert 30 in result["GridDay"].values


@pytest.mark.interpolation
def test_interpolation_standard_lc_returns_empty_with_expected_columns() -> None:
    """Interpolation returns an empty frame with expected schema when out of grid range."""
    group = pd.DataFrame(
        {"DaysInMilk": [1, 2, 3], "MilkingYield": [30.0, 29.0, 28.0], "TestId": [9, 9, 9]}
    )

    result = islc_mod.interpolation_standard_lc(
        group,
        column_name_dim="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc=_standard_curve_series(),
    )

    assert result.empty
    assert set(["TestId", "GridDay", "MilkYieldInterp"]).issubset(result.columns)


@pytest.mark.interpolation
def test_linear_interpd_all_to_grid_returns_all_grid_days() -> None:
    """Linear all-to-grid interpolation returns one row per grid day."""
    group = pd.DataFrame({"DaysInMilk": [10, 50], "MilkingYield": [30.0, 20.0], "TestId": [4, 4]})

    result = islc_mod.linear_interpd_all_to_grid(group, "DaysInMilk", "MilkingYield")

    assert len(result) == len(islc_mod.GRID_DAYS)
    assert set(result["GridDay"].tolist()) == set(islc_mod.GRID_DAYS)
    assert set(result["TestId"].tolist()) == {4}


@pytest.mark.interpolation
def test_linear_interpd_closest_to_grid_returns_empty_when_no_grid_day_in_range() -> None:
    """Closest-to-grid interpolation returns None when no grid day is in range."""
    group = pd.DataFrame(
        {"DaysInMilk": [1, 2, 3], "MilkingYield": [25.0, 26.0, 27.0], "TestId": [1, 1, 1]}
    )

    result = islc_mod.linear_interpd_closest_to_grid(group, "DaysInMilk", "MilkingYield")
    assert result is None


@pytest.mark.interpolation
def test_linear_interpd_closest_to_grid_returns_subset_of_grid_days() -> None:
    """Closest-to-grid interpolation returns grid days bounded by input DIM range."""
    group = pd.DataFrame({"DaysInMilk": [45, 115], "MilkingYield": [30.0, 24.0], "TestId": [2, 2]})

    result = islc_mod.linear_interpd_closest_to_grid(group, "DaysInMilk", "MilkingYield")

    assert isinstance(result, pd.DataFrame)
    assert result is not None
    assert result["GridDay"].min() >= 45
    assert result["GridDay"].max() <= 115
    assert set(result["TestId"].tolist()) == {2}


@pytest.mark.islc
def test_islc_adds_default_testid_when_missing() -> None:
    """ISLC treats data without TestId as a single lactation with TestId==0."""
    df = pd.DataFrame({"DaysInMilk": [10, 30, 50], "MilkingYield": [32.0, 31.0, 30.0]})
    corr, std = _identity_corr_and_std()

    result = islc_mod.ISLC(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert list(result.columns) == ["305_milk_yield", "TestId"]
    assert len(result) == 1
    assert int(result.loc[0, "TestId"]) == 0
    assert np.isfinite(float(result.loc[0, "305_milk_yield"]))


@pytest.mark.islc
def test_islc_processes_multiple_lactations() -> None:
    """ISLC returns one row per unique TestId for valid multi-lactation input."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [10, 30, 50, 10, 30, 50],
            "MilkingYield": [32.0, 31.0, 30.0, 26.0, 25.0, 24.0],
            "TestId": [1, 1, 1, 2, 2, 2],
        }
    )
    corr, std = _identity_corr_and_std()

    result = islc_mod.ISLC(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert len(result) == 2
    assert set(result["TestId"].astype(int).tolist()) == {1, 2}


@pytest.mark.islc
def test_islc_filters_days_over_305() -> None:
    """ISLC filters rows with DIM > 305 before interpolation/calculation."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [10, 30, 50, 400],
            "MilkingYield": [32.0, 31.0, 30.0, 99.0],
            "TestId": [1, 1, 1, 1],
        }
    )
    corr, std = _identity_corr_and_std()

    result = islc_mod.ISLC(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert len(result) == 1
    assert int(result.loc[0, "TestId"]) == 1
    assert np.isfinite(float(result.loc[0, "305_milk_yield"]))


@pytest.mark.islc
def test_islc_returns_nan_for_non_interpolable_lactation() -> None:
    """ISLC returns NaN yield instead of crashing when interpolation has no rows."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [1, 2, 3],
            "MilkingYield": [25.0, 26.0, 27.0],
            "TestId": [7, 7, 7],
        }
    )
    corr, std = _identity_corr_and_std()

    result = islc_mod.ISLC(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert len(result) == 1
    assert int(result.loc[0, "TestId"]) == 7
    assert np.isnan(float(result.loc[0, "305_milk_yield"]))


@pytest.mark.islc
def test_islc_icar_adds_default_testid_when_missing() -> None:
    """ISLC_ICAR treats data without TestId as a single lactation with TestId==0."""
    df = pd.DataFrame({"DaysInMilk": [10, 30, 50], "MilkingYield": [32.0, 31.0, 30.0]})
    corr, std = _identity_corr_and_std_icar()

    result = islc_mod.ISLC_ICAR(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert list(result.columns) == ["lactation_milk_yield", "TestId"]
    assert len(result) == 1
    assert int(result.loc[0, "TestId"]) == 0
    assert np.isfinite(float(result.loc[0, "lactation_milk_yield"]))


@pytest.mark.islc
def test_islc_icar_processes_multiple_lactations() -> None:
    """ISLC_ICAR returns one row per unique TestId for multi-lactation input."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [10, 30, 50, 10, 30, 50],
            "MilkingYield": [32.0, 31.0, 30.0, 26.0, 25.0, 24.0],
            "TestId": [1, 1, 1, 2, 2, 2],
        }
    )
    corr, std = _identity_corr_and_std_icar()

    result = islc_mod.ISLC_ICAR(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert len(result) == 2
    assert set(result["TestId"].astype(int).tolist()) == {1, 2}


@pytest.mark.islc
def test_islc_icar_filters_days_over_max_dim_by_default() -> None:
    """ISLC_ICAR defaults to max_dim=305 and drops records above that threshold."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [10, 30, 50, 400],
            "MilkingYield": [32.0, 31.0, 30.0, 99.0],
            "TestId": [1, 1, 1, 1],
        }
    )
    corr, std = _identity_corr_and_std_icar()

    result = islc_mod.ISLC_ICAR(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
    )

    assert len(result) == 1
    assert int(result.loc[0, "TestId"]) == 1
    assert np.isfinite(float(result.loc[0, "lactation_milk_yield"]))


@pytest.mark.islc
def test_islc_icar_allows_extended_horizon_with_max_dim_max() -> None:
    """ISLC_ICAR can include DIM > 305 when max_dim='max'."""
    df = pd.DataFrame(
        {
            "DaysInMilk": [10, 30, 50, 350],
            "MilkingYield": [32.0, 31.0, 30.0, 29.0],
            "TestId": [1, 1, 1, 1],
        }
    )
    corr, std = _identity_corr_and_std_icar()

    result_305 = islc_mod.ISLC_ICAR(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
        max_dim=305,
    )
    result_max = islc_mod.ISLC_ICAR(
        df=df,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
        standard_lc_305=_standard_curve_series(),
        correlation_matrix=corr,
        std_per_grid_day=std,
        max_dim="max",
    )

    assert np.isfinite(float(result_305.loc[0, "lactation_milk_yield"]))
    assert np.isfinite(float(result_max.loc[0, "lactation_milk_yield"]))
    assert float(result_max.loc[0, "lactation_milk_yield"]) >= float(
        result_305.loc[0, "lactation_milk_yield"]
    )


@pytest.mark.representation
def test_create_standard_lc_representation_requires_testid() -> None:
    """Representation builder requires TestId column."""
    df = pd.DataFrame({"DaysInMilk": [10], "MilkingYield": [30.0]})
    with pytest.raises(ValueError, match="TestId"):
        islc_mod.create_standard_lc_representation(
            df,
            _standard_curve_series(),
            "DaysInMilk",
            "MilkingYield",
        )


@pytest.mark.representation
def test_create_standard_lc_representation_requires_dim_and_yield_columns() -> None:
    """Representation builder validates required DIM and milk-yield columns."""
    df = pd.DataFrame({"TestId": [1], "x": [10], "y": [30.0]})
    with pytest.raises(ValueError, match="must be in df"):
        islc_mod.create_standard_lc_representation(
            df,
            _standard_curve_series(),
            "DaysInMilk",
            "MilkingYield",
        )


@pytest.mark.representation
def test_create_standard_lc_representation_raises_when_no_rows_interpolated() -> None:
    """Representation builder fails when all groups interpolate to empty frames."""
    df = pd.DataFrame(
        {
            "TestId": [1, 1],
            "DaysInMilk": [1, 2],
            "MilkingYield": [30.0, 29.0],
        }
    )
    with pytest.raises(ValueError, match="No interpolated rows"):
        islc_mod.create_standard_lc_representation(
            df,
            _standard_curve_series(),
            "DaysInMilk",
            "MilkingYield",
        )


@pytest.mark.representation
def test_create_standard_lc_representation_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Representation builder returns expected output types and lengths."""
    df = pd.DataFrame(
        {
            "TestId": [1, 1, 2, 2],
            "DaysInMilk": [10, 50, 10, 50],
            "MilkingYield": [30.0, 26.0, 28.0, 24.0],
        }
    )

    def fake_fit_lactation_curve(*args: object, **kwargs: object) -> np.ndarray:
        return np.linspace(10.0, 20.0, 305)

    monkeypatch.setattr(islc_mod, "fit_lactation_curve", fake_fit_lactation_curve)

    corr, std_per_grid_day, curve_grid = islc_mod.create_standard_lc_representation(
        df,
        _standard_curve_series(),
        "DaysInMilk",
        "MilkingYield",
    )

    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    assert isinstance(std_per_grid_day, np.ndarray)
    assert std_per_grid_day.shape[0] == corr.shape[0]
    assert isinstance(curve_grid, pd.Series)
    assert len(curve_grid) == 305


@pytest.mark.representation
def test_create_standard_lc_representation_adds_testid_when_interpolator_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Representation builder re-adds TestId if custom interpolation omits that column."""
    df = pd.DataFrame(
        {
            "TestId": [11, 11, 12, 12],
            "DaysInMilk": [10, 30, 10, 30],
            "MilkingYield": [30.0, 28.0, 29.0, 27.0],
        }
    )

    def fake_interpolation_method(
        group: pd.DataFrame,
        column_name_dim: str,
        milking_yield_col: str,
        standard_lc: pd.Series | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "GridDay": [10, 30],
                "MilkYieldInterp": [
                    float(group[milking_yield_col].iloc[0]),
                    float(group[milking_yield_col].iloc[-1]),
                ],
            }
        )

    def fake_fit_lactation_curve(*args: object, **kwargs: object) -> np.ndarray:
        return np.ones(305)

    monkeypatch.setattr(islc_mod, "fit_lactation_curve", fake_fit_lactation_curve)

    corr, std_per_grid_day, curve_grid = islc_mod.create_standard_lc_representation(
        df,
        _standard_curve_series(),
        "DaysInMilk",
        "MilkingYield",
        interpolation_method=fake_interpolation_method,
    )

    assert isinstance(corr, pd.DataFrame)
    assert isinstance(std_per_grid_day, np.ndarray)
    assert isinstance(curve_grid, pd.Series)


def test_run_demo_executes_with_monkeypatched_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_demo executes without touching filesystem when I/O is monkeypatched."""
    fake_df = pd.DataFrame(
        {
            "DaysInMilk": list(range(1, 13)),
            "TestDayMilkYield": [float(v) for v in range(10, 22)],
        }
    )
    fake_result = pd.DataFrame({"305_milk_yield": [1234.5], "TestId": [0]})
    fake_icar_result = pd.DataFrame({"lactation_milk_yield": [1200.0], "TestId": [0]})

    monkeypatch.setattr(islc_mod.pd, "read_csv", lambda *args, **kwargs: fake_df.copy())
    monkeypatch.setattr(islc_mod, "ISLC", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr(islc_mod, "ISLC_ICAR", lambda *args, **kwargs: fake_icar_result)

    islc_mod._run_demo()
