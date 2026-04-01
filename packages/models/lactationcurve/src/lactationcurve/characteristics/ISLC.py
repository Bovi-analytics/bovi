"""Interpolation using Standard Lactation Curves (ISLC).

This module implements the ISLC (Iterative Standard Lactation Curve)
method for estimating lactation milk production from intermittent test-day
records. The implementation follows the CRV/ICAR approach based on Wilmink-
style standard lactation curves and correlation-based deviation corrections.

The main entry points are:
- ``ISLC_method``: compute a single lactation's estimated 305-d yield from
    interpolated grid measurements.
- ``ISLC``: apply ``ISLC_method`` per `TestId` in a pandas DataFrame.
- ``ISLC_ICAR``: apply ``ISLC_ICAR_method`` per `TestId` in a pandas
    DataFrame.
- ``interpolation_standard_lc``: interpolate measured test-day yields onto
    the DIM grid using the standard lactation curve to guide slopes.
- ``linear_interpd_all_to_grid`` and ``linear_interpd_closest_to_grid``:
    alternative linear interpolation helpers.

Notes
-----
- Milk samples are often recorded on non-grid DIM values.
    Interpolation is therefore used to map measurements to grid days.
- Three interpolation variants are implemented.
    ``interpolation_standard_lc`` is recommended because it is the most
    consistent with the original method.
- ``interpolation_standard_lc`` requires a standard lactation curve.
    You can fit one with the ``lactationcurve`` package.
- The methods can be applied to lactations without any measurements,
    in which case the result will be the population mean from the standard curve.


Author: Meike
Date: 20 October 2025
Added: 03 March 2026
"""

# Import packages
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

from lactationcurve.fitting import fit_lactation_curve
from lactationcurve.preprocessing import standardize_lactation_columns

DATA_DIR = Path(__file__).resolve().parent / "data"
GRID_DAYS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290]

# get the standard lactation curve ingredients back from the data storage:
CORR_MATRIX = cast(pd.DataFrame, pd.read_pickle(DATA_DIR / "corr_matrix.pkl"))
STDs = np.load(DATA_DIR / "std_per_grid_day.npy")
STANDARD_CURVE = np.load(DATA_DIR / "standard_lc_grid.npy")

# add a day zero (that equals day 1) to the standard curve to prevent an index shift
STANDARD_CURVE = np.insert(STANDARD_CURVE, 0, STANDARD_CURVE[0])


def _curve_to_series(curve: pd.Series | npt.NDArray[np.float64]) -> pd.Series:
    """Normalize standard lactation curve input to a Series."""
    if isinstance(curve, pd.Series):
        return curve
    return pd.Series(curve)


def _curve_value(curve: pd.Series, day: int) -> float:
    """Return curve value for a DIM, supporting both label and positional indexing."""
    if day in curve.index:
        return float(curve.loc[day])
    return float(curve.iloc[day])


class InterpolationMethod(Protocol):
    """Protocol for milk-yield interpolation methods.

    Defines the signature expected by functions that interpolate test-day
    yields onto a predefined DIM grid.
    """

    def __call__(
        self,
        group: pd.DataFrame,
        column_name_dim: str,
        milking_yield_col: str,
        standard_lc: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Interpolate yields for a single lactation onto the DIM grid.

        Args:
            group: Test-day records for a single animal.
            column_name_dim: Column name containing days in milk (DIM).
            milking_yield_col: Column name containing milk yields.
            standard_lc: Optional standard lactation curve to guide
                interpolation (method-dependent).

        Returns:
            A DataFrame with interpolated yields, typically including
            columns ``GridDay`` and ``MilkYieldInterp``.
        """
        ...


def ISLC_method(
    df: pd.DataFrame,
    standard_lc: pd.Series | npt.NDArray[np.float64],
    correlation_matrix: pd.DataFrame,
    std_per_grid_day: npt.NDArray[np.float64],
) -> float:
    """Estimate 305-day milk yield for a single interpolated lactation.

    This function computes an estimated 305-day yield by combining three
    components: known production from interpolated measurements, the
    population mean for missing grid days (from ``standard_lc``), and a
    correlation-based deviation correction derived from ``correlation_matrix``.

    Args:
        df: DataFrame containing at least the columns ``GridDay`` and
            ``MilkYieldInterp`` (values already interpolated onto the grid).
        standard_lc: Standard lactation curve values indexed by grid day.
        correlation_matrix: Correlation coefficients between grid-day yields
            (matrix-like, rows/columns ordered as the grid).
        std_per_grid_day: 1-D array of standard deviations for each grid day.

    Returns:
        A float with the estimated 305-day milk yield.

    Raises:
        ValueError: if any measured day in ``df`` is not present in the
            expected global ``grid``.

    Notes:
        - The routine assumes fixed recording intervals of 20 days, except
          that the final interval (if day 290 is present) is 25 days.
        - The implementation expects a global ``grid`` variable to be
          defined and aligned with ``standard_lc`` and
          ``std_per_grid_day``.
        - The amount of estimated milk yields depends on the
            availability of measured milk yields.
        - Lactation yield is:
            sum(known measurements (p) + estimated measurements (q)).
        - In the implementation according to the CRV E2 document,
            there should also be a correction factor for the difference between
             the expected and actual comple lactation yield of the previous lactation.
            However, this correction factor is not included in the current implementation
            as it requires additional data (previous lactation yield) that is not always available.
    """
    grid = GRID_DAYS
    standard_lc_series = _curve_to_series(standard_lc)

    # --- Validate that all measurements lie on the expected grid -------
    measured_days = df["GridDay"].values
    if any(day not in grid for day in measured_days):
        raise ValueError("At least one DIM in the input df is not part of the grid.")

    # --- Prepare basic components -------------------------------------
    df = df.sort_values("GridDay")

    # days missing from measurement → need prediction
    days_to_predict = [day for day in grid if day not in measured_days]
    pred_idx = [list(grid).index(d) for d in days_to_predict]

    # mean expected milk yield from standard curve
    mean_lc = pd.Series([_curve_value(standard_lc_series, g) for g in grid], index=grid)

    # --- Compute known production (sum of actual measurements) ---------
    if 290 in measured_days:
        # special last interval: 25 days
        known_prod = df["MilkYieldInterp"][:-1].sum() * 20 + df["MilkYieldInterp"].iloc[-1] * 25
    else:
        known_prod = df["MilkYieldInterp"].sum() * 20

    # --- Compute population mean for missing days ----------------------
    pop_mean_missing = mean_lc.loc[days_to_predict]

    if 290 in pop_mean_missing.index:
        population_mean = pop_mean_missing.iloc[:-1].sum() * 20 + pop_mean_missing.iloc[-1] * 25
    else:
        population_mean = pop_mean_missing.sum() * 20

    # --- Compute correlation-based correction --------------------------
    days_to_predict_np = np.array(days_to_predict)
    measured_days_np = np.array(measured_days)

    # for each missing day: find closest measured day
    closest_measured = [
        measured_days_np[np.argmin(np.abs(measured_days_np - g))] for g in days_to_predict_np
    ]
    closest_idx = [list(grid).index(d) for d in closest_measured]

    b_star_list = []
    for pi, qi in zip(closest_idx, pred_idx):
        bpj = correlation_matrix.iloc[pi, qi]
        stdj = std_per_grid_day[qi]
        stdp = std_per_grid_day[pi]
        b_star_list.append(bpj * stdj / stdp)

    # scale correction by interval lengths
    if 290 in measured_days:
        # last interval belongs to prediction group
        correction_weighted = sum(b_star_list[:-1]) * 20
    else:
        correction_weighted = sum(b_star_list[:-1]) * 20 + b_star_list[-1] * 25

    last_day = df["GridDay"].iloc[-1]
    last_yield = df["MilkYieldInterp"].iloc[-1]
    mean_last = mean_lc.loc[last_day]

    correction = correction_weighted * (last_yield - mean_last)

    # --- Final sum -----------------------------------------------------
    final_milk_yield = known_prod + population_mean + correction
    return float(final_milk_yield)


def ISLC(
    df: pd.DataFrame,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    default_test_id: int = 0,
    standard_lc_305: pd.Series | npt.NDArray[np.float64] = STANDARD_CURVE,
    correlation_matrix: pd.DataFrame = CORR_MATRIX,
    std_per_grid_day: npt.NDArray[np.float64] = STDs,
) -> pd.DataFrame:
    """Compute estimated 305-day milk yields for lactations in a DataFrame.

    The function groups ``df`` by ``TestId`` (or treats the entire frame as a
    single lactation if no ``TestId`` column exists), interpolates observed
    yields onto the DIM grid, and applies :func:`ISLC_method` to each
    lactation to produce an estimated 305-d yield.

    Args:
       df (pd.DataFrame): Input DataFrame with at least DaysInMilk, MilkingYield,
            and (optionally) TestId columns (names can be provided via arguments
            or matched via known aliases, case-insensitive).
        days_in_milk_col (str | None): Column name in ``df`` that contains DIM.
        milking_yield_col (str | None): Column name in ``df`` that contains measured
            milk yield for the corresponding DIM.
        test_id_col (str | None): Column name in ``df`` that identifies
            unique lactations (e.g., animal ID or lactation ID).
            If None, a single TestId==default_test_id is created.
        default_test_id (int): If ``test_id_col`` is None, a new column
            ``TestId`` is created with this value for all rows.
            This means that the entire input DataFrame will be treated as one lactation.
        standard_lc_305 (pd.Series | npt.NDArray[np.float64]): Standard
            lactation curve values aligned to the grid
            (default: module ``STANDARD_CURVE``).
        correlation_matrix (pd.DataFrame): Correlation matrix for grid-day yields
            (default: module ``CORR_MATRIX``).
        std_per_grid_day (npt.NDArray[np.float64]): Standard deviations per
            grid day (default: module
            ``STDs``).

    Returns:
        pd.DataFrame: Two-column DataFrame with
            - "TestId": identifier per lactation,
            - "305_milk_yield": computed total milk yield over 305 days.

    Raises:
        ValueError: If required columns (DaysInMilk or MilkingYield) cannot be found.

    Notes:
        - Records with DIM > 305 are dropped before computation.
        - If ``TestId`` is absent in ``df``, a single
            TestId==default_test_id is created and
            the entire input is processed as one lactation.
    """
    # Standardize columns and filter DIM <= 305
    df = standardize_lactation_columns(
        df,
        days_in_milk_col=days_in_milk_col,
        milking_yield_col=milking_yield_col,
        test_id_col=test_id_col,
        default_test_id=default_test_id,
        max_dim=305,
    )

    rows: list[dict[str, Any]] = []
    for test_id, group in df.groupby("TestId", sort=False):
        lactation = interpolation_standard_lc(
            group,
            column_name_dim="DaysInMilk",
            milking_yield_col="MilkingYield",
            standard_lc=standard_lc_305,
        )
        if lactation.empty:
            rows.append({"TestId": test_id, "305_milk_yield": np.nan})
            continue

        cumulative_yield = ISLC_method(
            df=lactation,
            standard_lc=standard_lc_305,
            correlation_matrix=correlation_matrix,
            std_per_grid_day=std_per_grid_day,
        )
        rows.append({"TestId": test_id, "305_milk_yield": cumulative_yield})

    return pd.DataFrame(rows, columns=["305_milk_yield", "TestId"])


def ISLC_ICAR(
    df: pd.DataFrame,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    default_test_id: int = 0,
    standard_lc_305: pd.Series | npt.NDArray[np.float64] = STANDARD_CURVE,
    correlation_matrix: pd.DataFrame = CORR_MATRIX,
    std_per_grid_day: npt.NDArray[np.float64] = STDs,
    max_dim: int | str = 305,
) -> pd.DataFrame:
    """Compute ICAR-based lactation milk yields for lactations in a DataFrame.

    This method adapts the original ISLC approach with some key differences:
    - It includes measured values in the final integration, not only
      interpolated grid values.
    - The prediction grid includes day 0 and day 305.
    - Because measured values are included,
        the method can be applied to lactation longer then 305 days
        by adjusting the ``max_dim`` parameter.
    - The values above 305 DIM will not be taken into account during interpolation.


    The function groups ``df`` by ``TestId`` (or treats the entire frame as a
    single lactation if no ``TestId`` column exists) and applies
    :func:`ISLC_ICAR_method` to each lactation to produce an estimated
    lactation yield over the selected DIM horizon.

    Args:
        df (pd.DataFrame): Input DataFrame with at least DaysInMilk,
            MilkingYield, and (optionally) TestId columns (names can be
            provided via arguments or matched via known aliases,
            case-insensitive).
        days_in_milk_col (str | None): Column name in ``df`` that contains DIM.
        milking_yield_col (str | None): Column name in ``df`` that contains measured
            milk yield for the corresponding DIM.
        test_id_col (str | None): Column name in ``df`` that identifies
            unique lactations (e.g., animal ID or lactation ID).
            If None, a single TestId==default_test_id is created.
        default_test_id (int): If ``test_id_col`` is None, a new column
            ``TestId`` is created with this value for all rows.
            This means that the entire input DataFrame will be treated as one lactation.
        standard_lc_305 (pd.Series | npt.NDArray[np.float64]): Standard
            lactation curve values aligned to the grid
            (default: module ``STANDARD_CURVE``).
        correlation_matrix (pd.DataFrame): Correlation matrix for grid-day yields
            (default: module ``CORR_MATRIX``).
        std_per_grid_day (npt.NDArray[np.float64]): Standard deviations per
            grid day (default: module ``STDs``).
        max_dim (int | str): Maximum DIM to include in the analysis (default: 305).
            Records with DIM greater than this value will be dropped before computation.
            If set to ``"max"``, no DIM filtering is applied and all records are included.
            This changes the integration horizon and can increase total yield.


    Returns:
        pd.DataFrame: Two-column DataFrame with
                        - "lactation_milk_yield": computed total milk yield over the
                            selected DIM horizon,
            - "TestId": identifier per lactation.

    Raises:
        ValueError: If required columns (DaysInMilk or MilkingYield) cannot be found.

    Notes:
                - Records with DIM > ``max_dim`` are dropped before computation.
                - If ``max_dim="max"``, records are not filtered on DIM.
        - If ``TestId`` is absent in ``df``, a single
          TestId==default_test_id is created and
          the entire input is processed as one lactation.
                - Lactations without measured days return NaN for
                    ``lactation_milk_yield``.
    """
    # Standardize columns and filter DIM <= 305
    df = standardize_lactation_columns(
        df,
        days_in_milk_col=days_in_milk_col,
        milking_yield_col=milking_yield_col,
        test_id_col=test_id_col,
        default_test_id=default_test_id,
        max_dim=max_dim,
    )

    rows: list[dict[str, Any]] = []
    for test_id, group in df.groupby("TestId", sort=False):
        if group.empty:
            rows.append({"TestId": test_id, "305_milk_yield": np.nan})
            continue

        try:
            cumulative_yield = ISLC_ICAR_method(
                df=group,
                standard_lc=standard_lc_305,
                correlation_matrix=correlation_matrix,
                std_per_grid_day=std_per_grid_day,
                days_in_milk_col="DaysInMilk",
                milking_yield_col="MilkingYield",
            )
        except ValueError:
            cumulative_yield = np.nan

        rows.append({"TestId": test_id, "lactation_milk_yield": cumulative_yield})

    return pd.DataFrame(rows, columns=["lactation_milk_yield", "TestId"])


def ISLC_ICAR_method(
    df: pd.DataFrame,
    standard_lc: pd.Series | npt.NDArray[np.float64],
    correlation_matrix: pd.DataFrame,
    std_per_grid_day: npt.NDArray[np.float64],
    days_in_milk_col: str = "DaysInMilk",
    milking_yield_col: str = "MilkingYield",
) -> float:
    """Estimate total lactation yield by predicting missing grid days, then integrating.

    The method fills missing grid-day yields with the ICAR-style relation:

        yp = gp + b* (yi - gi)

    where for each missing day p, the nearest measured day i is used and
    b* = corr(i,p) * std(p) / std(i). The resulting complete grid profile is
    integrated with the test-interval style weighting used elsewhere in this
    package.

    Notes:
        The integration is performed over the days present after merging
        measured and predicted points. If the input contains DIM beyond 305,
        total yield can reflect that extended horizon.
    """
    # add zero and 305 to the grid
    grid = [0] + GRID_DAYS + [305]
    dim_col = days_in_milk_col
    yield_col = milking_yield_col

    standard_lc_series = _curve_to_series(standard_lc)

    # Extract the measured days
    measured_days = df[dim_col].to_numpy()
    if measured_days.size == 0:
        raise ValueError("Input df must contain at least one measured day.")

    # interpolate the measured days to the grid using the standard lactation curve
    interpolated_df = interpolation_standard_lc(
        group=df,
        column_name_dim=dim_col,
        milking_yield_col=yield_col,
        standard_lc=standard_lc_series,
    )

    # drop unnecessary columns
    interpolated_df = interpolated_df.drop(columns=[dim_col, yield_col])

    # rename grid colum
    interpolated_df = interpolated_df.rename(
        columns={"GridDay": dim_col, "MilkYieldInterp": yield_col}
    )[[dim_col, yield_col]]

    measured_subset = cast(pd.DataFrame, df.loc[:, [dim_col, yield_col]])
    interpolated_subset = cast(pd.DataFrame, interpolated_df.loc[:, [dim_col, yield_col]])

    # join the interpolated values with the original measurements, keeping all measurements
    merged_df = cast(
        pd.DataFrame,
        pd.merge(
            measured_subset,
            interpolated_subset,
            on=dim_col,
            how="outer",
            suffixes=("_meas", "_interp"),
        ),
    )
    meas_series = cast(pd.Series, merged_df[f"{yield_col}_meas"])
    interp_series = cast(pd.Series, merged_df[f"{yield_col}_interp"])
    merged_df[yield_col] = cast(pd.Series, meas_series.combine_first(interp_series))
    df = cast(
        pd.DataFrame,
        merged_df.loc[:, [dim_col, yield_col]].sort_values(by=dim_col),
    )

    # days missing from measurement → need prediction
    days_to_predict = [day for day in grid if day not in df[dim_col].values]

    predicted_rows: list[dict[str, float | int]] = []

    for day in days_to_predict:
        # identify closest measured day
        dim_series = cast(pd.Series, interpolated_df[dim_col])
        grid_vals = np.asarray(dim_series.to_numpy(dtype=float), dtype=float)
        closest_idx = int(np.argmin(np.abs(grid_vals - float(day))))
        closest_day = int(dim_series.iloc[closest_idx])

        # extract mean milk yield from the standard lactation curve for the day to predict
        expected_yield_grid_day = _curve_value(standard_lc_series, day)
        expected_yield_measured_day = _curve_value(standard_lc_series, closest_day)

        # calculate difference between measured and expected yield at closest measured day
        diff = (
            interpolated_df.loc[interpolated_df[dim_col] == closest_day, yield_col].iloc[0]
            - expected_yield_measured_day
        )

        # calculate the correlation-based correction factor b*
        # b* = corr(i,p) * std(p) / std(i)
        pi = list(grid).index(closest_day)
        qi = list(grid).index(day)
        bpj = correlation_matrix.iloc[pi, qi]
        std_q = std_per_grid_day[qi]
        std_p = std_per_grid_day[pi]
        b_star = 0.0 if std_p == 0 else float(bpj * std_q / std_p)

        # predict the yield for the missing day
        predicted_yield = expected_yield_grid_day + b_star * diff
        predicted_rows.append({dim_col: int(day), yield_col: float(predicted_yield)})

    if predicted_rows:
        df = pd.concat([df, pd.DataFrame(predicted_rows)], ignore_index=True)

    # apply the test interval method to calculate total milk yield over 305 days
    # sort the dataframe by days in milk and keep one value per day
    df = df.drop_duplicates(subset=[dim_col], keep="last")
    df = df.sort_values(by=dim_col)
    # Trapezoidal contributions
    df["width"] = df[dim_col].diff().shift(-1)
    df["avg_yield"] = (df[yield_col] + df[yield_col].shift(-1)) / 2
    df["trapezoid_area"] = df["width"] * df["avg_yield"]

    trapezoid_values = np.asarray(pd.to_numeric(df["trapezoid_area"], errors="coerce"), dtype=float)
    total_yield = float(np.nansum(trapezoid_values))

    return total_yield


"""
Create your own standard lactation curve, correlation matrix,
and standard deviations per grid day.

step 1 interpolate to the days on the grid
Based on the CRV E2 documents, interpolation is not linear and instead
depends on the standard lactation curve.
"""


def interpolation_standard_lc(
    group: pd.DataFrame,
    column_name_dim: str,
    milking_yield_col: str,
    standard_lc: pd.Series | npt.NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Interpolate a single lactation onto the DIM grid guided by a standard curve.

    The function interpolates measured yields for one animal onto the
    predefined grid of DIM values. When a grid day coincides with a measured
    DIM the observed yield is used; otherwise the interpolation adjusts the
    linear slope between neighboring measurements by the difference in the
    standard curve, inspired by the CRV E2 approach.

    The interpolation formula applied for a grid day ``gday`` between two
    measured points (x1,y1) and (x2,y2) is::

        slope = ((y2 - y1) - (g2 - g1)) / (x2 - x1)
        yi = gi + (slope * (gday - x1) + (y1 - g1))

    Args:
        group: DataFrame with test-day records for a single animal. Must
            contain columns named by ``column_name_dim`` and
            ``milking_yield_col``.
        column_name_dim: Name of the DIM column in ``group``.
        milking_yield_col: Name of the milk-yield column in ``group``.
        standard_lc: A Series indexed by DIM providing the expected yield on
            each grid day; used to guide the interpolation shape.

    Returns:
        A DataFrame with one row per interpolated grid day containing the
        original identifying columns (from the first row of ``group``), plus
        the columns ``GridDay`` and ``MilkYieldInterp``.

    Notes:
        - No interpolation is performed outside the range bounded by the
          first and last measured DIM for the lactation (those grid days are
          skipped).
                - No interpolation is performed for grid days where the right
                    neighboring measured DIM is >= 305, because this lies outside the
                    intended standard-curve support.
        - The implementation defines a local default ``grid`` of
          [10, 30, ..., 290].


    """
    if standard_lc is None:
        raise ValueError("A standard lactation curve is required for interpolation_standard_lc.")

    grid = GRID_DAYS
    standard_lc_series = _curve_to_series(standard_lc)

    # Sort and ensure unique DIMs
    group = group.sort_values(column_name_dim).drop_duplicates(column_name_dim)
    dims = group[column_name_dim].tolist()

    rows = []

    # loop over the grid days
    for gday in grid:
        # --- CASE 1: Exact measured day ---------------------------
        if gday in dims:
            y_val = group.loc[group[column_name_dim] == gday, milking_yield_col].iloc[0]

            row = group.iloc[0].to_dict()
            row["GridDay"] = gday
            row["MilkYieldInterp"] = float(y_val)
            rows.append(row)
            continue

        # --- CASE 2: interpolate between measured days -------------
        before_df = cast(pd.DataFrame, group.loc[group[column_name_dim] < gday])
        after_df = cast(pd.DataFrame, group.loc[group[column_name_dim] > gday])
        before = before_df.tail(1)
        after = after_df.head(1)

        # CASE 3 No interpolation if grid day is outside the range of measured DIM
        if before.empty or after.empty:
            continue

        # CASE 4: No interpolation if the day after is day 305 or higher,
        # as this is outside the range of the standard curve
        if float(after.iloc[0][column_name_dim]) >= 305:
            continue

        x1 = float(before.iloc[0][column_name_dim])
        y1 = float(before.iloc[0][milking_yield_col])
        x2 = float(after.iloc[0][column_name_dim])
        y2 = float(after.iloc[0][milking_yield_col])

        # expected yields from standard lactation curve
        g1 = _curve_value(standard_lc_series, int(x1))
        g2 = _curve_value(standard_lc_series, int(x2))
        gi = _curve_value(standard_lc_series, int(gday))

        # Wilmink-based interpolation formula
        slope = ((y2 - y1) - (g2 - g1)) / (x2 - x1)
        yi = gi + (slope * (gday - x1) + (y1 - g1))

        row = group.iloc[0].to_dict()
        row["GridDay"] = gday
        row["MilkYieldInterp"] = float(yi)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[*group.columns, "GridDay", "MilkYieldInterp"])

    return pd.DataFrame(rows)


def linear_interpd_all_to_grid(
    group: pd.DataFrame,
    column_name_dim: str,
    milking_yield_col: str,
    standard_lc: pd.Series | None = None,
) -> pd.DataFrame:
    """Linearly interpolate all grid days for a lactation.

    This helper uses linear interpolation (with extrapolation) to produce
    milk-yield values for every grid day regardless of whether the grid day
    lies between measured observations.

    Args:
        group: DataFrame containing measured DIM and yield values for one
            lactation.
        days_in_milk_col: Name of the DIM column.
        milking_yield_col: Name of the milk-yield column.

    Returns:
        A DataFrame with identifying columns copied from ``group``'s first
        row and columns ``GridDay`` and ``MilkYieldInterp`` containing the
        interpolated (or extrapolated) yields for every grid day.
    """
    grid = GRID_DAYS
    group = group.sort_values(column_name_dim).drop_duplicates(subset=column_name_dim)
    f = interp1d(
        group[column_name_dim],
        group[milking_yield_col],
        kind="linear",
        fill_value=cast(Any, "extrapolate"),
    )

    base = {
        col: group[col].iloc[0]
        for col in group.columns
        if col not in [column_name_dim, milking_yield_col]
    }
    base["GridDay"] = grid
    base["MilkYieldInterp"] = f(grid)

    return pd.DataFrame(base)


# Create an interpolation function that only outputs grid days between two
# milk measurements using linear interpolation.
def linear_interpd_closest_to_grid(
    group: pd.DataFrame,
    column_name_dim: str,
    milking_yield_col: str,
    standard_lc: pd.Series | None = None,
) -> pd.DataFrame | None:
    """Linearly interpolate grid days between measured observations.

    This helper returns interpolated yields only for grid days that lie
    between the first and last measured DIM for the lactation. If no grid
    days fall within the measured range the function returns ``None``.

    Args:
        group: DataFrame for a single lactation.
        days_in_milk_col: Name of the DIM column.
        milking_yield_col: Name of the milk-yield column.

    Returns:
        A DataFrame with identifying columns plus ``GridDay`` and
        ``MilkYieldInterp``, or ``None`` if there are no grid days within
        the measured range.
    """
    grid = GRID_DAYS

    group = group.sort_values(column_name_dim).drop_duplicates(subset=column_name_dim)

    # Find the range of interpolatable DIM
    min_dim = group[column_name_dim].min()
    max_dim = group[column_name_dim].max()
    grid_in_range = [g for g in grid if min_dim <= g <= max_dim]

    # apply linear interpolation
    if not grid_in_range:
        return None
    f = interp1d(
        group[column_name_dim],
        group[milking_yield_col],
        kind="linear",
        fill_value=cast(Any, np.nan),
    )

    # create a new dataframe with the newly created columns
    base = {
        col: group[col].iloc[0]
        for col in group.columns
        if col not in [column_name_dim, milking_yield_col]
    }
    base["GridDay"] = grid_in_range
    base["MilkYieldInterp"] = f(grid_in_range)

    return pd.DataFrame(base)


def create_standard_lc_representation(
    df: pd.DataFrame,
    standard_lactation_curve: pd.Series,
    column_name_dim: str,
    col_milk_yield: str,
    interpolation_method: InterpolationMethod = interpolation_standard_lc,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64], pd.Series]:
    """Create a standard lactation curve with correlation matrix from data.

    This function estimates the correlation structure and standard deviations
    of milk yields across grid days, and refits a Wilmink lactation curve to
    the interpolated data. The outputs can be used as inputs to the
    ``ISLC_method`` or ``ISLC`` functions for predicting 305-d yields.

    The process standardizes yields per animal and computes row-wise
    (per-animal) statistics, then derives relationships across grid days.

    Args:
        df: Input DataFrame containing test-day records. Must include a
            ``TestId`` column to identify unique lactations. Also requires
            columns specified by ``column_name_dim`` and ``col_milk_yield``.
        standard_lactation_curve: A Series providing a reference lactation
            curve (e.g., the fitted Wilmink curve on which to base
            interpolation).
        column_name_dim: Name of the column in ``df`` containing days in milk (DIM).
        col_milk_yield: Name of the column in ``df`` containing measured
            milk yields.
        interpolation_method: An interpolation method conforming to
            ``InterpolationMethod`` protocol (default: ``interpolation_standard_lc``).
            Must be callable with signature
            (group, column_name_dim, milking_yield_col, standard_lc).

    Returns:
        A tuple of three elements:
        - corr (pd.DataFrame): Correlation matrix between grid-day yields.
        - std_per_grid_day (np.ndarray): Standard deviations of standardized
          yields per grid day.
        - standard_lactation_curve_grid (pd.Series): Refitted Wilmink model
          indexed by DIM (1–305).

    Notes:
        - Expects ``df`` to have a ``TestId`` column to group lactations.
        - Performs standardization per animal (row-wise), so output
          correlations and standard deviations reflect between-day variation
          within standardized animal profiles.
        - Uses ``fit_lactation_curve`` from the ``lactationcurve`` package
          with model='Wilmink' and fitting='frequentist'.
    """
    if "TestId" not in df.columns:
        raise ValueError("DataFrame must contain a 'TestId' column.")
    if column_name_dim not in df.columns or col_milk_yield not in df.columns:
        raise ValueError(f"Columns '{column_name_dim}' and '{col_milk_yield}' must be in df.")

    interpolated_groups: list[pd.DataFrame] = []
    for test_id, group in df.groupby("TestId", sort=False):
        interp_group = interpolation_method(
            group,
            column_name_dim=column_name_dim,
            milking_yield_col=col_milk_yield,
            standard_lc=standard_lactation_curve,
        )
        if interp_group.empty:
            continue
        if "TestId" not in interp_group.columns:
            interp_group = interp_group.copy()
            interp_group["TestId"] = test_id
        interpolated_groups.append(interp_group)

    if not interpolated_groups:
        raise ValueError("No interpolated rows were produced; cannot build representation.")

    df_grid = pd.concat(interpolated_groups, ignore_index=True)

    # define Znp
    # pivot df to make GridDay the columns
    # Z should be the unscaled pivot table (TestId × GridDay)
    Z = df_grid.pivot(index="TestId", columns="GridDay", values="MilkYieldInterp")

    # Standardize records per cow (so each row has mean=0, std=1)
    Z_std = (Z.sub(Z.mean(axis=1), axis=0)).div(Z.std(axis=1), axis=0)

    # Convert to NumPy array
    Znp = Z_std.to_numpy()

    # calculate correlation matrix
    corr = pd.DataFrame(Znp, columns=Z.columns).corr()

    # calculate standard deviations for each grid day from the Znp matrix
    std_per_grid_day = np.nanstd(Znp, axis=0)  # ignores NaN

    # fit Wilmink model to get mean values and std for each cow
    standard_lactation_curve_grid = pd.Series(
        fit_lactation_curve(
            df_grid["GridDay"].values,
            df_grid["MilkYieldInterp"].values,
            model="Wilmink",
            fitting="frequentist",
        ),
        index=range(1, 306),
    )
    return corr, std_per_grid_day, standard_lactation_curve_grid


def _run_demo() -> None:
    """Run a local demonstration of ISLC on a sample CSV file.

    The sample path points to a local research directory and is intended for
    manual development runs only.
    """
    df_test = pd.read_csv(
        r"C:\Users\Meike van Leerdam\lactation-curve-research\test_data\L2Anim2Herd654.csv",
        sep=",",
    )

    df_test["MilkingYield"] = df_test["TestDayMilkYield"]
    df_test = df_test.drop(columns=["TestDayMilkYield"])
    df_test = df_test.loc[df_test["DaysInMilk"] <= 305].copy()

    # subset random samples from the test data to create a new dataframe with only 10 samples
    df_test_subset = (
        df_test.sample(n=5, random_state=42).sort_values("DaysInMilk").reset_index(drop=True)
    )

    # add row of a sample at a gridday as well to see function behavior
    df_test_subset = (
        pd.concat(
            [
                df_test_subset,
                pd.DataFrame(
                    {
                        "DaysInMilk": [350],
                        "MilkingYield": [10],
                    }
                ),
            ],
            ignore_index=True,
        )
        .sort_values("DaysInMilk")
        .reset_index(drop=True)
    )

    # test with multiple testid's
    df_test_subset["TestId"] = 1
    df_test_subset_two = (
        df_test.sample(n=5, random_state=32).sort_values("DaysInMilk").reset_index(drop=True)
    )
    df_test_subset_two["TestId"] = 2

    # combine
    df_test_subset = pd.concat([df_test_subset, df_test_subset_two], ignore_index=True).reset_index(
        drop=True
    )

    islc = ISLC(
        df=df_test_subset,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
    )

    print(f"ISLC method result for 305-day yield: {islc}")

    # try out the ICAR method as well
    islc_icar = ISLC_ICAR(
        df=df_test_subset,
        days_in_milk_col="DaysInMilk",
        milking_yield_col="MilkingYield",
    )

    print(f"ICAR method result for 305-day yield: {islc_icar}")


if __name__ == "__main__":
    _run_demo()
