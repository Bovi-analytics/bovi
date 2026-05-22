"""
ICAR Test Interval Method for accumulated lactation yield.

Purpose
-------
This module implements the ICAR Test Interval Method (Procedure 2,
Section 2) to compute cumulative milk yield from test-day records.

Method Summary
--------------
Calculate total milk yield over a lactation by summing three parts:
        - First test-day milk yield multiplied by the number of days between
            calving and the first test day.
        - Trapezoidal integration between consecutive test days (the average of
            two consecutive yields multiplied by the interval length).
        - Last test-day milk yield multiplied by the number of days from the last
            test day to the end of the calculation window.

Formula:

    MY = I0*M1
         + I1*(M1 + M2)/2
         + I2*(M2 + M3)/2
         + ...
         + I(n-1)*(M(n-1) + Mn)/2
         + In*Mn

Where:
- ``MY``: total milk yield over the lactation window.
- ``M1, M2, ..., Mn``: milk yield measured in the 24 hours of each test day.
- ``I1, I2, ..., I(n-1)``: interval lengths (days) between consecutive test days.
- ``I0``: days from lactation start (calving) to first test day.
- ``In``: days from last test day to the end of the calculation window (e.g., DIM 305).


Key Entry Points
----------------
- ``test_interval_method``: Computes cumulative lactation milk yield per
    ``TestId`` using start, interval, and end segments.


Column Flexibility
------------------
The function accepts several case-insensitive column name aliases and can
create a default ``TestId`` if one is missing. Recognized aliases:

- Days in Milk: `["daysinmilk", "dim", "testday"]`
- Milk Yield: `["milkingyield", "testdaymilkyield", "milkyield", "yield"]`
- Test Id: `["animalid", "testid", "id"]`

Returns a DataFrame with columns: ``["TestId", "LactationMilkYield"]``.

Notes
-----
- Units: DIM is measured in days, and milk yield can be in kg or lb. The
    output stays in the same unit as the input.
- Records with ``DIM > max_dim`` are excluded before computation.
- This method's main strength is its ease of use and simplicity.
- Its main disadvantage is that it does not account for the shape of the
    lactation curve, which can lead to underestimation of total yield,
    especially for lactations with few test days or irregular patterns.
    Outliers in test-day records can also have a large influence on the final
    result.

Author: Meike van Leerdam
Date: 07-31-2025
Last update: 22-May-2026
"""

import pandas as pd

from lactationcurve.preprocessing import standardize_lactation_columns


def test_interval_method(
    df: pd.DataFrame,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    default_test_id: int = 0,
    max_dim: int = 305,
) -> pd.DataFrame:
    """Compute total lactation milk yield using the ICAR Test Interval Method.

    The method applies:
    - First test day milk yield from calving to the first test day,
    - Trapezoidal integration between consecutive test days,
    - Last test day milk yield from the last test day to DIM = max_dim (default = 305).

    Args:
        df (pd.DataFrame): Input DataFrame with at least DaysInMilk and
            MilkingYield columns, plus an optional TestId column. Column names
            can be provided explicitly or matched via known aliases.
        days_in_milk_col (str | None): Optional column name override for
            DaysInMilk.
        milking_yield_col (str | None): Optional column name override for
            MilkingYield.
        test_id_col (str | None): Optional column name override for TestId.
        default_test_id (int): Value used to create a default TestId column if
            one is missing.
        max_dim (int): Lactation length used to calculate cumulative
            production. The default is 305 days.
            Records with DIM > max_dim are excluded.

    Returns:
        pd.DataFrame: Two-column DataFrame with
            - "TestId": identifier per lactation,
            - "LactationMilkYield": computed total milk yield over the
              specified window.

    Raises:
        ValueError: If required columns (DaysInMilk or MilkingYield) cannot be found.

    Notes:
        - Records with DIM > max_dim are dropped before computation.
        - At least two data points per TestId are required for trapezoidal integration;
          otherwise the lactation is skipped.
    """

    # Standardize columns and filter DIM <= max_dim
    df = standardize_lactation_columns(
        df,
        days_in_milk_col=days_in_milk_col,
        milking_yield_col=milking_yield_col,
        test_id_col=test_id_col,
        default_test_id=default_test_id,
        max_dim=max_dim,
    )

    result = []

    # Iterate over each lactation
    for lactation in df["TestId"].unique():
        lactation_df = pd.DataFrame(df[df["TestId"] == lactation])

        # Sort by DaysInMilk ascending
        lactation_df.sort_values(by="DaysInMilk", ascending=True, inplace=True)

        if len(lactation_df) < 2:
            print(
                f"Skipping TestId {lactation}: not enough data points for "
                "interpolation."
            )
            continue

        # Start and end points
        start = lactation_df.iloc[0]
        end = lactation_df.iloc[-1]

        # Start contribution
        MY0 = start["DaysInMilk"] * start["MilkingYield"]

        # End contribution
        MYend = (max_dim + 1 - end["DaysInMilk"]) * end["MilkingYield"]

        # Intermediate trapezoidal contributions
        lactation_df["width"] = lactation_df["DaysInMilk"].diff().shift(-1)
        lactation_df["avg_yield"] = (
            lactation_df["MilkingYield"] + lactation_df["MilkingYield"].shift(-1)
        ) / 2
        lactation_df["trapezoid_area"] = (
            lactation_df["width"] * lactation_df["avg_yield"]
        )

        total_intermediate = lactation_df["trapezoid_area"].sum()

        total_yield = MY0 + total_intermediate + MYend
        result.append((lactation, total_yield))

    return pd.DataFrame(result, columns=pd.Index(["TestId", "LactationMilkYield"]))


# to prevent pytest from trying to collect this function as a test
test_interval_method.__test__ = False
