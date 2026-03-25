"""
ICAR 305-day yield calculation — Test Interval Method.

This module implements the **Test Interval Method** described in ICAR guidelines
(Procedure 2, Section 2: Computing of Accumulated Lactation Yield) to compute
total **305-day milk yield** from test-day data.

Approach
--------
- **Start segment**: Linear projection from calving (DIM=0) to the first test day.
- **Intermediate segments**: **Trapezoidal rule** between consecutive test days.
- **End segment**: Linear projection from the last test day to DIM=305 (exclusive
  upper bound 306 for day counting).

Column Flexibility
------------------
The function can accept various column name aliases (case-insensitive) and
optionally create a default `TestId` if missing. Recognized aliases:

- Days in Milk: `["daysinmilk", "dim", "testday"]`
- Milk Yield: `["milkingyield", "testdaymilkyield", "milkyield", "yield"]`
- Test Id: `["animalid", "testid", "id"]`

Returns a DataFrame with columns: `["TestId", "Total305Yield"]`.

Notes
-----
- Units: DIM in days, milk yield in kg.
- Records with `DIM > 305` are excluded prior to computation.

Author: Meike van Leerdam, Date: 07-31-2025
"""

import pandas as pd

from lactationcurve.preprocessing import standardize_lactation_columns


def test_interval_method(
    df,
    days_in_milk_col=None,
    milking_yield_col=None,
    test_id_col=None,
    default_test_id=1,
) -> pd.DataFrame:
    """Compute 305-day total milk yield using the ICAR Test Interval Method.

    The method applies:
    - Linear projection from calving to the first test day,
    - Trapezoidal integration between consecutive test days,
    - Linear projection from the last test day to DIM=305.

    Args:
        df (pd.DataFrame): Input DataFrame with at least DaysInMilk, MilkingYield,
            and (optionally) TestId columns (names can be provided via arguments
            or matched via known aliases, case-insensitive).
        days_in_milk_col (str | None): Optional column name override for DaysInMilk.
        milking_yield_col (str | None): Optional column name override for MilkingYield.
        test_id_col (str | None): Optional column name override for TestId.
        default_test_id (Any): If TestId is missing, a new `TestId` column is created
            with this value.

    Returns:
        pd.DataFrame: Two-column DataFrame with
            - "TestId": identifier per lactation,
            - "Total305Yield": computed total milk yield over 305 days.

    Raises:
        ValueError: If required columns (DaysInMilk or MilkingYield) cannot be found.

    Notes:
        - Records with DIM > 305 are dropped before computation.
        - At least two data points per TestId are required for trapezoidal integration;
          otherwise the lactation is skipped.
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

    result = []

    # Iterate over each lactation
    for lactation in df["TestId"].unique():
        lactation_df = pd.DataFrame(df[df["TestId"] == lactation])

        # Sort by DaysInMilk ascending
        lactation_df.sort_values(by="DaysInMilk", ascending=True, inplace=True)

        if len(lactation_df) < 2:
            print(f"Skipping TestId {lactation}: not enough data points for interpolation.")
            continue

        # Start and end points
        start = lactation_df.iloc[0]
        end = lactation_df.iloc[-1]

        # Start contribution
        MY0 = start["DaysInMilk"] * start["MilkingYield"]

        # End contribution
        MYend = (306 - end["DaysInMilk"]) * end["MilkingYield"]

        # Intermediate trapezoidal contributions
        lactation_df["width"] = lactation_df["DaysInMilk"].diff().shift(-1)
        lactation_df["avg_yield"] = (
            lactation_df["MilkingYield"] + lactation_df["MilkingYield"].shift(-1)
        ) / 2
        lactation_df["trapezoid_area"] = lactation_df["width"] * lactation_df["avg_yield"]

        total_intermediate = lactation_df["trapezoid_area"].sum()

        total_yield = MY0 + total_intermediate + MYend
        result.append((lactation, total_yield))

    return pd.DataFrame(result, columns=["TestId", "Total305Yield"])


# to prevent pytest from trying to collect this function as a test
test_interval_method.__test__ = False
