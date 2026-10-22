"""
Centering date method for accumulated lactation yield.

Purpose
-------
This module implements the centering date method to compute cumulative milk yield
from test-day records.
This method is also known as the McDowell mehod for centering the testing day.
This approach was the standard method used by U.S. DHIA before 1969.

Method Summary
--------------
The centering date method (CDM) estimates lactation yield by assuming that each
test-day milk yield is representative for the calender month it was taken in.
It estimates total milk yield over a lactation by taking a milk yield measurement
each month and multiplying it by the number of days in that month.
If a cow calved before the start of the month,
the extra days are multiplied by the first test-day milk yield. If a cow goes
dry after the end of the month, the extra days are multiplied by the last
test-day milk yield.
So each testing interval has the same amount of days as days in that month,
whereby the testing day is considered as the middle of the testing period.
The total lactation yield is then calculated by multiplying each test-day yield
by the length of its corresponding centered interval and summing these
contributions across the lactation.


Formula:

MY_total = (d₁ + ℓ₁/2) · y₁ + Σ ℓᵢ · yᵢ (i = 2 … n−1) + (L − dₙ + ℓₙ/2) · yₙ

Where:
- ``d``: days in milk of the test day
- ``i``: index of the test day in the lactation
- ``yᵢ``: milk yield measured in the 24 hours of the test day i
- ``ℓ``: number of days in the month that the measurement was taken
- ``L``: lactation length (e.g., 305 days or the days in milk that the cow went dry)
- ``n``: number of test days in the lactation
- ``MY_total``: total cumulative milk yield over the lactation



Key Entry Points
----------------
- ``centering_date_method``: Computes cumulative lactation milk yield for
    a group of lactations, each identified by a unique ``TestId``
    and returns the results as a DataFrame with a ``TestId`` and ``LactationMilkYield`` column.
- ``centering_date_method_single_lactation``: Computes cumulative lactation
    milk yield for a single lactation.


Column Flexibility
------------------
The function accepts several case-insensitive column name aliases and can
create a default ``TestId`` if one is missing. Recognized aliases:

- Days in Milk: `["daysinmilk", "dim", "testday"]`
- Milk Yield: `["milkingyield", "testdaymilkyield", "milkyield", "yield"]`
- Test Id: `["animalid", "testid", "id"]`

It is also possible to provide your own column names so the function
can be applied to dataframes with different column naming conventions.

The main function returns a DataFrame with columns: ``["TestId", "LactationMilkYield"]``.

Notes
-----
- Units: DIM (days in milk) is measured in days,
    and milk yield can be in kg or lb. The
    output stays in the same unit as the input.
- Records with ``DIM > lactation_length`` are excluded before computation.
- This method's main strength is its ease of use and simplicity.
- Its main disadvantages are that it does not account for the shape of the
    lactation curve, which can lead to overestimation or underestimation
    of total yield,
    especially for lactations with few test days or irregular patterns.
    Outliers in test-day records can also have a large influence on the final
    result.
- This deviation is most prevalent at the beginning and end of the lactation
    as it ignores the increase in milk yield at the start of lactation
    and the decrease at the end of lactation.
    Everett et al. (1968) has proposed correction factors to account for this,
    but they are not implemented in this method.
    The ICAR guideline allows for adjustment of the first recorded value when
    computing the accumulated record.
    If the first recording occurs within 14 days from calving,
    then no adjustment is required to the first recorded value when computing
    the accumulated record. If the first recording occurs 15 to 95 days from
    calving, then an adjustment procedure may be applied.
- Months are of inequal length so the cummulative milk yield is biased
    by when the cow freshens, e.g. a cow that has her peak yield in January
    will likely have a higher cumulative milk yield.
    according to this method than a cow that has her peak yield in February,
    even if they have the same total milk yield over the lactation.
- Because the method assumes a once per month testing scheme,
    you cannot use it with a testing scheme that has multiple tests per month.
- This method works best with a very consistent once per month testing scheme.
    For instance always take the first day of the month so that the interval
    length is always the same.
- If a month is missing, the milk yield for that month is estimated as the average
    of the nearest test before and the nearest test after.
- If the first test day is missing, the second test day is used to estimate
the milk yield for the first month.
- This method can also be applied to bimonthly or trimonthly testing schemes,
    but the accuracy rapidly decreases as the number of test days per lactation
    decreases.
    This has been shown by Everett et al. (1968) reporting errors up to 28%
    and is also true for the test interval method.
-The code takes into account leap years if dates include years.
    The milk yield of February will than be multiplied by 29 instead of 28.
- This method is different from the test interval method **only**
because it uses the number of days in the month as the interval length
instead of the actual number of days between test days.
Otherwise they mathematically yield the same result.



Reference
---------
U.S. Department of Agriculture, Bureau of Dairying. (1925).
Cow Testing Association Letter No. 1. Washington, DC.

R.W. Everett, B.T. McDaniel, H.W. Carter,
Accuracy of Monthly, Bimonthly, and Trimonthly Dairy Herd Improvement Association Records,
Journal of Dairy Science,
Volume 51, Issue 7,
1968,
Pages 1051-1058,
ISSN 0022-0302,
https://doi.org/10.3168/jds.S0022-0302(68)87123-1.


---------
Author: Meike van Leerdam
Date: 07-31-2025
Last update: 22-May-2026
"""

import numpy as np
import pandas as pd

from lactationcurve.preprocessing import standardize_lactation_columns


def centering_date_method_single_lactation(
    lactation_df: pd.DataFrame,
    lactation_length: int = 305,
    lactation_start_grace_period: int = 0,
) -> float:
    """Compute total lactation milk yield using the centering date method.

    Args:
        lactation_df (pd.DataFrame): Input DataFrame that contains all milk
            recordings of a single lactation
            with at least TestCalendarDate,DaysInMilk and MilkingYield columns.
            Column names can be provided explicitly or matched via known aliases.
        lactation_length (int): Lactation length used to calculate cumulative
            production. The default is 305 days.
            Records with DIM > lactation_length are excluded.
            If dry dates are known it is recommended to use the actual lactation
            length instead of the default 305 days.
        lactation_start_grace_period (int): Number of days after calving that
            are not considered for cumulative yield becuase the colostrum is
            not representative for the lactation. The default is 0 days.

    Returns:
        float: The computed total milk yield over the lactation for a single cow.

    """
    df = lactation_df.sort_values("TestCalendarDate")

    # One yield per calendar month (if a month has several tests, take their mean)
    month = df["TestCalendarDate"].dt.to_period("M")
    y = df.groupby(month)["MilkingYield"].mean()

    # Full monthly range from first to last test; missing months become NaN
    full_range = pd.period_range(y.index.min(), y.index.max(), freq="M")
    y = y.reindex(full_range)

    # Missing month = average of the nearest test before and the nearest test after
    y = y.fillna(y.ffill().add(y.bfill()).div(2))

    # Weights: days in each month, with the first and last month adjusted
    w = np.array([period.days_in_month for period in full_range], dtype=float)
    d_first = df["DaysInMilk"].iloc[0] - lactation_start_grace_period
    d_last = df["DaysInMilk"].iloc[-1]
    w[0] = d_first + w[0] / 2
    w[-1] = lactation_length - d_last + w[-1] / 2

    total_milk_yield = (w * y).sum()
    return total_milk_yield


def centering_date_method(
    df: pd.DataFrame,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    test_calender_date_col: str | None = None,
    default_test_id: int = 0,
    lactation_length: int = 305,
    lactation_start_grace_period: int = 0,
) -> pd.DataFrame:
    """Compute total lactation milk yield using the centering date method for each test id.

    The method applies:


    Args:
        df (pd.DataFrame): Input DataFrame with at least DaysInMilk and
            MilkingYield columns, plus an optional TestId column. Column names
            can be provided explicitly or matched via known aliases.
        days_in_milk_col (str | None): Optional column name override for
            DaysInMilk.
        milking_yield_col (str | None): Optional column name override for
            MilkingYield.
        test_id_col (str | None): Optional column name override for TestId.
        test_calender_date_col (str | None): Optional column name override for
        TestCalendarDate.
        default_test_id (int): Value used to create a default TestId column if
            one is missing.
        lactation_length (int): Lactation length used to calculate cumulative
            production. The default is 305 days.
            Records with DIM > lactation_length are excluded.
        lactation_start_grace_period (int): Number of days after calving that
            are not considered for cumulative yield becuase the colostrum is
            not representative for the lactation. The default is 0 days.

    Returns:
        pd.DataFrame: Two-column DataFrame with
            - "TestId": identifier per lactation,
            - "LactationMilkYield": computed total milk yield over the
              specified window.

    Raises:
        ValueError: If required columns (DaysInMilk or MilkingYield) cannot be found.
        ValueError: If there are multiple measurements in the same calendar month.
        ValueError: If there are fewer than two data points per TestId.

    """

    # Standardize columns and filter DIM <= lactation_length
    df = standardize_lactation_columns(
        df,
        days_in_milk_col=days_in_milk_col,
        milking_yield_col=milking_yield_col,
        test_id_col=test_id_col,
        test_calender_date_col=test_calender_date_col,
        default_test_id=default_test_id,
        max_dim=lactation_length,
        lactation_start_grace_period=lactation_start_grace_period,
    )

    result = []

    # Iterate over each lactation
    for lactation in df["TestId"].unique():
        lactation_df = pd.DataFrame(df[df["TestId"] == lactation])

        if len(lactation_df) < 2:
            print(f"Skipping TestId {lactation}: not enough data points for interpolation.")
            continue

        # Check that the date column is datetime and convert it if needed.
        if not pd.api.types.is_datetime64_any_dtype(lactation_df["TestCalendarDate"]):
            lactation_df["TestCalendarDate"] = pd.to_datetime(lactation_df["TestCalendarDate"])

        # Check that there is no more than one test per calendar month
        month_counts = lactation_df["TestCalendarDate"].dt.to_period("M").value_counts()

        if (month_counts > 1).any():
            duplicated_months = month_counts[month_counts > 1].index.tolist()

            raise ValueError(
                f"TestId {lactation}: multiple measurements found in "
                f"calendar month(s) {duplicated_months}. "
                "The McDowell centering date method requires maximum one test per calendar month."
            )

        # Calculate total milk yield for this lactation
        total_milk_yield = centering_date_method_single_lactation(
            lactation_df,
            lactation_start_grace_period=lactation_start_grace_period,
            lactation_length=lactation_length,
        )

        result.append({"TestId": lactation, "LactationMilkYield": total_milk_yield})

    return pd.DataFrame(result, columns=pd.Index(["TestId", "LactationMilkYield"]))
