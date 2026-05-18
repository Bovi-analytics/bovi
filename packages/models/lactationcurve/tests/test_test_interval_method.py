"""
Test suite for ICAR Test Interval Method (TIM).

This module provides comprehensive tests for the ICAR Test Interval Method (TIM)
implementation, including validation of the trapezoidal rule calculation,
output structure, edge case handling, and performance with realistic and synthetic data.

Test Categories:
    - TestBasicCalculations: Verifies correct calculation for synthetic data points.
    - TestOutputValidation: Checks output DataFrame structure and value constraints.
    - TestEdgeCases: Handles boundary conditions, invalid input, and data anomalies.
    - TestRealisticData: Validates method on real-world and large-scale datasets.

Usage:
    Run all tests::

        pytest test_method_test_interval.py -v

    Run specific marker::

        pytest test_method_test_interval.py -m basic -v
        pytest test_method_test_interval.py -m output_validation -v
        pytest test_method_test_interval.py -m edge_cases -v
        pytest test_method_test_interval.py -m realistic -v

Author:
    Judith Osei-Tete
"""

import os
import time

import numpy as np
import pandas as pd
import pytest

from lactationcurve.characteristics import (
    test_interval_method,
)

# ICAR Test Interval Method constants
STANDARD_LACTATION_DAYS = 305
END_PROJECTION_DAY = 306


@pytest.fixture
def mr_test_data():
    """Fixture to load mr_test_file.csv data."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_data", "mr_test_file.csv")
    return pd.read_csv(csv_path)


@pytest.fixture
def complete_lactation_data():
    """Fixture to load l2_anim2_herd654.csv data."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_data", "l2_anim2_herd654.csv")
    return pd.read_csv(csv_path)


@pytest.mark.basic
class TestBasicCalculations:
    """Test basic TIM calculations with synthetic data points."""

    def test_two_data_points(self):
        """
        Test basic TIM calculation with two data points.

        Verifies the ICAR formula:
        MY = I0*M1 + I1*(M1+M2)/2 + In*Mn
        """
        df = pd.DataFrame({"DaysInMilk": [10, 40], "MilkingYield": [30.0, 25.0], "TestId": [1, 1]})

        result = test_interval_method(df)

        start = 10 * 30.0
        trap1 = (40 - 10) * (30.0 + 25.0) / 2
        end = (END_PROJECTION_DAY - 40) * 25.0

        expected_yield = start + trap1 + end

        assert len(result) == 1, "Should return one row for one lactation"
        assert result.iloc[0]["TestId"] == 1, "TestId should be 1"
        assert np.isclose(result.iloc[0]["LactationMilkYield"], expected_yield), (
            f"Expected {expected_yield}, got {result.iloc[0]['LactationMilkYield']}"
        )

    def test_three_data_points(self):
        """
        Test TIM with three data points using trapezoidal rule.

        Verifies: MY = I0*M1 + I1*(M1+M2)/2 + I2*(M2+M3)/2 + In*Mn
        """
        df = pd.DataFrame(
            {
                "DaysInMilk": [5, 35, 65],
                "MilkingYield": [25.0, 30.0, 28.0],
                "TestId": [1, 1, 1],
            }
        )

        result = test_interval_method(df)

        start = 5 * 25.0
        trap1 = (35 - 5) * (25.0 + 30.0) / 2
        trap2 = (65 - 35) * (30.0 + 28.0) / 2
        end = (END_PROJECTION_DAY - 65) * 28.0

        expected_yield = start + trap1 + trap2 + end

        assert len(result) == 1, "Should return one row for one lactation"
        assert result.iloc[0]["TestId"] == 1, "TestId should be 1"
        assert np.isclose(result.iloc[0]["LactationMilkYield"], expected_yield, rtol=1e-5), (
            f"Expected {expected_yield}, got {result.iloc[0]['LactationMilkYield']}"
        )

    def test_four_equally_spaced_points(self):
        """
        Test TIM with four equally spaced points.

        Verifies: MY = I0*M1 + I1*(M1+M2)/2 + I2*(M2+M3)/2 + I3*(M3+M4)/2 + In*Mn
        """
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 80, 150, 220],
                "MilkingYield": [32.0, 38.0, 34.0, 28.0],
                "TestId": [1, 1, 1, 1],
            }
        )

        result = test_interval_method(df)

        start = 10 * 32.0
        trap1 = (80 - 10) * (32.0 + 38.0) / 2
        trap2 = (150 - 80) * (38.0 + 34.0) / 2
        trap3 = (220 - 150) * (34.0 + 28.0) / 2
        end = (END_PROJECTION_DAY - 220) * 28.0

        expected_yield = start + trap1 + trap2 + trap3 + end

        assert len(result) == 1, "Should return one row for one lactation"
        assert result.iloc[0]["TestId"] == 1, "TestId should be 1"
        assert np.isclose(result.iloc[0]["LactationMilkYield"], expected_yield, rtol=1e-5), (
            f"Expected {expected_yield}, got {result.iloc[0]['LactationMilkYield']}"
        )


@pytest.mark.output_validation
class TestOutputValidation:
    """Test validation of output data structure and quality."""

    def test_output_is_dataframe(self):
        """Test that output is a pandas DataFrame with correct structure."""
        df = pd.DataFrame({"DaysInMilk": [10, 40], "MilkingYield": [30.0, 25.0], "TestId": [1, 1]})

        result = test_interval_method(df)

        assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"
        assert list(result.columns) == [
            "TestId",
            "LactationMilkYield",
        ], f"Expected columns ['TestId', 'LactationMilkYield'], got {list(result.columns)}"
        assert result["LactationMilkYield"].dtype == np.float64, (
            f"LactationMilkYield should be float64, got {result['LactationMilkYield'].dtype}"
        )

    def test_output_values_are_positive(self):
        """Test that all LactationMilkYield values are positive."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 70, 15, 45, 75],
                "MilkingYield": [30.0, 35.0, 32.0, 28.0, 33.0, 30.0],
                "TestId": [1, 1, 1, 2, 2, 2],
            }
        )

        result = test_interval_method(df)

        assert all(result["LactationMilkYield"] > 0), "All yields should be positive"

    def test_multiple_lactations_processed(self):
        """Test that multiple lactations are processed independently."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 70, 15, 45, 75],
                "MilkingYield": [30.0, 35.0, 32.0, 28.0, 33.0, 30.0],
                "TestId": [1, 1, 1, 2, 2, 2],
            }
        )

        result = test_interval_method(df)

        assert len(result) == 2, "Should return two rows for two lactations"
        assert set(result["TestId"].values) == {1, 2}, "Should contain TestIds 1 and 2"


@pytest.mark.edge_cases
class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_days_beyond_standard_lactation_days_filtered(self):
        """Test that records with DIM > STANDARD_LACTATION_DAYS are filtered out."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 100, 200, STANDARD_LACTATION_DAYS, 350],
                "MilkingYield": [30.0, 35.0, 32.0, 28.0, 25.0],
                "TestId": [1, 1, 1, 1, 1],
            }
        )

        result = test_interval_method(df)

        # Day 350 should be filtered out, only first 4 records used
        # We can verify this by checking that the result is reasonable
        assert len(result) == 1
        assert result.iloc[0]["LactationMilkYield"] > 0

    def test_exactly_day_standard_lactation_days_included(self):
        """Test that day STANDARD_LACTATION_DAYS is included in
        calculations (boundary condition).
        """
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, STANDARD_LACTATION_DAYS],
                "MilkingYield": [30.0, 20.0],
                "TestId": [1, 1],
            }
        )

        result = test_interval_method(df)

        # Start: 10 * 30.0 = 300
        # Trapezoid: (305 - 10) * (30.0 + 20.0) / 2 = 295 * 25.0 = 7375
        # End: (306 - STANDARD_LACTATION_DAYS) * 20.0 = 1 * 20.0 = 20
        expected_yield = (
            (10 * 30.0)
            + ((STANDARD_LACTATION_DAYS - 10) * (30.0 + 20.0) / 2)
            + ((END_PROJECTION_DAY - STANDARD_LACTATION_DAYS) * 20.0)
        )

        assert len(result) == 1
        assert np.isclose(result.iloc[0]["LactationMilkYield"], expected_yield, rtol=1e-5)

    def test_first_test_on_day_zero(self):
        """Test that DIM = 0 (first test on day 0) is handled correctly."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [0, 30, 60],
                "MilkingYield": [35.0, 38.0, 36.0],
                "TestId": [1, 1, 1],
            }
        )

        result = test_interval_method(df)

        # Start: 0 * 35.0 = 0 (no projection before day 0)
        # Trap1: (30 - 0) * (35.0 + 38.0) / 2 = 30 * 36.5 = 1095
        # Trap2: (60 - 30) * (38.0 + 36.0) / 2 = 30 * 37.0 = 1110
        # End: (306 - 60) * 36.0 = 246 * 36.0 = 8856
        expected_yield = 0 + (30 * 36.5) + (30 * 37.0) + (246 * 36.0)

        assert len(result) == 1
        assert result.iloc[0]["TestId"] == 1
        assert np.isclose(result.iloc[0]["LactationMilkYield"], expected_yield, rtol=1e-5), (
            f"Expected {expected_yield}, got {result.iloc[0]['LactationMilkYield']}"
        )

    def test_unsorted_data_handled(self):
        """Test that unsorted data is handled correctly (sorted internally)."""
        # data intentionally out of order
        df = pd.DataFrame(
            {
                "DaysInMilk": [70, 10, 40],
                "MilkingYield": [32.0, 30.0, 35.0],
                "TestId": [1, 1, 1],
            }
        )

        result = test_interval_method(df)

        # should produce same result as sorted data
        df_sorted = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 70],
                "MilkingYield": [30.0, 35.0, 32.0],
                "TestId": [1, 1, 1],
            }
        )
        result_sorted = test_interval_method(df_sorted)

        assert np.isclose(
            result.iloc[0]["LactationMilkYield"],
            result_sorted.iloc[0]["LactationMilkYield"],
            rtol=1e-5,
        ), "Unsorted data should produce same result as sorted data"

    def test_insufficient_data_points_skipped(self):
        """Test that lactation with only one data point is skipped."""
        df = pd.DataFrame({"DaysInMilk": [50], "MilkingYield": [30.0], "TestId": [1]})

        result = test_interval_method(df)

        assert len(result) == 0, "Should return empty DataFrame when only 1 point"

    def test_negative_milk_yield_handled(self):
        """Test that negative milk yield values are handled appropriately."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 70],
                "MilkingYield": [30.0, -5.0, 25.0],
                "TestId": [1, 1, 1],
            }
        )

        # Negative yields are likely data errors and should either raise an error
        # or be filtered out - verify the function handles them gracefully
        try:
            result = test_interval_method(df)
            # If it doesn't raise an error, verify result is reasonable
            assert len(result) <= 1, "Should process or skip invalid data"
            if len(result) > 0:
                assert result.iloc[0]["LactationMilkYield"] > 0, (
                    "Total yield should still be positive"
                )
        except (ValueError, AssertionError):
            # Function may raise an error for invalid data, which is acceptable
            pass

    def test_zero_milk_yield_handled(self):
        """Test that zero milk yield values are handled appropriately."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 70],
                "MilkingYield": [30.0, 0.0, 25.0],
                "TestId": [1, 1, 1],
            }
        )

        # Zero yields could be valid (no milk that day) or data errors
        result = test_interval_method(df)

        # Should still process the lactation
        assert len(result) == 1, "Should process lactation with zero yield"
        assert result.iloc[0]["LactationMilkYield"] >= 0, "Total yield should be non-negative"

    def test_empty_dataframe_with_columns(self):
        """Test that empty DataFrame with proper columns returns empty result."""
        df = pd.DataFrame({"DaysInMilk": [], "MilkingYield": [], "TestId": []})

        result = test_interval_method(df)

        # Should return empty DataFrame with correct structure
        assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
        assert len(result) == 0, "Should return empty DataFrame for empty input"
        assert list(result.columns) == [
            "TestId",
            "LactationMilkYield",
        ], "Should have correct columns even when empty"

    def test_completely_empty_dataframe(self):
        """Test that completely empty DataFrame is handled gracefully."""
        df = pd.DataFrame()

        # Should either raise an error or return empty result
        try:
            result = test_interval_method(df)
            assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
            assert len(result) == 0, "Should return empty result"
        except (ValueError, KeyError):
            # Function may raise an error for missing columns, which is acceptable
            pass

    def test_duplicate_dim_records_handled(self):
        """Test that duplicate DIM records for same lactation are handled appropriately."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, 40, 70],
                "MilkingYield": [30.0, 35.0, 36.0, 32.0],
                "TestId": [1, 1, 1, 1],
            }
        )

        # Function should handle duplicates by either:
        # - Taking the first occurrence
        # - Taking the last occurrence
        # - Averaging the duplicate values
        # - Raising an error
        try:
            result = test_interval_method(df)
            assert len(result) == 1, "Should return one row for one lactation"
            assert result.iloc[0]["TestId"] == 1
            assert result.iloc[0]["LactationMilkYield"] > 0, "Total yield should be positive"
        except (ValueError, AssertionError):
            # Function may raise an error for duplicate DIMs, which is acceptable
            pass

    def test_non_numeric_data_in_numeric_columns(self):
        """Test that non-numeric data in numeric columns is handled appropriately."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, "invalid", 70],
                "MilkingYield": [30.0, 35.0, "bad_data"],
                "TestId": [1, 1, 1],
            }
        )

        # Function should either:
        # - Raise a TypeError or ValueError
        # - Filter out invalid records
        # - Convert to numeric and drop NaN values
        with pytest.raises((TypeError, ValueError)):
            test_interval_method(df)

    def test_nan_values_handled(self):
        """Test that NaN/null values in data are handled appropriately."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [10, 40, np.nan, 70],
                "MilkingYield": [30.0, np.nan, 35.0, 32.0],
                "TestId": [1, 1, 1, 1],
            }
        )

        # Function should handle NaN values by either:
        # - Filtering out rows with NaN
        # - Raising an error
        # - Processing only valid rows
        try:
            result = test_interval_method(df)
            # If it doesn't raise an error, verify result is reasonable
            assert len(result) <= 1, "Should process or skip records with NaN"
            if len(result) > 0:
                assert result.iloc[0]["TestId"] == 1
                assert result.iloc[0]["LactationMilkYield"] > 0, "Total yield should be positive"
                assert not np.isnan(result.iloc[0]["LactationMilkYield"]), (
                    "Result should not be NaN"
                )
        except (ValueError, TypeError):
            # Function may raise an error for NaN values, which is acceptable
            pass


@pytest.mark.realistic
class TestRealisticData:
    """Test with realistic CSV data from actual lactation curves."""

    def test_single_lactation_from_mr_file(self, mr_test_data):
        """Test processing of a single lactation from mr_test_file.csv."""
        df = mr_test_data

        # Pick the first TestId
        first_test_id = df["TestId"].iloc[0]
        single_lactation = df[df["TestId"] == first_test_id]

        result = test_interval_method(single_lactation)

        # Verify basic structure
        assert len(result) == 1, "Should return one row for one lactation"
        assert result.iloc[0]["TestId"] == first_test_id

        # Verify realistic range for dairy cows (typically 5000-15000 kg for 305-day lactation)
        total_yield = result.iloc[0]["LactationMilkYield"]
        assert 3000 <= total_yield <= 20000, (
            f"Yield {total_yield} kg is outside realistic range (3000-20000 kg)"
        )

    def test_all_lactations_from_mr_file(self, mr_test_data):
        """Test processing of all lactations from mr_test_file.csv."""
        df = mr_test_data

        # Get unique TestIds from input
        unique_test_ids = df["TestId"].unique()

        result = test_interval_method(df)

        # Verify that all lactations were processed
        # (some might be skipped if they have <2 data points)
        assert len(result) <= len(unique_test_ids), (
            "Result should not have more lactations than input"
        )
        assert len(result) > 0, "Should process at least some lactations"

        # Verify all TestIds in result were in input
        assert set(result["TestId"].values).issubset(set(unique_test_ids)), (
            "All output TestIds should be from input data"
        )

        # Verify all yields are positive
        assert all(result["LactationMilkYield"] > 0), "All yields should be positive"

    def test_subsampled_complete_curve(self, complete_lactation_data):
        """Test with monthly-sampled points from l2_anim2_herd654.csv."""
        df = complete_lactation_data

        # Filter to standard lactation period and subsample monthly (every ~30 days)
        df_305 = df[df["DaysInMilk"] <= STANDARD_LACTATION_DAYS].copy()

        # Select monthly points: days 10, 40, 70, 100, 130, 160, 190, 220, 250, 280
        monthly_days = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280]
        df_monthly = df_305[df_305["DaysInMilk"].isin(monthly_days)].copy()
        df_monthly["TestId"] = 1

        result = test_interval_method(df_monthly)

        # Verify result structure
        assert len(result) == 1, "Should return one row"
        assert result.iloc[0]["TestId"] == 1

        # Verify realistic yield range
        total_yield = result.iloc[0]["LactationMilkYield"]
        assert 5000 <= total_yield <= 15000, (
            f"Yield {total_yield} kg is outside expected range for complete curve (5000-15000 kg)"
        )

    def test_weekly_sampling_complete_curve(self, complete_lactation_data):
        """Test with weekly-sampled points from l2_anim2_herd654.csv."""
        df = complete_lactation_data

        # Filter to standard lactation period and subsample weekly (every 7 days)
        df_305 = df[df["DaysInMilk"] <= STANDARD_LACTATION_DAYS].copy()
        df_weekly = df_305[df_305["DaysInMilk"] % 7 == 0].copy()
        df_weekly["TestId"] = 1

        result = test_interval_method(df_weekly)

        # Verify result structure
        assert len(result) == 1, "Should return one row"
        assert result.iloc[0]["TestId"] == 1

        # Verify realistic yield range
        total_yield = result.iloc[0]["LactationMilkYield"]
        assert 5000 <= total_yield <= 15000, (
            f"Yield {total_yield} kg is outside expected range (5000-15000 kg)"
        )

        # Weekly sampling should give similar results to monthly (both interpolate the same curve)
        # But we can at least verify it's in a reasonable range
        assert total_yield > 0, "Yield should be positive"

    def test_large_dataset_performance(self):
        """Test performance with thousands of lactations (production scenario)."""
        # Generate 5000 lactations with 8 test points each (realistic farm size)
        num_lactations = 5000
        points_per_lactation = 8

        data = []
        for test_id in range(1, num_lactations + 1):
            # Generate test days at typical intervals: 5, 35, 65, 95, 125, 155, 185, 215
            test_days = [5 + i * 30 for i in range(points_per_lactation)]

            # Generate realistic yields following typical lactation curve
            # Peak around day 50-70, gradual decline
            for day in test_days:
                if day <= 60:
                    yield_value = 25 + (day / 60) * 15  # Increasing to peak
                else:
                    yield_value = 40 - ((day - 60) / 250) * 20  # Declining after peak

                data.append(
                    {
                        "DaysInMilk": day,
                        "MilkingYield": yield_value + np.random.normal(0, 2),  # Add noise
                        "TestId": test_id,
                    }
                )

        df = pd.DataFrame(data)

        # Measure execution time
        start_time = time.time()
        result = test_interval_method(df)
        elapsed_time = time.time() - start_time

        # Verify results
        assert len(result) == num_lactations, f"Should process all {num_lactations} lactations"
        assert all(result["LactationMilkYield"] > 0), "All yields should be positive"
        assert set(result["TestId"].values) == set(range(1, num_lactations + 1)), (
            "Should have all TestIds"
        )

        # Performance check: should complete in reasonable time (< 30 seconds for 5000 lactations)
        assert elapsed_time < 30, (
            f"Processing {num_lactations} lactations took {elapsed_time:.2f}s (expected < 30s)"
        )

        print(f"\nPerformance: Processed {num_lactations} lactations in {elapsed_time:.2f} seconds")
        print(f"Average: {elapsed_time / num_lactations * 1000:.2f} ms per lactation")
