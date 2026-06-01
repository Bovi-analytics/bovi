"""Test suite for best-prediction utilities and 305-day yield estimation.

This module validates core helpers and end-to-end prediction behavior for the
best-prediction implementation in ``lactationcurve.characteristics.best_predict``.

Test Categories:
    - TestUtilityFunctions: matrix-building, preprocessing, covariance helpers.
    - TestPredictionFunctions: single-lactation and multi-lactation prediction.
    - TestIntegration: real-file smoke and leave-one-id-out fit checks.

Usage:
    Run all tests::

        pytest tests/lactationcurve/test_best_predict.py -v

    Run specific marker::

        pytest tests/lactationcurve/test_best_predict.py -m utility -v
        pytest tests/lactationcurve/test_best_predict.py -m integration -v

Author:
    Meike van Leerdam and copilot
"""

import numpy as np
import pandas as pd
import pytest

from lactationcurve.characteristics.best_predict import (
    best_predict_method,
    best_predict_method_single_lac,
    build_covariance_matrix,
    pivot_milk_recordings_to_matrix,
    preprocess_measured_data,
)
from lactationcurve.preprocessing import standardize_lactation_columns


@pytest.fixture
def standard_curve() -> np.ndarray:
    """Return a simple flat standard curve used in deterministic unit tests."""
    return np.full(305, 10.0)


@pytest.fixture
def identity_covariance() -> np.ndarray:
    """Return identity covariance matrix used in deterministic unit tests."""
    return np.eye(305)


@pytest.mark.utility
class TestUtilityFunctions:
    """Test helper behavior used by best prediction."""

    def test_pivot_milk_recordings_to_matrix_enforces_fixed_305_grid(self):
        """Pivoted matrix should always use two dimensions and 305 columns."""
        df = pd.DataFrame(
            {
                "TestId": [1, 1, 2],
                "DaysInMilk": [1, 3, 1],
                "MilkingYield": [10.0, 12.0, 8.0],
            }
        )

        matrix = pivot_milk_recordings_to_matrix(df)

        assert matrix.shape == (2, 305)
        assert matrix[0, 0] == 10.0
        assert np.isnan(matrix[0, 1])
        assert matrix[0, 2] == 12.0
        assert matrix[1, 0] == 8.0

    def test_preprocess_measured_data_returns_305_series_for_valid_days(self, standard_curve):
        """Preprocess should return day-indexed 305-length deviation vector."""
        lactation = pd.DataFrame(
            {
                "DaysInMilk": [1, 10],
                "MilkingYield": [11.0, 8.0],
            }
        )

        corrected = preprocess_measured_data(lactation, standard_curve)

        assert len(corrected) == 305
        assert corrected.index.min() == 1
        assert corrected.index.max() == 305
        assert corrected.loc[1] == 1.0
        assert corrected.loc[10] == -2.0
        assert float(corrected.sum()) == -1.0

    def test_preprocess_measured_data_raises_for_out_of_range_days(self, standard_curve):
        """DIM above 305 should fail because indexing exceeds standard curve."""
        lactation = pd.DataFrame(
            {
                "DaysInMilk": [1, 306],
                "MilkingYield": [11.0, 20.0],
            }
        )

        with pytest.raises(IndexError):
            preprocess_measured_data(lactation, standard_curve)

    def test_build_covariance_matrix_matches_ar1_structure(self):
        """AR(1) covariance helper should match known small matrix values."""
        cov = build_covariance_matrix(rho=0.5, size=4)

        expected = np.array(
            [
                [1.0, 0.5, 0.25, 0.125],
                [0.5, 1.0, 0.5, 0.25],
                [0.25, 0.5, 1.0, 0.5],
                [0.125, 0.25, 0.5, 1.0],
            ]
        )

        assert np.allclose(cov, expected)
        assert np.allclose(cov, cov.T)
        assert np.allclose(np.diag(cov), np.ones(4))

    def test_standardize_lactation_columns_renames_and_filters_rows(self):
        """Standardization should accept aliased column names and filter DIM > 305."""
        df = pd.DataFrame(
            {
                "animalid": [7, 7, 7],
                "dim": [1, 305, 306],
                "milk_yield": [10.0, 11.0, 12.0],
            }
        )

        standardized = standardize_lactation_columns(df)

        assert list(standardized.columns) == ["TestId", "DaysInMilk", "MilkingYield"]
        assert len(standardized) == 2
        assert standardized["TestId"].tolist() == [7, 7]
        assert standardized["DaysInMilk"].tolist() == [1, 305]
        assert standardized["MilkingYield"].tolist() == [10.0, 11.0]

    def test_standardize_lactation_columns_accepts_human_readable_headers(self):
        """Standardization should tolerate common uploaded milk-recording headers."""
        df = pd.DataFrame(
            {
                "Cow ID": ["cow1", "cow1"],
                "Days in Milk": [10, 20],
                "Milk Yield (kg)": [25.5, 28.0],
            }
        )

        standardized = standardize_lactation_columns(df)

        assert list(standardized.columns) == ["TestId", "DaysInMilk", "MilkingYield"]
        assert standardized["TestId"].tolist() == ["cow1", "cow1"]
        assert standardized["DaysInMilk"].tolist() == [10, 20]
        assert standardized["MilkingYield"].tolist() == [25.5, 28.0]


class TestPredictionFunctions:
    """Test prediction behavior for single and multiple lactations."""

    def test_best_predict_method_single_lac_uses_cumulative_sum_adjustment(
        self, standard_curve, identity_covariance
    ):
        """One positive residual with identity covariance should shift total by 1."""
        lactation = pd.DataFrame(
            {
                "DaysInMilk": [1],
                "MilkingYield": [11.0],
            }
        )

        predicted = best_predict_method_single_lac(
            lactation,
            standard_curve,
            identity_covariance,
        )

        assert predicted == 3051.0

    def test_best_predict_method_single_lac_empty_returns_standard_curve_sum(
        self, standard_curve, identity_covariance
    ):
        """If no valid DIM in 1..305 exists, prediction should equal baseline sum."""
        lactation = pd.DataFrame(
            {
                "DaysInMilk": [0, 400],
                "MilkingYield": [11.0, 12.0],
            }
        )

        predicted = best_predict_method_single_lac(
            lactation,
            standard_curve,
            identity_covariance,
        )

        assert predicted == pytest.approx(float(np.sum(standard_curve)))

    def test_best_predict_method_single_lac_duplicate_days_keep_last(
        self, standard_curve, identity_covariance
    ):
        """Duplicate DIM records should use the last value before prediction."""
        lactation = pd.DataFrame(
            {
                "DaysInMilk": [1, 1],
                "MilkingYield": [9.0, 11.0],
            }
        )

        predicted = best_predict_method_single_lac(
            lactation,
            standard_curve,
            identity_covariance,
        )

        assert predicted == 3051.0

    def test_best_predict_method_does_not_mutate_input_dataframe_without_testid(
        self, standard_curve, identity_covariance
    ):
        """Main API should not mutate caller dataframe when creating TestId."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [1, 2],
                "MilkingYield": [11.0, 12.0],
            }
        )
        original_columns = list(df.columns)

        result = best_predict_method(
            df,
            standard_lc=standard_curve,
            covariance_matrix=identity_covariance,
        )

        assert list(df.columns) == original_columns
        assert list(result.columns) == ["TestId", "LactationMilkYield"]
        assert len(result) == 1

    def test_best_predict_method_accepts_alias_column_names(
        self, standard_curve, identity_covariance
    ):
        """Main API should standardize aliased input columns before prediction."""
        df = pd.DataFrame(
            {
                "animalid": [5, 5],
                "dim": [1, 2],
                "milk_yield": [11.0, 12.0],
                "ignored": ["x", "y"],
            }
        )

        result = best_predict_method(
            df,
            standard_lc=standard_curve,
            covariance_matrix=identity_covariance,
        )

        assert list(result.columns) == ["TestId", "LactationMilkYield"]
        assert result["TestId"].tolist() == [5]
        assert result["LactationMilkYield"].iloc[0] == 3053.0

    def test_best_predict_method_uses_defaults_when_no_curve_or_covariance_passed(self):
        """Main API should run with module defaults when optional args are omitted."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [1, 2],
                "MilkingYield": [11.0, 12.0],
            }
        )

        result = best_predict_method(df)

        assert list(result.columns) == ["TestId", "LactationMilkYield"]
        assert len(result) == 1
        assert np.isfinite(result["LactationMilkYield"]).all()

    def test_best_predict_method_fit_from_reference_requires_reference_df(self):
        """Fit-from-reference mode should fail fast when no reference dataframe is supplied."""
        df = pd.DataFrame(
            {
                "DaysInMilk": [1, 2],
                "MilkingYield": [11.0, 12.0],
            }
        )

        with pytest.raises(
            ValueError, match="Provide reference_df to fit your own standard lactation curve."
        ):
            best_predict_method(df, fit_standard_lc_from_data=True)


class TestIntegration:
    """Integration tests using real fixture files and package-provided matrices."""

    def test_best_predict_method_real_file_smoke_test(self, test_data_dir):
        """Real-file prediction with provided covariance should be finite and stable."""
        full_lac_single = pd.read_csv(test_data_dir / "l2_anim2_herd654.csv")
        full_lac_single = full_lac_single.rename(columns={"TestDayMilkYield": "MilkingYield"})

        result = best_predict_method(full_lac_single)

        assert not result.empty
        assert "LactationMilkYield" in result.columns
        assert np.isfinite(result["LactationMilkYield"]).all()
        assert result["LactationMilkYield"].iloc[0] == pytest.approx(10795.472636)

    def test_best_predict_method_fitted_covariance_returns_target_lactation(
        self, reference_lactations: pd.DataFrame
    ):
        """Fit-from-reference mode should fit covariance without replacing target data."""
        test_day_df = reference_lactations
        training_data = test_day_df.loc[test_day_df["TestId"] != 1].copy()
        target_data = test_day_df.loc[test_day_df["TestId"] == 1].copy()

        result_cov = best_predict_method(
            target_data,
            days_in_milk_col="DaysInMilk",
            milking_yield_col="MilkingYield",
            fit_standard_lc_from_data=True,
            reference_df=training_data,
        )

        assert result_cov["TestId"].tolist() == [1]
        assert np.isfinite(float(result_cov["LactationMilkYield"].iloc[0]))
