"""
Test suite for lactation curve fitting models.

This module provides comprehensive tests for 14 lactation curve models including
model output validation, parameter estimation, edge case handling, and Bayesian
fitting with the MilkBot API.

Models Tested:
    1. MilkBot (Bayesian and frequentist)
    2. Wood
    3. Wilmink
    4. Ali-Schaeffer
    5. Fischer
    6. Brody
    7. Sikka
    8. Nelder
    9. Dhanoa
    10. Emmans
    11. Hayashi
    12. Rook
    13. Dijkstra
    14. Prasad

Test Categories:
    - TestModelFunctions: Basic model output validity (14 models)
    - TestParameterFitting: Parameter recovery for fitted models (5 models)
    - TestEdgeCases: Invalid inputs and boundary conditions
    - TestBayesianFitting: MilkBot Bayesian API integration

Usage:
    Run all tests::

        pytest test_lactation_curve_fitting.py -v

    Run specific marker::

        pytest test_lactation_curve_fitting.py -m models -v
        pytest test_lactation_curve_fitting.py -m fitting -v
        pytest test_lactation_curve_fitting.py -m edge_cases -v
        pytest test_lactation_curve_fitting.py -m bayesian -v

Authors:
    Lucia Trapanese, Judith Osei-Tete
"""

import os

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

from lactationcurve.fitting import (
    ali_schaeffer_model,
    bayesian_fit_milkbot_single_lactation,
    brody_model,
    build_prior,
    dhanoa_model,
    dijkstra_model,
    emmans_model,
    fischer_model,
    fit_lactation_curve,
    get_chen_priors,
    get_lc_parameters,
    get_lc_parameters_least_squares,
    hayashi_model,
    milkbot_model,
    nelder_model,
    prasad_model,
    rook_model,
    sikka_model,
    wilmink_model,
    wood_model,
)


@pytest.fixture
def milkbot_api_key() -> str:
    """Return the MilkBot API key from environment."""
    load_dotenv(find_dotenv())
    key = os.getenv("milkbot_key")
    if not key:
        raise ValueError("milkbot_key not found in environment. Check your .env file.")
    return key


@pytest.mark.utility
class TestUtilityFunctions:
    """Test utility and helper functions for lactation curve fitting."""

    def test_build_prior_structure(self):
        """Should return a dictionary with correct structure and value types."""
        # Example values for prior parameters
        scale_mean = 40
        scale_sd = 5
        ramp_mean = 20
        ramp_sd = 2
        decay_mean = 0.002
        decay_sd = 0.001
        offset_mean = 0
        offset_sd = 0.01
        prior = build_prior(
            scale_mean,
            scale_sd,
            ramp_mean,
            ramp_sd,
            decay_mean,
            decay_sd,
            offset_mean,
            offset_sd,
        )
        assert isinstance(prior, dict), "build_prior should return a dictionary"
        for key in ["scale", "ramp", "decay", "offset"]:
            assert key in prior, f"Missing key: {key} in prior"
            value = prior[key]
            assert isinstance(value, dict), f"Prior value for {key} should be a dict"
            assert "mean" in value and "sd" in value, (
                f"Prior {key} should have 'mean' and 'sd' keys"
            )
            assert isinstance(value["mean"], (int, float)), f"Prior {key}['mean'] should be numeric"
            assert isinstance(value["sd"], (int, float)), f"Prior {key}['sd'] should be numeric"
        # Optionally check for seMilk if present
        if "seMilk" in prior:
            assert isinstance(prior["seMilk"], (int, float)), "seMilk should be numeric"


@pytest.fixture
def sample_dim():
    """Standard DIM array for testing (1-199 days).

    Returns:
        np.ndarray: Days in milk from 1 to 199.
    """
    return np.arange(1, 200)


@pytest.fixture
def short_dim():
    """Short DIM array for quick tests.

    Returns:
        np.ndarray: Days in milk from 1 to 9.
    """
    return np.arange(1, 10)


@pytest.fixture
def sample_lactation_data():
    """Realistic lactation curve data for testing.

    Returns:
        tuple: A tuple containing:
            - list of int: DIM values (10 time points)
            - list of float: Milk yield values (kg/day)
    """
    return [1, 5, 10, 20, 50, 100, 150, 200, 250, 300], [
        10,
        12,
        15,
        18,
        22,
        25,
        23,
        20,
        18,
        15,
    ]


@pytest.mark.models
class TestModelFunctions:
    """Test basic model output validity for all 14 lactation curve models.

    This test class verifies that all lactation curve model functions produce
    valid numerical output with correct shape, dtype, and finite values.

    Attributes:
        Tests cover: MilkBot, Wood, Wilmink, Ali-Schaeffer, Fischer, Brody,
        Sikka, Nelder, Dhanoa, Emmans, Hayashi, Rook, Dijkstra, Prasad.
    """

    @pytest.mark.parametrize(
        "model, params",
        [
            (milkbot_model, (30, 50, 10, 0.01)),
            (wood_model, (20, 0.2, 0.01)),
            (wilmink_model, (30, 0.05, 5)),
            (ali_schaeffer_model, (10, 5, -2, 1, 0.5)),
            (fischer_model, (25, 0.01, 0.01)),
            (brody_model, (30, 0.01)),
            (sikka_model, (20, 0.01, 0.0001)),
            (nelder_model, (1, 0.1, 0.01)),
            (dhanoa_model, (10, 0.5, 0.01)),
            (emmans_model, (20, 0.1, 0.01, 2)),
            (hayashi_model, (2, 30, 10, 0.5)),
            (rook_model, (20, 1, 1, 0.01)),
            (dijkstra_model, (30, 0.1, 0.01, 0.01)),
            (prasad_model, (10, 0.1, 0.001, 5)),
        ],
    )
    def test_models_produce_valid_output(self, model, params):
        """Test that each model produces a valid numpy array with finite values."""
        t = np.array([1, 10, 100, 150, 200, 300])
        y = model(t, *params)
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))

    def test_model_output_shape_matches_input(self):
        """Should produce output array with same shape as input DIM array."""
        t = np.array([1, 10, 100, 150, 200, 300])
        y = wood_model(t, 30, 0.2, 0.003)
        assert isinstance(y, np.ndarray)
        assert y.shape == t.shape, f"Output shape {y.shape} should match input shape {t.shape}"

    def test_model_output_dtype_is_float(self):
        """Should produce output array with float dtype."""
        t = np.array([1, 10, 100, 150, 200, 300])
        y = wood_model(t, 30, 0.2, 0.003)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(y.dtype, np.floating), f"Output dtype should be float, got {y.dtype}"

    def test_model_output_length_correct(self):
        """Should produce output array with correct length."""
        t = np.arange(1, 200)
        y = wood_model(t, 30, 0.2, 0.003)
        assert isinstance(y, np.ndarray)
        assert len(y) == len(t), f"Output length {len(y)} should match input length {len(t)}"
        assert len(y) == 199, "Should have 199 values for DIM 1-199"

    def test_fitted_parameters_have_correct_length(self, sample_dim):
        """Should return correct number of parameters for each model."""
        y = wood_model(sample_dim, 30, 0.2, 0.003)
        params = get_lc_parameters(sample_dim, y, model="wood")
        assert len(params) == 3, "Wood model should return 3 parameters"

        y_wilmink = wilmink_model(sample_dim, 30, 0.1, 10, -0.05)
        params_wilmink = get_lc_parameters(sample_dim, y_wilmink, model="wilmink")
        assert len(params_wilmink) == 4, "Wilmink model should return 4 parameters"

        y_milkbot = milkbot_model(sample_dim, 40, 30, 10, 0.005)
        params_milkbot = get_lc_parameters(sample_dim, y_milkbot, model="milkbot")
        assert len(params_milkbot) == 4, "MilkBot model should return 4 parameters"

    def test_fit_lactation_curve_output_length(self, sample_lactation_data, milkbot_api_key):
        """Should produce curve values covering full lactation period."""
        dim, milkrecordings = sample_lactation_data
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            parity=2,
            breed="H",
            continent="USA",
        )
        # Should generate values for at least 305 days (standard lactation)
        assert len(y) >= 305, f"Should generate at least 305 days of curve, got {len(y)}"
        assert y.dtype == np.float64, f"Output should be float64, got {y.dtype}"


@pytest.mark.fitting
class TestParameterFitting:
    """Test parameter estimation for models with frequentist fitting support.

    This test class validates parameter recovery by fitting synthetic data
    generated from known parameters and verifying the estimated parameters
    match the true values within acceptable tolerance.

    Models tested:
        - Wood (3 parameters)
        - Wilmink (4 parameters)
        - Ali-Schaeffer (5 parameters)
        - Fischer (3 parameters)
        - MilkBot (4 parameters)
        - MilkBot least squares (alternative fitting method)
    """

    def test_wood_recovers_parameters(self, sample_dim):
        """Wood model should recover known parameters from synthetic data."""
        true_params = (30, 0.2, 0.003)
        y = wood_model(sample_dim, *true_params)
        est_params = get_lc_parameters(sample_dim, y, model="wood")
        assert np.allclose(est_params, true_params, rtol=0.2)

    def test_wilmink_recovers_parameters(self, sample_dim):
        """Wilmink model should recover known parameters from synthetic data."""
        true_params = (30, 0.1, 10, -0.05)
        y = wilmink_model(sample_dim, *true_params)
        est_params = get_lc_parameters(sample_dim, y, model="wilmink")
        assert np.allclose(est_params[:3], true_params[:3], rtol=0.2)

    def test_ali_schaeffer_recovers_parameters(self, sample_dim):
        """Ali-Schaeffer model should recover known parameters from synthetic data."""
        true_params = (25, 5, -2, 1, 0.5)
        y = ali_schaeffer_model(sample_dim, *true_params)
        est_params = get_lc_parameters(sample_dim, y, model="ali_schaeffer")
        assert np.allclose(est_params, true_params, rtol=0.3)

    def test_fischer_recovers_parameters(self, sample_dim):
        """Fischer model should recover known parameters from synthetic data."""
        true_params = (30, 0.01, 0.01)
        y = fischer_model(sample_dim, *true_params)
        est_params = get_lc_parameters(sample_dim, y, model="fischer")
        assert np.allclose(est_params, true_params, rtol=0.3)

    def test_milkbot_recovers_parameters(self, sample_dim):
        """MilkBot model should recover known parameters from synthetic data."""
        true_params = (40, 30, 10, 0.005)
        y = milkbot_model(sample_dim, *true_params)
        est_params = get_lc_parameters(sample_dim, y, model="milkbot")
        assert np.allclose(est_params, true_params, rtol=0.5)

    def test_milkbot_least_squares_produces_valid_parameters(self, sample_dim):
        """Least squares fitting should return valid MilkBot parameters."""
        # Generate synthetic data
        true_params = (40, 30, 10, 0.005)
        y = milkbot_model(sample_dim, *true_params)

        # Fit using least squares method
        est_params = get_lc_parameters_least_squares(sample_dim, y, model="milkbot")

        # Should return 4 parameters
        assert len(est_params) == 4, "MilkBot least squares should return 4 parameters"

        # All parameters should be finite
        assert np.all(np.isfinite(est_params)), "All parameters should be finite"

        # Parameters should be in reasonable ranges
        a, b, c, d = est_params
        assert a > 0, "Scale parameter 'a' should be positive"
        assert b > 0, "Ramp parameter 'b' should be positive"
        assert -600 < c < 300, "Offset parameter 'c' should be in reasonable range"
        assert d > 0, "Decay parameter 'd' should be positive"

    def test_milkbot_least_squares_recovers_parameters(self, sample_dim):
        """Least squares should recover known MilkBot parameters from clean data."""
        true_params = (40, 30, 10, 0.005)
        y = milkbot_model(sample_dim, *true_params)

        # Fit using least squares
        est_params = get_lc_parameters_least_squares(sample_dim, y, model="milkbot")

        # Should recover parameters within tolerance
        assert np.allclose(est_params, true_params, rtol=0.5), (
            f"Estimated {est_params} should be close to true {true_params}"
        )

    def test_milkbot_least_squares_handles_noisy_data(self, sample_dim):
        """Least squares should handle noisy data and produce finite parameters."""
        # Generate noisy synthetic data
        true_params = (40, 30, 10, 0.005)
        y_clean = milkbot_model(sample_dim, *true_params)
        assert isinstance(y_clean, np.ndarray)
        y_noisy = y_clean + np.random.normal(0, 1.0, size=y_clean.shape)

        # Fit noisy data
        est_params = get_lc_parameters_least_squares(sample_dim, y_noisy, model="milkbot")

        # Should still produce finite parameters
        assert np.all(np.isfinite(est_params)), "Should handle noisy data"
        assert len(est_params) == 4, "Should return 4 parameters"


@pytest.mark.edge_cases
class TestEdgeCases:
    """Test handling of invalid inputs and boundary conditions.

    This test class ensures robust error handling and validates that the
    fitting functions gracefully handle edge cases including:

    Edge Cases:
        - Empty input arrays
        - Insufficient data points
        - Missing values (NaN)
        - Negative milk yields
        - Unordered DIM values
        - Noisy data
        - Invalid model/breed/continent names
        - Invalid fitting methods
        - Non-numeric input
        - Case-insensitive model name handling
    """

    @pytest.mark.parametrize(
        "model_name, expected_type",
        [
            ("wood", np.ndarray),
            ("wilmink", np.ndarray),
            ("ali_schaeffer", np.ndarray),
            ("fischer", np.ndarray),
            ("milkbot", np.ndarray),
        ],
    )
    def test_fit_lactation_curve_frequentist_models(self, sample_dim, model_name, expected_type):
        """Should cover all frequentist model branches in fit_lactation_curve."""
        milkrecordings = np.random.uniform(20, 40, size=len(sample_dim))
        y = fit_lactation_curve(sample_dim, milkrecordings, model=model_name, fitting="frequentist")
        assert isinstance(y, expected_type)
        assert len(y) >= len(sample_dim)

    @pytest.mark.parametrize(
        "model_name, expected_type",
        [
            ("wood", np.ndarray),
            ("wilmink", np.ndarray),
            ("ali_schaeffer", np.ndarray),
            ("fischer", np.ndarray),
            ("milkbot", np.ndarray),
        ],
    )
    def test_fit_lactation_curve_frequentist_max_dim_greater_than_305(
        self, model_name, expected_type
    ):
        """Should cover branch where max(dim) > 305 in fit_lactation_curve (frequentist)."""
        dim = np.arange(1, 400)  # 399 days
        milkrecordings = np.random.uniform(20, 40, size=len(dim))
        y = fit_lactation_curve(dim, milkrecordings, model=model_name, fitting="frequentist")
        assert isinstance(y, expected_type)
        assert len(y) == 399, f"Output length should match max(dim), got {len(y)}"

    def test_fit_lactation_curve_frequentist_unknown_model_raises(self, sample_dim):
        """Should raise Exception for unknown frequentist model."""
        milkrecordings = np.random.uniform(20, 40, size=len(sample_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(
                sample_dim, milkrecordings, model="notamodel", fitting="frequentist"
            )
        assert "Unknown model" in str(excinfo.value)

    def test_fit_lactation_curve_bayesian_milkbot(
        self, sample_lactation_data, milkbot_api_key, monkeypatch
    ):
        """Should cover bayesian branch for milkbot model."""
        dim, milkrecordings = sample_lactation_data

        def mock_post(url, headers=None, json=None):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"fittedParams": {"scale": 1, "ramp": 2, "decay": 3, "offset": 4}}

            return MockResponse()

        monkeypatch.setattr("requests.post", mock_post)
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
        )
        assert isinstance(y, np.ndarray)
        assert len(y) >= len(dim)

    def test_fit_lactation_curve_bayesian_max_dim_greater_than_305(
        self, milkbot_api_key, monkeypatch
    ):
        """Should cover branch where max(dim) > 305 in fit_lactation_curve (bayesian)."""
        dim = np.arange(1, 400)  # 399 days
        milkrecordings = np.random.uniform(20, 40, size=len(dim))

        def mock_post(url, headers=None, json=None):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"fittedParams": {"scale": 1, "ramp": 2, "decay": 3, "offset": 4}}

            return MockResponse()

        monkeypatch.setattr("requests.post", mock_post)
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
        )
        assert isinstance(y, np.ndarray)
        assert len(y) == 399, f"Output length should match max(dim), got {len(y)}"

    def test_fit_lactation_curve_bayesian_missing_key_raises(self, sample_lactation_data):
        """Should raise Exception if key is missing for bayesian fitting."""
        dim, milkrecordings = sample_lactation_data
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(dim, milkrecordings, model="milkbot", fitting="bayesian", key=None)
        assert "Key needed to use Bayesian fitting engine milkbot" in str(excinfo.value)

    def test_fit_lactation_curve_bayesian_non_milkbot_raises(
        self, sample_lactation_data, milkbot_api_key
    ):
        """Should raise Exception for bayesian fitting with non-milkbot model."""
        dim, milkrecordings = sample_lactation_data
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(
                dim,
                milkrecordings,
                model="wood",
                fitting="bayesian",
                key=milkbot_api_key,
            )
        assert "Bayesian fitting is currently only implemented for milkbot models" in str(
            excinfo.value
        )

    def test_empty_input_raises_value_error(self):
        """Should raise ValueError for empty input arrays."""
        with pytest.raises(ValueError):
            get_lc_parameters([], [], model="wilmink")

    def test_insufficient_points_raises_value_error(self):
        """Should raise ValueError when insufficient points for parameter estimation."""
        x = [1]
        y = [10]
        with pytest.raises(ValueError):
            get_lc_parameters(x, y, model="Milkbot")

    def test_handles_unordered_dim(self):
        """Should handle unordered DIM input and produce finite parameters."""
        x = [100, 1, 50, 10]
        y = [10, 30, 20, 15]
        est_params = get_lc_parameters(x, y, model="milkbot")
        assert np.all(np.isfinite(est_params))

    def test_handles_negative_milk_values(self):
        """Should handle negative milk values and produce finite parameters."""
        x = np.arange(1, 10)
        y = np.array([-1, 0, 5, 10, -3, 7, 8, 9, 0])
        est_params = get_lc_parameters(x, y, model="wood")
        assert np.all(np.isfinite(est_params))

    def test_fitting_with_noise(self, sample_dim):
        """Should handle noisy data and produce finite parameters."""
        y = wood_model(sample_dim, 30, 0.2, 0.003)
        assert isinstance(y, np.ndarray)
        y_noisy = y + np.random.normal(0, 0.5, size=y.shape)
        est_params = get_lc_parameters(sample_dim, y_noisy, model="MILKBOT")
        assert np.all(np.isfinite(est_params))

    def test_missing_yields_nan_dropped(self):
        """Should drop NaN values in milk yield and continue fitting."""
        x = [1.0, 10.0, 15.0, 20.0]
        y = [10.0, 15.0, np.nan, 25.0]
        result = get_lc_parameters(x, y)
        assert isinstance(result, (np.ndarray, tuple))
        assert len(result) > 0
        assert np.all(np.isfinite(result))

    def test_missing_dim_nan_dropped(self):
        """Should drop NaN values in DIM and continue fitting."""
        x = [1.0, np.nan, 15.0, 20.0]
        y = [10.0, 15.0, 20.0, 25.0]
        result = get_lc_parameters(x, y)
        assert isinstance(result, (np.ndarray, tuple))
        assert len(result) > 0
        assert np.all(np.isfinite(result))

    def test_all_values_nan_raises(self):
        """Should raise ValueError when all values are NaN."""
        x = [np.nan, np.nan]
        y = [np.nan, np.nan]
        with pytest.raises(
            ValueError,
            match="At least two non missing points are required to fit a lactation curve",
        ):
            get_lc_parameters(x, y)

    def test_single_valid_point_raises(self):
        """Should raise ValueError when only one valid point remains after dropping NaN."""
        x = [1.0, np.nan, np.nan]
        y = [10.0, np.nan, np.nan]
        with pytest.raises(
            ValueError,
            match="At least two non missing points are required to fit a lactation curve",
        ):
            get_lc_parameters(x, y)

    def test_invalid_model_name_raises(self, short_dim):
        """Should raise exception for invalid model name."""
        milkrecordings = np.random.uniform(20, 40, size=len(short_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(short_dim, milkrecordings, model="nomodel")
        assert "Unknown model" in str(excinfo.value)

    def test_invalid_breed_raises(self, sample_dim, milkbot_api_key):
        """Should raise exception for invalid breed in Bayesian fitting."""
        milkrecordings = np.random.uniform(20, 40, size=len(sample_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(
                sample_dim,
                milkrecordings,
                model="milkbot",
                fitting="bayesian",
                key=milkbot_api_key,
                breed="W",
            )
        assert "Breed must be either Holstein = 'H' or Jersey 'J'" in str(excinfo.value)

    def test_invalid_continent_raises(self, short_dim, milkbot_api_key):
        """Should raise exception for invalid continent in Bayesian
        fitting (only 'USA' and 'EU' allowed).
        """
        milkrecordings = np.random.uniform(20, 40, size=len(short_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(
                short_dim,
                milkrecordings,
                model="milkbot",
                fitting="bayesian",
                key=milkbot_api_key,
                continent="EW",
            )
        assert "continent must be 'USA' or 'EU'" in str(excinfo.value)

    def test_invalid_fitting_method_raises(self, short_dim):
        """Should raise exception for invalid fitting method."""
        milk = np.random.uniform(20, 40, size=len(short_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(short_dim, milk, model="wood", fitting="")
        assert "Fitting method must be either frequentist or bayesian" in str(excinfo.value)

    def test_non_numeric_input_raises(self):
        """Should raise ValueError for non-numeric input."""
        dim = ["a", "b", "c"]
        milkrecordings = ["x", "y", "z"]
        with pytest.raises(ValueError):
            fit_lactation_curve(dim, milkrecordings, model="wood")

    def test_bayesian_fitting_non_milkbot_raises(self, short_dim):
        """Should raise exception when requesting Bayesian fitting for non-MilkBot models."""
        milk = np.random.uniform(20, 40, size=len(short_dim))
        with pytest.raises(Exception) as excinfo:
            fit_lactation_curve(short_dim, milk, model="wood", fitting="bayesian")
        assert "Bayesian fitting is currently only implemented for milkbot models" in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "wood",
            "Wood",
            "WOOD",
            "wOoD",
            "milkbot",
            "MilkBot",
            "MILKBOT",
            "wilmink",
            "WILMINK",
        ],
    )
    def test_model_name_case_insensitive(self, sample_dim, model_name):
        """Should handle model names in any case (lowercase, uppercase, mixed)."""
        y = wood_model(sample_dim, 30, 0.2, 0.003)
        # Should not raise an error for case variations
        result = get_lc_parameters(sample_dim, y, model=model_name)
        assert isinstance(result, (np.ndarray, tuple))
        assert len(result) > 0
        assert np.all(np.isfinite(result))


@pytest.mark.bayesian
class TestBayesianFitting:
    """Test MilkBot Bayesian API integration and parameter estimation.

    This test class validates the MilkBot Bayesian fitting functionality
    through external API integration, testing various parameter combinations
    and prior distributions.

    Tests include:
        - API response structure validation
        - Parameter estimation with different parities (1, 2, 3+)
        - Breed variations (Holstein='H', Jersey='J')
        - Continental priors (USA, EU)
        - Chen priors structure and consistency
        - Edge cases (unordered DIM, minimal points, invalid keys)

    Note:
        Requires valid MilkBot API key from key_milkbot module.
    """

    def test_bayesian_fit_with_chen_priors(
        self, sample_lactation_data, milkbot_api_key, monkeypatch
    ):
        """Should use Chen priors when custom_priors='CHEN'."""
        dim, milkrecordings = sample_lactation_data
        called = {}

        def mock_post(url, headers=None, json=None):
            assert isinstance(json, dict)
            called["priors"] = json.get("priors", None)

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"fittedParams": {"scale": 1, "ramp": 2, "decay": 3, "offset": 4}}

            return MockResponse()

        monkeypatch.setattr("requests.post", mock_post)
        fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            custom_priors="CHEN",
        )
        assert called["priors"] is not None, "Chen priors should be included in payload"
        assert isinstance(called["priors"], dict)
        assert "scale" in called["priors"]

    def test_bayesian_fit_with_custom_dict_priors(
        self, sample_lactation_data, milkbot_api_key, monkeypatch
    ):
        """Should use custom dict priors when provided."""
        dim, milkrecordings = sample_lactation_data
        custom = build_prior(1, 2, 3, 4, 5, 6, 7, 8)
        called = {}

        def mock_post(url, headers=None, json=None):
            assert isinstance(json, dict)
            called["priors"] = json.get("priors", None)

            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"fittedParams": {"scale": 1, "ramp": 2, "decay": 3, "offset": 4}}

            return MockResponse()

        monkeypatch.setattr("requests.post", mock_post)
        fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            custom_priors=custom,
        )
        assert called["priors"] == custom, "Custom priors dict should be included in payload"

    def test_fit_lactation_curve_milkbot_produces_valid_output(
        self, sample_lactation_data, milkbot_api_key
    ):
        """Should produce valid numpy array for full lactation curve fitting."""
        dim, milkrecordings = sample_lactation_data
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            parity=2,
            breed="H",
            continent="USA",
        )
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))
        assert len(y) >= max(dim)

    def test_bayesian_returns_valid_structure(self, milkbot_api_key):
        """Should return dict with required parameter keys."""
        dim = [1, 5, 10, 20, 50]
        milkrecordings = [10, 12, 15, 18, 22]
        res = bayesian_fit_milkbot_single_lactation(dim, milkrecordings, milkbot_api_key)
        assert isinstance(res, dict)
        for key in ["scale", "ramp", "decay", "offset", "nPoints"]:
            assert key in res
            assert res[key] is not None

    def test_bayesian_handles_unordered_dim(self, milkbot_api_key):
        """Should handle unordered DIM input for Bayesian fitting."""
        dim = [50, 1, 20, 5, 10]
        milkrecordings = [22, 10, 18, 12, 15]
        res = bayesian_fit_milkbot_single_lactation(dim, milkrecordings, milkbot_api_key)
        for key in ["scale", "ramp", "decay", "offset"]:
            assert isinstance(res[key], (float, int))

    def test_bayesian_minimal_points(self, milkbot_api_key):
        """Should work with minimal number of data points."""
        dim = [1, 5, 13, 67]
        milkrecordings = [10, 12, 15, 30]
        res = bayesian_fit_milkbot_single_lactation(dim, milkrecordings, milkbot_api_key)
        assert isinstance(res["scale"], float)

    def test_bayesian_invalid_key_raises(self):
        """Should raise exception for invalid API key."""
        dim = [1, 5, 13, 67]
        milkrecordings = [10, 12, 15, 30]
        with pytest.raises(Exception):
            fit_lactation_curve(
                dim,
                milkrecordings,
                model="milkbot",
                fitting="bayesian",
                key="INCORRECT_KEY",
            )

    def test_get_priors_returns_valid_structure(self):
        """Should return dict structure for Chen priors."""
        priors = get_chen_priors(1)
        assert isinstance(priors, dict)

    @pytest.mark.parametrize("parity", [1, 2, 3])
    def test_bayesian_different_parities(self, milkbot_api_key, parity):
        """Should handle different parity values (1, 2, 3+)."""
        dim = [1, 5, 10, 20, 50, 100, 150, 200]
        milkrecordings = [10, 12, 15, 18, 22, 25, 23, 20]
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            parity=parity,
            breed="H",
            continent="USA",
        )
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))
        assert len(y) >= max(dim)

    @pytest.mark.parametrize("breed", ["H", "J"])
    def test_bayesian_different_breeds(self, milkbot_api_key, breed):
        """Should handle different breeds (Holstein='H', Jersey='J')."""
        dim = [1, 5, 10, 20, 50, 100, 150, 200]
        milkrecordings = [10, 12, 15, 18, 22, 25, 23, 20]
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            parity=2,
            breed=breed,
            continent="USA",
        )
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))
        assert len(y) >= max(dim)

    @pytest.mark.parametrize("continent", ["USA", "EU"])
    def test_bayesian_different_continents(self, milkbot_api_key, continent):
        """Should handle different continental priors (USA, EU)."""
        dim = [1, 5, 10, 20, 50, 100, 150, 200]
        milkrecordings = [10, 12, 15, 18, 22, 25, 23, 20]
        y = fit_lactation_curve(
            dim,
            milkrecordings,
            model="milkbot",
            fitting="bayesian",
            key=milkbot_api_key,
            parity=2,
            breed="H",
            continent=continent,
        )
        assert isinstance(y, np.ndarray)
        assert np.all(np.isfinite(y))
        assert len(y) >= max(dim)

    @pytest.mark.parametrize("parity", [1, 2, 3])
    def test_get_chen_priors_all_parities(self, parity):
        """Should return valid priors structure for all parity values."""
        priors = get_chen_priors(parity)
        assert isinstance(priors, dict)
        assert len(priors) > 0
        # Verify priors contain expected parameter keys
        expected_keys = {"scale", "ramp", "decay", "offset"}
        assert any(key in priors for key in expected_keys), (
            f"Priors should contain at least one of {expected_keys}"
        )

    def test_get_chen_priors_detailed_structure(self):
        """Should return priors with correct structure and value types."""
        priors = get_chen_priors(2)

        # Should be a dictionary
        assert isinstance(priors, dict), "Priors should be a dictionary"

        # Should not be empty
        assert len(priors) > 0, "Priors dictionary should not be empty"

        # Main parameters should have 'mean' and 'sd' keys
        main_params = ["scale", "ramp", "decay", "offset"]
        for param in main_params:
            if param in priors:
                value = priors[param]
                assert isinstance(value, dict), (
                    f"Prior value for {param} should be a dict, got {type(value)}"
                )
                assert "mean" in value, f"Prior {param} should have 'mean' key"
                assert "sd" in value, f"Prior {param} should have 'sd' key"
                assert isinstance(value["mean"], (int, float, np.number)), (
                    f"Prior {param}['mean'] should be numeric, got {type(value['mean'])}"
                )
                assert isinstance(value["sd"], (int, float, np.number)), (
                    f"Prior {param}['sd'] should be numeric, got {type(value['sd'])}"
                )
                assert np.isfinite(value["mean"]), f"Prior {param}['mean'] should be finite"
                assert np.isfinite(value["sd"]), f"Prior {param}['sd'] should be finite"

        # Other keys may be flat numeric values
        for key, value in priors.items():
            if key not in main_params:
                if isinstance(value, (int, float, np.number)):
                    assert np.isfinite(value), f"Prior {key} should be finite, got {value}"

    def test_get_chen_priors_parameter_ranges(self):
        """Should return priors with reasonable parameter ranges."""
        priors = get_chen_priors(2)

        # Check main parameters with nested structure
        main_params = ["scale", "ramp", "decay", "offset"]
        for key in main_params:
            if key in priors and isinstance(priors[key], dict):
                value = priors[key]
                # Check mean values
                assert not np.isnan(value["mean"]), f"Prior {key}['mean'] should not be NaN"
                assert not np.isinf(value["mean"]), f"Prior {key}['mean'] should not be infinite"
                assert abs(value["mean"]) < 1e6, (
                    f"Prior {key}['mean']={value['mean']} seems unreasonably large"
                )

                # Check sd (standard deviation) values
                assert not np.isnan(value["sd"]), f"Prior {key}['sd'] should not be NaN"
                assert not np.isinf(value["sd"]), f"Prior {key}['sd'] should not be infinite"
                assert value["sd"] > 0, f"Prior {key}['sd'] should be positive, got {value['sd']}"
                assert value["sd"] < 1e6, (
                    f"Prior {key}['sd']={value['sd']} seems unreasonably large"
                )

        # Check other numeric values (like seMilk)
        for key, value in priors.items():
            if key not in main_params and isinstance(value, (int, float, np.number)):
                assert not np.isnan(value), f"Prior {key} should not be NaN"
                assert not np.isinf(value), f"Prior {key} should not be infinite"
                assert value > 0, f"Prior {key} should be positive, got {value}"
                assert value < 1e6, f"Prior {key}={value} seems unreasonably large"

    def test_get_chen_priors_consistency_across_parities(self):
        """Should return different but consistent priors for different parities."""
        priors_parity1 = get_chen_priors(1)
        priors_parity2 = get_chen_priors(2)
        priors_parity3 = get_chen_priors(3)

        # All should have the same keys
        assert set(priors_parity1.keys()) == set(priors_parity2.keys()), (
            "Priors for different parities should have same keys"
        )
        assert set(priors_parity2.keys()) == set(priors_parity3.keys()), (
            "Priors for different parities should have same keys"
        )

        # Main parameters should have consistent structure (mean and sd)
        main_params = ["scale", "ramp", "decay", "offset"]
        for key in main_params:
            if key in priors_parity1:
                assert isinstance(priors_parity1[key], dict) and "mean" in priors_parity1[key]
                assert isinstance(priors_parity2[key], dict) and "mean" in priors_parity2[key]
                assert isinstance(priors_parity3[key], dict) and "mean" in priors_parity3[key]

        # At least some mean values should differ between parities
        # (lactation curves differ by parity)
        values_differ = False
        for key in main_params:
            if key in priors_parity1 and isinstance(priors_parity1[key], dict):
                if priors_parity1[key]["mean"] != priors_parity2[key]["mean"]:
                    values_differ = True
                    break
        assert values_differ, "Prior means should differ between parities"
