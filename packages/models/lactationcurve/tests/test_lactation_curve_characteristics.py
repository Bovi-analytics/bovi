"""
Test suite for lactation curve characteristics and persistency calculations.

This module provides comprehensive tests for characteristic functions, persistency calculations,
numeric helpers, and characteristic extraction for lactation curve models.

Test Categories:
    - TestCharacteristicFunction: Symbolic and functional characteristic extraction for models
    - TestPersistencyFunctions: Persistency calculations for Wood and MilkBot models
    - TestCalculateCharacteristicFrequentist: Frequentist characteristic extraction from real data
    - TestCalculateCharacteristicBayesian: Bayesian characteristic extraction (MilkBot)
    - TestCalculateCharacteristicErrors: Error handling and validation
    - TestNumericHelpers: Numeric characteristic calculations
    - TestPersistencyFittedCurve: Persistency from fitted curves

Usage:
    Run all tests::

        pytest test_lactation_curve_characteristics.py -v

    Run specific marker::

        pytest test_lactation_curve_characteristics.py -m characteristic -v
        pytest test_lactation_curve_characteristics.py -m persistency -v
        pytest test_lactation_curve_characteristics.py -m frequentist -v
        pytest test_lactation_curve_characteristics.py -m bayesian -v
        pytest test_lactation_curve_characteristics.py -m errorhandling -v
        pytest test_lactation_curve_characteristics.py -m numeric -v

Authors:
    Meike van Leerdam, Judith Osei-Tete
"""

import math
import os
from pathlib import Path

import pandas as pd
import pytest
import sympy as sp
from dotenv import find_dotenv, load_dotenv

from lactationcurve.characteristics import (
    calculate_characteristic,
    lactation_curve_characteristic_function,
    numeric_cumulative_yield,
    numeric_peak_yield,
    numeric_time_to_peak,
    persistency_fitted_curve,
    persistency_milkbot,
    persistency_wood,
)


@pytest.fixture
def key() -> str:
    """Return the MilkBot API key from environment."""
    load_dotenv(find_dotenv())
    key = os.getenv("milkbot_key")
    if not key:
        raise ValueError("milkbot_key not found in environment. Check your .env file.")
    return key


@pytest.fixture
def test_df():
    data_path = Path(__file__).parent / "test_data" / "l2_anim2_herd654.csv"
    df = pd.read_csv(data_path, sep=",")
    dim = df.DaysInMilk.values
    my = df.TestDayMilkYield.values
    return dim, my


@pytest.mark.characteristic
class TestCharacteristicFunction:
    """
    Verify symbolic and functional characteristic extraction.

    This test class verifies symbolic and functional characteristic
    extraction for lactation curve models. It checks symbolic
    expressions, parameter extraction, and callable generation
    for a variety of supported models.

    Attributes:
        Tests cover: Wood, Brody, Sikka, Wilmink, Ali-Schaeffer.
    """

    def test_invalid_model_raises(self):
        """Test that an Exception is raised for an unknown model name."""
        with pytest.raises(Exception, match="Unknown model"):
            lactation_curve_characteristic_function(model="not_a_model")

    def test_wood_time_to_peak_returns_expr_params_and_func(self):
        """Test for correct symbolic expression, parameter symbols,
        and callable for Wood time to peak.
        """
        expr, params, func = lactation_curve_characteristic_function(
            model="wood", characteristic="time_to_peak"
        )
        assert isinstance(expr, sp.Expr)
        assert all(isinstance(s, sp.Symbol) for s in params)
        assert {s.name for s in params} == {"a", "b", "c"}
        assert callable(func)

    def test_wood_peak_yield_returns_expr_params_and_func(self):
        """Test for correct symbolic expression, parameter symbols,
        and callable for Wood peak yield.
        """
        expr, params, func = lactation_curve_characteristic_function(
            model="wood", characteristic="peak_yield"
        )
        assert isinstance(expr, sp.Expr)
        assert {s.name for s in params} == {"a", "b", "c"}
        assert callable(func)

    def test_wood_cumulative_returns_expr_params_and_func(self):
        """Test for correct symbolic expression, parameter symbols,
        and callable for Wood cumulative yield.
        """
        expr, params, func = lactation_curve_characteristic_function(
            model="wood", characteristic="cumulative_milk_yield"
        )
        assert isinstance(expr, sp.Expr)
        assert {s.name for s in params} == {"a", "b", "c"}
        assert callable(func)

    def test_default_returns_dict_with_all_four(self):
        """Test that the default call returns a dict with all four
        characteristics for the Wood model.
        """
        result, params, func = lactation_curve_characteristic_function(model="wood")
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "time_to_peak",
            "peak_yield",
            "cumulative_milk_yield",
            "persistency",
        }
        assert all(isinstance(v, (sp.Expr, type(None))) for v in result.values())
        assert {s.name for s in params} == {"a", "b", "c"}
        assert isinstance(func, dict)
        assert callable(list(func.values())[0])

    def test_brody_cumulative_returns_expr_params_and_func(self):
        """Test for correct symbolic expression, parameter symbols,
        and callable for Brody cumulative yield.
        """
        expr, params, func = lactation_curve_characteristic_function(
            "brody", characteristic="cumulative_milk_yield"
        )
        assert isinstance(expr, sp.Expr)
        assert {s.name for s in params} == {"a", "b", "k1", "k2"}
        assert callable(func)

    @pytest.mark.parametrize(
        "model,expected_symbols",
        [
            ("sikka", {"a", "b", "c"}),
            ("wilmink", {"a", "b", "c", "k"}),
            ("ali_schaeffer", {"a", "b", "c", "d", "k"}),
        ],
    )
    def test_models_return_expected_symbols(self, model, expected_symbols):
        """Test that the correct parameter symbols are returned for each model."""
        _, params, func = lactation_curve_characteristic_function(
            model=model, characteristic="time_to_peak"
        )
        assert {s.name for s in params} == expected_symbols
        assert callable(func)


@pytest.mark.persistency
class TestPersistencyFunctions:
    """
    Validate persistency calculations for Wood and MilkBot models.

    It ensures correct computation, return types, and error
    handling for persistency-related functions.

    Attributes:
        Tests cover: persistency_wood, persistency_milkbot,
        error handling for zero decay.
    """

    def test_persistency_wood_basic(self):
        """Test for correct persistency calculation for the Wood model
        with known input and output."""
        result = persistency_wood(2, 0.5)
        expected = -(2 + 1) * math.log(0.5)  # = -3 * ln(0.5)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_persistency_wood_type(self):
        """Test that persistency_wood returns a float value."""
        result = persistency_wood(1, 0.8)
        assert isinstance(result, float)

    def test_persistency_milkbot_basic(self):
        """Test for correct persistency calculation for the MilkBot
        model with known input and output.
        """
        result = persistency_milkbot(2)
        expected = 0.693 / 2
        assert pytest.approx(result, rel=1e-9) == expected

    def test_persistency_milkbot_type(self):
        """Test that persistency_milkbot returns a float value."""
        result = persistency_milkbot(5)
        assert isinstance(result, float)

    def test_persistency_milkbot_zero_division(self):
        """Test that ZeroDivisionError is raised for zero decay in MilkBot persistency."""
        with pytest.raises(ZeroDivisionError):
            persistency_milkbot(0)


@pytest.mark.frequentist
class TestCalculateCharacteristicFrequentist:
    """
    Check frequentist characteristic extraction from real data.

    It includes tests for time to peak, peak yield, cumulative
    yield, and persistency calculations using the Wood model.

    Attributes:
        Tests cover: calculate_characteristic (Wood, frequentist),
        all characteristic types, custom lactation length.
    """

    def test_time_to_peak_frequentist_wood(self, test_df):
        """Test for correct time to peak calculation for the Wood model (frequentist)."""
        dim, my = test_df
        result = calculate_characteristic(dim, my, model="wood", characteristic="time_to_peak")
        expected = 45
        assert isinstance(result, float)
        assert 0 < result < 305
        assert pytest.approx(result, abs=1) == expected

    def test_peak_yield_frequentist_wood(self, test_df):
        """Test for correct peak yield calculation for the Wood model (frequentist)."""
        dim, my = test_df
        result = calculate_characteristic(dim, my, model="wood", characteristic="peak_yield")
        expected = 46.40100132439492
        assert isinstance(result, float)
        assert result > 0
        assert pytest.approx(result, abs=1e-3) == expected

    def test_cumulative_frequentist_wood(self, test_df):
        """Test for correct cumulative yield calculation for the Wood model (frequentist)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim, my, model="wood", characteristic="cumulative_milk_yield"
        )
        expected = 10644.871485478427
        assert isinstance(result, float)
        assert result > 0
        assert pytest.approx(result, abs=2) == expected

    def test_cumulative_frequentist_wood_with_lactation_length(self, test_df):
        """Test for cumulative yield calculation for the Wood model
        with a custom lactation length.
        """
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="wood",
            characteristic="cumulative_milk_yield",
            lactation_length=200,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_persistency_frequentist_wood(self, test_df):
        """Test for persistency calculation for the Wood model (frequentist, literature method)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="wood",
            characteristic="persistency",
            persistency_method="literature",
        )
        assert isinstance(result, float)

    def test_persistency_derived_wood(self, test_df):
        """Test for persistency calculation for the Wood model (frequentist, derived method)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="wood",
            characteristic="persistency",
            persistency_method="derived",
        )
        assert isinstance(result, float)

    def test_persistency_derived_with_lactation_length(self, test_df):
        """Test for persistency calculation for the Wood model
        (frequentist, derived, custom lactation length).
        """
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="wood",
            characteristic="persistency",
            persistency_method="derived",
            lactation_length=200,
        )
        assert isinstance(result, float)


@pytest.mark.bayesian
class TestCalculateCharacteristicBayesian:
    """
    Validate Bayesian characteristic extraction using MilkBot.

    It covers Bayesian fitting and characteristic calculations,
    including API integration and both literature and derived
    persistency methods.

    Attributes:
        Tests cover: calculate_characteristic (MilkBot, Bayesian),
        all characteristic types, API key usage.
    """

    def test_time_to_peak_bayesian_milkbot(self, test_df, key):
        """Test for correct time to peak calculation for the MilkBot model (Bayesian)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="milkbot",
            characteristic="time_to_peak",
            fitting="Bayesian",
            key=key,
        )
        expected = 35
        assert isinstance(result, float)
        assert 0 < result < 305
        assert pytest.approx(result, abs=1) == expected

    def test_peak_yield_bayesian_milkbot(self, test_df, key):
        """Test for correct peak yield calculation for the MilkBot model (Bayesian)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="milkbot",
            characteristic="peak_yield",
            fitting="Bayesian",
            key=key,
        )
        expected = 48
        assert isinstance(result, float)
        assert result > 0
        assert pytest.approx(result, abs=2) == expected

    def test_cumulative_bayesian_milkbot(self, test_df, key):
        """Test for correct cumulative yield calculation for the MilkBot model (Bayesian)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            milkrecordings=my,
            model="milkbot",
            characteristic="cumulative_milk_yield",
            fitting="Bayesian",
            key=key,
        )
        expected = 10550
        assert isinstance(result, float)
        assert result > 0
        assert pytest.approx(result, abs=50) == expected

    def test_persistency_bayesian_milkbot(self, test_df, key):
        """Test for persistency calculation for the MilkBot model (Bayesian, literature method)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="milkbot",
            characteristic="persistency",
            fitting="Bayesian",
            key=key,
            persistency_method="literature",
        )
        assert isinstance(result, float)

    def test_persistency_derived_bayesian_milkbot(self, test_df, key):
        """Test for persistency calculation for the MilkBot model (Bayesian, derived method)."""
        dim, my = test_df
        result = calculate_characteristic(
            dim,
            my,
            model="milkbot",
            characteristic="persistency",
            fitting="Bayesian",
            key=key,
            persistency_method="derived",
        )
        assert isinstance(result, float)


@pytest.mark.errorhandling
class TestCalculateCharacteristicErrors:
    """
    Ensure robust error handling and validation.

    It checks that proper exceptions are raised for invalid
    input, missing keys, and unsupported models or
    characteristics.

    Attributes:
        Tests cover: invalid model names, invalid
        characteristics, missing API keys, unsupported
        Bayesian models.
    """

    def test_invalid_characteristic_raises(self, test_df):
        """Test that an Exception is raised for an unknown characteristic name."""
        dim, my = test_df
        with pytest.raises(Exception, match="Unknown characteristic"):
            calculate_characteristic(dim, my, model="wood", characteristic="foobar")

    def test_invalid_model_in_calculate_characteristic_raises(self, test_df):
        """Test that an Exception is raised for an unsupported model in calculate_characteristic."""
        dim, my = test_df
        with pytest.raises(Exception, match="only works for"):
            calculate_characteristic(dim, my, model="brody", characteristic="cumulative_milk_yield")

    def test_bayesian_requires_key(self, test_df):
        """Test that an Exception is raised if key is missing for Bayesian fitting."""
        dim, my = test_df
        with pytest.raises(Exception, match="Key needed"):
            calculate_characteristic(
                dim,
                my,
                model="milkbot",
                characteristic="time_to_peak",
                fitting="Bayesian",
            )

    def test_bayesian_non_milkbot_raises(self, test_df, key):
        """Test that an Exception is raised if Bayesian fitting is
        requested for a non-MilkBot model.
        """
        dim, my = test_df
        with pytest.raises(
            Exception,
            match="Bayesian fitting is currently only implemented for MilkBot",
        ):
            calculate_characteristic(
                dim,
                my,
                model="wood",
                characteristic="time_to_peak",
                fitting="Bayesian",
                key=key,
            )


@pytest.mark.numeric
class TestNumericHelpers:
    """
    Cover numeric characteristic calculations for models.

    It includes numeric time to peak, peak yield, and cumulative
    yield calculations for the Wood model.

    Attributes:
        Tests cover: numeric_time_to_peak, numeric_peak_yield,
        numeric_cumulative_yield, custom lactation length.
    """

    def test_numeric_time_to_peak(self, test_df):
        """Test for correct numeric time to peak calculation for the Wood model."""
        dim, my = test_df
        result = numeric_time_to_peak(dim, my, model="wood")
        assert isinstance(result, (float, int))
        assert 0 < result < 305

    def test_numeric_peak_yield(self, test_df):
        """Test for correct numeric peak yield calculation for the Wood model."""
        dim, my = test_df
        result = numeric_peak_yield(dim, my, model="wood")
        assert isinstance(result, (float, int))
        assert result > 0

    def test_numeric_cumulative_yield(self, test_df):
        """Test for correct numeric cumulative yield calculation for the Wood model."""
        dim, my = test_df
        result = numeric_cumulative_yield(dim, my, model="wood")
        assert isinstance(result, (float, int))
        assert result > 0

    def test_numeric_cumulative_yield_with_lactation_length(self, test_df):
        """Test for numeric cumulative yield calculation for the
        Wood model with a custom lactation length.
        """
        dim, my = test_df
        result = numeric_cumulative_yield(dim, my, model="wood", lactation_length=200)
        assert isinstance(result, (float, int))
        assert result > 0


@pytest.mark.persistency
class TestPersistencyFittedCurve:
    """
    This test class validates persistency calculation from fitted lactation curves.

    It checks persistency extraction from fitted Wood model curves using frequentist fitting.

    Attributes:
        Tests cover: persistency_fitted_curve (Wood, frequentist).
    """

    def test_persistency_fitted_curve_wood(self, test_df):
        """Test for persistency calculation from a fitted Wood curve (frequentist)."""
        dim, my = test_df
        result = persistency_fitted_curve(dim, my, model="wood", fitting="frequentist")
        assert isinstance(result, (float, int))
