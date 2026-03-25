"""
Lactation curve fitting module.

This module provides functions for fitting lactation curve models to dairy cow
lactation data and predicting milk yield.

Pre-defined lactation curve models
----------------------------------
- Models that can be fitted using frequentist statistics (numeric or least squares optimization):
    - Wood
    - Wilmink
    - Ali & Schaeffer
    - Fischer
    - MilkBot

- Models that can be fitted using Bayesian fitting:
    - MilkBot

- Models that are currently implemented but not yet available for fitting:
    - Brody
    - Sikka
    - Nelder
    - Dhanoa
    - Emmans
    - Hayashi
    - Rook
    - Dijkstra
    - Prasad

Author: Meike van Leerdam
Date: 12-8-2025
Last update: 11-feb-2025

Requires
--------
- numpy
- scipy
- requests
- lactationcurve.preprocessing.validate_and_prepare_inputs

Public functions
----------------
- fit_lactation_curve(dim, milkrecordings, model="wood",
  fitting="frequentist", breed="H", parity=3,
  continent="USA", key=None)
  Fit a lactation curve to the provided data and return predicted milk yield
  for each day in milk (DIM) in the range 1–305 (or up to the maximum DIM if it exceeds 305).

- get_lc_parameters(dim, milkrecordings, model="wood")
  Fit a lactation curve to the provided data and return model parameters using
  frequentist statistics: minimize/curve_fit.

- get_lc_parameters_least_squares(dim, milkrecordings, model="milkbot")
  Fit a lactation curve to the provided data and return model parameters using
  least-squares estimation (frequentist).

Notes
-----
- Units: DIM in days, milk in kg or lb.
- Input validation and normalization are delegated to
  `lactationcurve.preprocessing.validate_and_prepare_inputs`.
- Bayesian fitting for the MilkBot model is performed via the MilkBot API,
  this requires an API key and accepts additional parameters for breed,
  parity, continent, and custom priors.
- For information on the fitting API see https://api.milkbot.com/
  or contact Jim Ehrlich, DVM: jehrlich@MilkBot.com
"""

# packages
from __future__ import annotations

import numpy as np
import requests
from scipy.optimize import curve_fit, least_squares, minimize

from lactationcurve.preprocessing import validate_and_prepare_inputs
from lactationcurve.preprocessing.validate_and_standardize import MilkBotPriors


# --- Models ---
def milkbot_model(t, a, b, c, d) -> np.floating | np.ndarray:
    """MilkBot lactation curve model.

    Args:
        t: Time since calving in days (DIM), scalar or array-like.
        a: Scale; overall level of milk production.
        b: Ramp; governs the rate of rise in early lactation.
        c: Offset; small (usually minor) correction around the theoretical start of lactation.
        d: Decay; exponential decline rate, evident in late lactation.

    Returns:
        Predicted milk yield at `t` (same shape as `t`).

    Notes:
        Formula: `y(t) = a * (1 - exp((c - t) / b) / 2) * exp(-d * t)`.
    """
    return a * (1 - np.exp((c - t) / b) / 2) * np.exp(-d * t)


def wood_model(t, a, b, c) -> np.floating | np.ndarray:
    """Wood lactation curve model.

    Args:
        t: Time since calving in days (DIM), scalar or array-like.
        a: Scale parameter (numerical).
        b: Shape parameter controlling rise (numerical).
        c: Decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a * t**b * exp(-c * t)`.
    """
    return a * (t**b) * np.exp(-c * t)


def wilmink_model(t, a, b, c, k=-0.05) -> np.floating | np.ndarray:
    """Wilmink lactation curve model.

    Args:
        t: Time since calving in days (DIM), scalar or array-like.
        a: Intercept-like parameter (numerical).
        b: Linear trend coefficient (numerical).
        c: Exponential-term scale (numerical).
        k: Fixed exponential rate (numerical), default -0.05.

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a + b * t + c * exp(k * t)`.
    """
    t = np.asarray(t)
    return a + b * t + c * np.exp(k * t)


def ali_schaeffer_model(t, a, b, c, d, k) -> np.floating | np.ndarray:
    """Ali & Schaeffer lactation curve model.

    Args:
        t: Time since calving in days (DIM). Use `t >= 1` to avoid `log(0)`.
        a: Intercept-like parameter (numerical).
        b: Linear coefficient on scaled time `t/305` (numerical).
        c: Quadratic coefficient on scaled time `t/305` (numerical).
        d: Coefficient on `log(305/t)` (numerical).
        k: Coefficient on `[log(305/t)]^2` (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Uses `t_scaled = t / 305` and `log_term = ln(305 / t)`.
    """
    t_scaled = t / 305
    log_term = np.log(305 / t)
    return a + b * t_scaled + c * (t_scaled**2) + d * log_term + k * (log_term**2)


def fischer_model(t, a, b, c) -> np.floating | np.ndarray:
    """Fischer lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Linear decline parameter (numerical).
        c: Exponential decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.
    """
    return a - b * t - a * np.exp(-c * t)


def brody_model(t, a, k) -> float:
    """Brody lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        k: Decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.
    """
    return a * np.exp(-k * t)


def sikka_model(t, a, b, c) -> float:
    """Sikka lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Growth parameter (numerical).
        c: Quadratic decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.
    """
    return a * np.exp(b * t - c * t**2)


def nelder_model(t, a, b, c) -> float:
    """Nelder lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Denominator intercept (numerical).
        b: Denominator linear coefficient (numerical).
        c: Denominator quadratic coefficient (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = t / (a + b*t + c*t**2)`.
    """
    return t / (a + b * t + c * t**2)


def dhanoa_model(t, a, b, c) -> float:
    """Dhanoa lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Shape parameter (numerical).
        c: Decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a * t ** (b * c) * exp(-c * t)`.
    """
    return a * t ** (b * c) * np.exp(-c * t)


def emmans_model(t, a, b, c, d) -> float:
    """Emmans lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Growth parameter (numerical).
        c: Decay parameter (numerical).
        d: Location parameter in nested exponential (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a * exp(-exp(d - b*t)) * exp(-c*t)`.
    """
    return a * np.exp(-np.exp(d - b * t)) * np.exp(-c * t)


def hayashi_model(t, a, b, c, d) -> float:
    """Hayashi lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Ratio parameter (> 0) (numerical).
        b: Scale parameter (numerical).
        c: Time constant for the first exponential term (numerical).
        d: Parameter retained for compatibility with literature (unused in this expression).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = b * (exp(-t / c) - exp(-t / (a * c)))`.
    """
    return b * (np.exp(-t / c) - np.exp(-t / (a * c)))


def rook_model(t, a, b, c, d) -> float:
    """Rook lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Shape parameter in rational term (numerical).
        c: Offset parameter in rational term (numerical).
        d: Exponential decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a * (1 / (1 + b / (c + t))) * exp(-d * t)`.
    """
    return a * (1 / (1 + b / (c + t))) * np.exp(-d * t)


def dijkstra_model(t, a, b, c, d) -> float:
    """Dijkstra lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Scale parameter (numerical).
        b: Growth parameter (numerical).
        c: Saturation rate parameter (numerical).
        d: Decay parameter (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a * exp((b * (1 - exp(-c * t)) / c) - d * t)`.
    """
    return a * np.exp((b * (1 - np.exp(-c * t)) / c) - d * t)


def prasad_model(t, a, b, c, d) -> float:
    """Prasad lactation curve model.

    Args:
        t: Time since calving in days (DIM).
        a: Intercept-like parameter (numerical).
        b: Linear coefficient (numerical).
        c: Quadratic coefficient (numerical).
        d: Inverse-time coefficient (numerical).

    Returns:
        Predicted milk yield at `t`.

    Notes:
        Formula: `y(t) = a + b*t + c*t**2 + d/t`.
    """
    return a + b * t + c * t**2 + d / t


# objectives for minimize
def wood_objective(par, x, y) -> float:
    """Objective function (sum of squared errors) for the Wood model.

    Args:
        par: Parameter vector `(a, b, c)`.
        x: DIM values.
        y: Observed milk yields.

    Returns:
        Sum of squared residuals between observed values and Wood model predictions.
    """
    return np.sum((y - wood_model(x, *par)) ** 2)


def milkbot_objective(par, x, y) -> float:
    """Objective function (sum of squared errors) for the MilkBot model.

    Args:
        par: Parameter vector `(a, b, c, d)`.
        x: DIM values.
        y: Observed milk yields.

    Returns:
        Sum of squared residuals between observed values and MilkBot predictions.
    """
    return np.sum((y - milkbot_model(x, *par)) ** 2)


def residuals_milkbot(par, x, y) -> np.ndarray:
    """Residuals for least-squares fitting of the MilkBot model.

    Args:
        par: Parameter vector `(a, b, c, d)`.
        x: DIM values.
        y: Observed milk yields.

    Returns:
        Vector of residuals `y - y_pred`.
    """
    return y - milkbot_model(x, *par)


def fit_lactation_curve(
    dim,
    milkrecordings,
    model="wood",
    fitting="frequentist",
    breed="H",
    parity=3,
    continent="USA",
    custom_priors=None,
    key=None,
    milk_unit="kg",
) -> np.ndarray:
    """Fit lactation data to a lactation curve model and return predictions.

    Depending on `fitting`:
    - **frequentist**: Fits parameters using `minimize` and/or `curve_fit`
      for the specified `model`, then predicts over DIM 1–305 (or up to `max(dim)` if greater).
    - **bayesian**: (MilkBot only) Calls the MilkBot Bayesian fitting API and
      returns predictions using the fitted parameters.

    Args:
        dim (Int): List/array of days in milk (DIM).
        milkrecordings (Float): List/array of milk recordings (kg).
        model (Str): Model name (lowercase), default "wood".
            Supported for frequentist: "wood", "wilmink", "ali_schaeffer", "fischer", "milkbot".
        fitting (Str): "frequentist" (default) or "bayesian".
            Bayesian fitting is currently implemented only for "milkbot".
        breed (Str): "H" (Holstein, default) or "J" (Jersey). Only used for Bayesian.
        parity (Int): Lactation number; all parities >= 3 considered one group in priors.
            Only used for Bayesian.
        continent (Str): priors chosen by MilkBot API based on continent averages.
            Only used for Bayesian, options: "USA" (default) and "EU".
        custom_priors (Dict | str | None): Custom prior
            distributions for Bayesian fitting.
            If a dict is provided, it must be a dictionary
            of prior distributions for each parameter
            in the model.
            Set the correct dictionary using the `build_prior` helper function.
            If the string "CHEN" is provided, the default Chen et al. priors are used.
            Only used for Bayesian.
        key = Str: API key for MilkBot API (required for Bayesian fitting).
            Only used for Bayesian.
        milk_unit (Str): Unit of milk yield measurements. Must be either "kg" or "lbs".
            Default is "kg".
            Only used for Bayesian.


    Returns:
        List/array of predicted milk yield for DIM 1–305 (or up to the maximum DIM if > 305).

    Raises:
        Exception: If an unknown model is requested (frequentist),
            or Bayesian is requested for a non-MilkBot model,
            or `key` is missing when Bayesian fitting is requested.

    Notes:
        Uses `validate_and_prepare_inputs` for input checking and normalization.
    """
    # check and prepare input
    inputs = validate_and_prepare_inputs(
        dim,
        milkrecordings,
        model=model,
        fitting=fitting,
        breed=breed,
        parity=parity,
        continent=continent,
        custom_priors=custom_priors,
        milk_unit=milk_unit,
    )

    dim = inputs.dim
    milkrecordings = inputs.milkrecordings
    model = inputs.model
    fitting = inputs.fitting
    breed = inputs.breed
    parity = inputs.parity
    continent = inputs.continent
    custom_priors = inputs.custom_priors
    milk_unit = inputs.milk_unit

    if fitting == "frequentist":
        if model == "wood":
            params = get_lc_parameters(dim, milkrecordings, model)
            assert params is not None, "Failed to fit Wood model parameters"
            a_w, b_w, c_w = params[0], params[1], params[2]
            if max(dim) > 305:
                t_range = np.arange(1, (max(dim) + 1))
                y_w = wood_model(t_range, a_w, b_w, c_w)
            else:
                t_range = np.arange(1, 306)
                y_w = wood_model(t_range, a_w, b_w, c_w)
            return np.asarray(y_w)

        elif model == "wilmink":
            params = get_lc_parameters(dim, milkrecordings, model)
            assert params is not None, "Failed to fit Wilmink model parameters"
            a_wil, b_wil, c_wil, k_wil = (params[0], params[1], params[2], params[3])
            if max(dim) > 305:
                t_range = np.arange(1, (max(dim) + 1))
                y_wil = wilmink_model(t_range, a_wil, b_wil, c_wil, k_wil)
            else:
                t_range = np.arange(1, 306)
                y_wil = wilmink_model(t_range, a_wil, b_wil, c_wil, k_wil)
            return np.asarray(y_wil)

        elif model == "ali_schaeffer":
            params = get_lc_parameters(dim, milkrecordings, model)
            assert params is not None, "Failed to fit Ali & Schaeffer model parameters"
            a_as, b_as, c_as, d_as, k_as = (params[0], params[1], params[2], params[3], params[4])
            if max(dim) > 305:
                t_range = np.arange(1, (max(dim) + 1))
                y_as = ali_schaeffer_model(t_range, a_as, b_as, c_as, d_as, k_as)
            else:
                t_range = np.arange(1, 306)
                y_as = ali_schaeffer_model(t_range, a_as, b_as, c_as, d_as, k_as)
            return np.asarray(y_as)

        elif model == "fischer":
            params = get_lc_parameters(dim, milkrecordings, model)
            assert params is not None, "Failed to fit Fischer model parameters"
            a_f, b_f, c_f = params[0], params[1], params[2]
            if max(dim) > 305:
                t_range = np.arange(1, (max(dim) + 1))
                y_f = fischer_model(t_range, a_f, b_f, c_f)
            else:
                t_range = np.arange(1, 306)
                y_f = fischer_model(t_range, a_f, b_f, c_f)
            return np.asarray(y_f)

        elif model == "milkbot":
            params = get_lc_parameters(dim, milkrecordings, model)
            assert params is not None, "Failed to fit MilkBot model parameters"
            a_mb, b_mb, c_mb, d_mb = (params[0], params[1], params[2], params[3])
            if max(dim) > 305:
                t_range = np.arange(1, (max(dim) + 1))
            else:
                t_range = np.arange(1, 306)

            y_mb = milkbot_model(t_range, a_mb, b_mb, c_mb, d_mb)
            return np.asarray(y_mb)

        else:
            raise Exception("Unknown model")
    else:
        if model == "milkbot":
            if key is None:
                raise Exception("Key needed to use Bayesian fitting engine milkbot")
            else:
                assert parity is not None, "parity is required for Bayesian fitting"
                assert breed is not None, "breed is required for Bayesian fitting"
                assert continent is not None, "continent is required for Bayesian fitting"
                parameters = bayesian_fit_milkbot_single_lactation(
                    dim,
                    milkrecordings,
                    key,
                    parity,
                    breed,
                    custom_priors,
                    continent,
                    milk_unit or "kg",
                )
                if max(dim) > 305:
                    t_range = np.arange(1, (max(dim) + 1))
                    y_mb_bay = milkbot_model(
                        t_range,
                        parameters["scale"],
                        parameters["ramp"],
                        parameters["offset"],
                        parameters["decay"],
                    )
                else:
                    t_range = np.arange(1, 306)
                    y_mb_bay = milkbot_model(
                        t_range,
                        parameters["scale"],
                        parameters["ramp"],
                        parameters["offset"],
                        parameters["decay"],
                    )
                return np.asarray(y_mb_bay)
        else:
            raise Exception("Bayesian fitting is currently only implemented for milkbot models")


def get_lc_parameters_least_squares(
    dim, milkrecordings, model="milkbot"
) -> tuple[float, float, float, float]:
    """Fit lactation data and return model parameters (least squares; frequentist).

    This helper uses `scipy.optimize.least_squares` to fit the MilkBot model with bounds,
    and returns the fitted parameters.
    Currently implemented only for the MilkBot model, as it is
    more complex and benefits from the robust optimization approach.
    Other models can be fitted using `get_lc_parameters` with
    numerical optimisation, which is generally faster for simpler
    models.

    Args:
        dim (int): List/array of DIM values.
        milkrecordings (float): List/array of milk recordings (kg).
        model (str): Pre-defined model name; currently used with "milkbot".

    Returns:
        Parameters `(a, b, c, d)` as `np.float` in alphabetic order.

    """
    # check and prepare input
    inputs = validate_and_prepare_inputs(dim, milkrecordings, model=model)

    dim = inputs.dim
    milkrecordings = inputs.milkrecordings
    model = inputs.model

    # ------------------------------
    # Initial guess
    # ------------------------------
    a0 = np.max(milkrecordings)
    b0 = 50.0
    c0 = 30.0
    d0 = 0.01
    p0 = [a0, b0, c0, d0]

    # ------------------------------
    # Parameter bounds
    # ------------------------------
    lower = [np.max(milkrecordings) * 0.5, 1.0, -300.0, 1e-6]
    upper = [np.max(milkrecordings) * 8.0, 400.0, 300.0, 1.0]

    # ------------------------------
    # Fit using least-squares
    # ------------------------------
    res = least_squares(
        residuals_milkbot,
        p0,
        args=(dim, milkrecordings),
        bounds=(lower, upper),
        method="trf",  # trust region reflective, works well with bounds
    )

    # ------------------------------
    # Extract parameters
    # ------------------------------
    a_mb, b_mb, c_mb, d_mb = res.x

    return a_mb, b_mb, c_mb, d_mb


def get_lc_parameters(dim, milkrecordings, model="wood") -> tuple[float, ...]:
    """Fit lactation data to a model and return fitted parameters (frequentist).

    Depending on `model`, this uses `scipy.optimize.minimize` and/or
    `scipy.optimize.curve_fit` with model-specific starting values and bounds.

    Args:
        dim (int): List/array of DIM values.
        milkrecordings (float): List/array of milk recordings (kg).
        model (str): One of "wood", "wilmink", "ali_schaeffer", "fischer", "milkbot".

    Returns:
        Fitted parameters as floats, in alphabetical order by parameter name:
            - wood: (a, b, c)
            - wilmink: (a, b, c, k) with k fixed at -0.05
            - ali_schaeffer: (a, b, c, d, k)
            - fischer: (a, b, c)
            - milkbot: (a, b, c, d)
    """
    # check and prepare input
    inputs = validate_and_prepare_inputs(dim, milkrecordings, model=model)

    dim = inputs.dim
    milkrecordings = inputs.milkrecordings
    model = inputs.model

    if model == "wood":
        wood_guess = [30, 0.2, 0.01]
        wood_bounds = [(1, 100), (0.01, 1.5), (0.0001, 0.1)]
        wood_res = minimize(
            wood_objective, wood_guess, args=(dim, milkrecordings), bounds=wood_bounds
        )
        a_w, b_w, c_w = wood_res.x
        return a_w, b_w, c_w

    elif model == "wilmink":
        wil_guess = [10, 0.1, 30]
        wil_params, _ = curve_fit(wilmink_model, dim, milkrecordings, p0=wil_guess)
        a_wil, b_wil, c_wil = wil_params
        k_wil = -0.05  # set fixed
        return a_wil, b_wil, c_wil, k_wil

    elif model == "ali_schaeffer":
        ali_schaeffer_guess = [10, 10, -5, 1, 1]
        ali_schaeffer_params, _ = curve_fit(
            ali_schaeffer_model, dim, milkrecordings, p0=ali_schaeffer_guess
        )
        a_as, b_as, c_as, d_as, k_as = ali_schaeffer_params
        return a_as, b_as, c_as, d_as, k_as

    elif model == "fischer":
        fischer_guess = [max(milkrecordings), 0.01, 0.01]
        fischer_bounds = [(0, 100), (0, 1), (0.0001, 1)]
        fischer_params, _ = curve_fit(
            fischer_model,
            dim,
            milkrecordings,
            p0=fischer_guess,
            bounds=np.transpose(fischer_bounds),
        )
        a_f, b_f, c_f = fischer_params
        return a_f, b_f, c_f

    elif model == "milkbot":
        mb_guess = [max(milkrecordings), 20.0, -0.7, 0.022]
        mb_bounds = [(1, 100), (1, 100), (-600, 300), (0.0001, 0.1)]
        mb_res = minimize(milkbot_objective, mb_guess, args=(dim, milkrecordings), bounds=mb_bounds)
        a_mb, b_mb, c_mb, d_mb = mb_res.x
        return a_mb, b_mb, c_mb, d_mb

    raise ValueError(f"Unknown model: {model}")


def get_chen_priors(parity: int) -> dict:
    """
    Return Chen et al. priors in MilkBot format.

    Args:
        parity: Lactation number (1, 2, or >= 3).

    Returns:
        Dictionary with parameter priors:
        - "scale": {"mean", "sd"}
        - "ramp": {"mean", "sd"}
        - "decay": {"mean", "sd"}
        - "offset": {"mean", "sd"}
        - "seMilk": Standard error of milk measurement.
        - "milkUnit": Unit string (e.g., "kg").
    """
    if parity == 1:
        return {
            "scale": {"mean": 34.11, "sd": 7},
            "ramp": {"mean": 29.96, "sd": 3},
            "decay": {"mean": 0.001835, "sd": 0.000738},
            "offset": {"mean": -0.5, "sd": 0.02},
            "seMilk": 4,
            "milkUnit": "kg",
        }

    if parity == 2:
        return {
            "scale": {"mean": 44.26, "sd": 9.57},
            "ramp": {"mean": 22.52, "sd": 3},
            "decay": {"mean": 0.002745, "sd": 0.000979},
            "offset": {"mean": -0.78, "sd": 0.07},
            "seMilk": 4,
            "milkUnit": "kg",
        }

    # parity >= 3
    return {
        "scale": {"mean": 48.41, "sd": 10.66},
        "ramp": {"mean": 22.54, "sd": 8.724},
        "decay": {"mean": 0.002997, "sd": 0.000972},
        "offset": {"mean": 0.0, "sd": 0.03},
        "seMilk": 4,
        "milkUnit": "kg",
    }


def build_prior(
    scale_mean: float,
    scale_sd: float,
    ramp_mean: float,
    ramp_sd: float,
    decay_mean: float,
    decay_sd: float,
    offset_mean: float,
    offset_sd: float,
    se_milk: float = 4,
) -> dict:
    return {
        "scale": {"mean": scale_mean, "sd": scale_sd},
        "ramp": {"mean": ramp_mean, "sd": ramp_sd},
        "decay": {"mean": decay_mean, "sd": decay_sd},
        "offset": {"mean": offset_mean, "sd": offset_sd},
        "seMilk": se_milk,
    }


def bayesian_fit_milkbot_single_lactation(
    dim,
    milkrecordings,
    key: str,
    parity=3,
    breed="H",
    custom_priors: MilkBotPriors | str | None = None,
    continent="USA",
    milk_unit="kg",
) -> dict:
    """
    Fit a single lactation using the MilkBot API.

    Args:
        dim: List/array of DIM values.
        milkrecordings: List/array of milk recordings (kg).
        key: API key for MilkBot.
        parity: Lactation number; values >= 3 are treated as one group in priors.
        breed: "H" (Holstein) or "J" (Jersey).
        custom_priors:
            - "CHEN"  → Chen et al. published priors
            - dict    → Custom priors in MilkBot format (overrides `continent`)
        continent: priors used by MilkBot API for fitting; options:
            - "USA"   → MilkBot USA priors
            - "EU"    → MilkBot EU priors > estimates lower milk production


    Returns:
        Dictionary with fitted parameters and metadata:
            {
                "scale": float,
                "ramp": float,
                "decay": float,
                "offset": float,
                "nPoints": int
            }

    Raises:
        requests.HTTPError: For unsuccessful HTTP response codes.
        RuntimeError: If the response format is unexpected.

    Notes:
        - When `continent == "CHEN"`, Chen et al. priors are included in the request payload.
        - EU calls use the GCP EU endpoint; others use `milkbot.com`.
    """
    # check and prepare input
    inputs = validate_and_prepare_inputs(
        dim,
        milkrecordings,
        breed=breed,
        parity=parity,
        custom_priors=custom_priors,
        continent=continent,
        milk_unit=milk_unit,
    )

    dim = inputs.dim
    milkrecordings = inputs.milkrecordings
    breed = inputs.breed
    parity = inputs.parity
    continent = inputs.continent
    custom_priors = inputs.custom_priors
    milk_unit = inputs.milk_unit

    # -----------------------------
    # Select server (USA vs EU)
    # -----------------------------
    if continent == "EU":
        base_url = "https://europe-west1-numeric-analogy-337601.cloudfunctions.net/milkBot-fitter"
    else:
        base_url = "https://milkbot.com"

    # -----------------------------
    # Prepare headers
    # -----------------------------
    headers = {"Content-Type": "application/json", "X-API-KEY": key}

    # -----------------------------
    # Prepare milk points
    # -----------------------------
    points = sorted(
        ({"dim": int(d), "milk": float(m)} for d, m in zip(dim, milkrecordings)),
        key=lambda p: p["dim"],
    )

    # -----------------------------
    # Lactation metadata
    # -----------------------------
    payload = {
        "lactation": {
            "lacKey": "single_lactation_fit",
            "breed": breed,
            "parity": parity,
            "points": points,
        },
        "options": {
            "returnInputData": False,
            "returnPath": False,
            "returnDiscriminatorPath": False,
            # "fitEngine": "AnnealingFitter@2.0", #comment out to use the default fitter
            # "fitObjective": "MB2@2.0",
            "preferredMilkUnit": milk_unit,
        },
    }

    # -----------------------------
    # Add priors if provided or when using Chen et al. priors
    # -----------------------------
    if custom_priors == "CHEN":
        assert parity is not None, "parity is required for Chen priors"
        payload["priors"] = get_chen_priors(parity)

    elif isinstance(custom_priors, dict):
        payload["priors"] = dict(custom_priors)

    # -----------------------------
    # Call API
    # -----------------------------
    response = requests.post(f"{base_url}/fitLactation", headers=headers, json=payload)
    response.raise_for_status()
    res = response.json()

    # -----------------------------
    # Normalize response (USA vs EU)
    # -----------------------------
    if "fittedParams" in res:
        # USA-style response
        fitted = res["fittedParams"]
    elif "params" in res:
        fitted = res["params"]
    else:
        raise RuntimeError(f"Unexpected MilkBot response format: {res}")

    # -----------------------------
    # Parse result
    # -----------------------------

    return {
        "scale": fitted["scale"],
        "ramp": fitted["ramp"],
        "decay": fitted["decay"],
        "offset": fitted["offset"],
        "nPoints": len(points),
    }


def get_milkbot_version() -> None:
    """Get the current version of the MilkBot API."""
    r = requests.get(url="https://milkbot.com/version")
    print(r.json())
