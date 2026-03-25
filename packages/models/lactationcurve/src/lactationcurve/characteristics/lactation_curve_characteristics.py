"""
Lactation Curve Characteristics (LCC) utilities.

This module derives and evaluates **lactation curve characteristics** (LCCs) for
common lactation models using symbolic calculus (via SymPy) and fast numeric
evaluation (via NumPy). Characteristics include:

- **time_to_peak**: day in milk (DIM) at which the model reaches its maximum
  (derived where the first derivative is zero).
- **peak_yield**: the yield at the time of peak.
- **cumulative_milk_yield**: total yield over a lactation horizon (default 305 days),
  computed as the definite integral of the model over time.
- **persistency**: the average slope after peak until the end of lactation (derived),
  or literature-based formulas (for Wood and MilkBot).

Features
--------
- **Symbolic derivation** (derivative/solve and integration) **cached** per
  (model, characteristic) to avoid recomputation.
- **Fallback numeric methods** for robustness when the symbolic expression is
  not suitable for lambdification or yields invalid values.
- Works with frequentist- or Bayesian-fitted parameters via the
  `lactationcurve.fitting` API.

Notes
-----
- Units: DIM in days, milk in kg or lbs.
- Symbolic expressions can be complex; a light safety check is applied before
  lambdification (`is_valid_sympy_expr`) and large/invalid expressions are rejected.


Author: Meike van Leerdam
Last update: 11-feb-2025
"""

import numpy as np
from sympy import (
    diff,
    exp,
    integrate,
    lambdify,
    ln,
    log,
    nan,
    oo,
    simplify,
    solve,
    symbols,
    zoo,
)

from lactationcurve.fitting import (
    bayesian_fit_milkbot_single_lactation,
    fit_lactation_curve,
    get_lc_parameters,
)
from lactationcurve.preprocessing import validate_and_prepare_inputs


# safe guard for extreme expressions or weird results
def is_valid_sympy_expr(expr) -> bool:
    """Check whether a SymPy expression looks safe to lambdify.

    The validation rejects expressions that contain infinities, NaNs, or an
    excessive number of operations.

    Args:
        expr: A SymPy expression to validate.

    Returns:
        True if the expression appears safe to lambdify; False otherwise.

    Notes:
        - Expressions containing `oo`, `-oo`, `zoo`, or `nan` are rejected.
        - Expressions with `count_ops() > 5000` are rejected as too large/complex.
    """
    try:
        if expr.has(oo, -oo, zoo, nan):
            return False
        # reject absurdly large expressions
        if expr.count_ops() > 5000:
            return False
        return True
    except Exception:
        return False


# Store derived function in cache so they do not need to be calculated all the time again.
_LCC_CACHE = {}


def lactation_curve_characteristic_function(
    model="wood", characteristic=None, lactation_length=305
) -> tuple:
    """Build (or fetch from cache) a symbolic expression and fast numeric function for an LCC.

    This function derives the requested **lactation curve characteristic** for a given
    model using SymPy (derivative / root finding / integration). It returns the symbolic
    expression, the tuple of parameter symbols (argument order), and a lambdified
    numeric function that can be evaluated with numerical parameters.

    The symbolic derivation and integration are done only once per (model, characteristic)
    and then **cached** for reuse.

    Args:
        model (str): Model name. Options:
            'milkbot', 'wood', 'wilmink', 'ali_schaeffer', 'fischer',
            'brody', 'sikka', 'nelder', 'dhanoa', 'emmans', 'hayashi',
            'rook', 'dijkstra', 'prasad'.
        characteristic (str | None): Desired characteristic. Options:
            'time_to_peak', 'peak_yield', 'cumulative_milk_yield', 'persistency'.
            If `None` or unrecognized, a dict of all available characteristics is returned
            (with `persistency` possibly `None` if derivation is not feasible).
        lactation_length (int): Length of lactation in days used in persistency
            computation (default 305).

    Returns:
        tuple:
            expr: SymPy expression (or dict of expressions if `characteristic` is None).
            params: Tuple of SymPy symbols for model parameters (argument order).
            func: Lambdified numeric function `f(*params)` (or dict of functions).

    Raises:
        Exception: If the model is unknown, or if no positive real solution for
            peak timing/yield exists where required.
    """
    # check the cache
    storage = (model, characteristic)
    if storage in _LCC_CACHE:
        return (
            _LCC_CACHE[storage]["expr"],
            _LCC_CACHE[storage]["params"],
            _LCC_CACHE[storage]["func"],
        )

    # make sure model is all lowercase
    model = model.lower()

    # define functions
    if model == "brody":
        # === BRODY 1 ===
        a, b, k1, k2, t = symbols("a b k1 k2 t", real=True, positive=True)
        function = a * exp(-k1 * t) - b * exp(-k2 * t)

    elif model == "sikka":
        # === SIKKA ===
        a, b, c, t = symbols("a b c t", real=True, positive=True)
        function = a * exp(b * t - c * t**2)

    elif model == "fischer":
        # === FISCHER ===
        a, b, c, t = symbols("a b c t", real=True, positive=True)
        function = a - b * t - a * exp(-c * t)

    elif model == "nelder":
        # === NELDER ===
        a, b, c, t = symbols("a b c t", real=True, positive=True)
        function = t / (a + b * t + c * t**2)

    elif model == "wood":
        # === WOOD ===
        a, b, c, t = symbols("a b c t", real=True, positive=True)
        function = a * t**b * exp(-c * t)

    elif model == "dhanoa":
        # === DHANOA ===
        a, b, c, t = symbols("a b c t", real=True, positive=True)
        function = a * t ** (b * c) * exp(-c * t)

    elif model == "emmans":
        # === EMMANS ===
        a, b, c, d, t = symbols("a b c d t")
        function = a * exp(-exp(d - b * t)) * exp(-c * t)

    elif model == "ali_schaeffer":
        # ====ALI=====
        a, b, c, d, k, t = symbols("a b c d k t", real=True, positive=True)
        function = (
            a + b * (t / 305) + c * (t / 305) ** 2 + d * log(t / 305) + k * log((305 / t) ** 2)
        )

    elif model == "wilmink":
        # === WILMINK ===
        a, b, c, k, t = symbols("a b c k t", real=True, positive=True)
        function = a + b * t + c * exp(-k * t)

    elif model == "hayashi":
        # === HAYASHI ===
        a, b, c, d, t = symbols("a b c d t", real=True, positive=True)
        function = b * (exp(-t / c) - exp(-t / (a * c)))

    elif model == "rook":
        # === ROOK ===
        a, b, c, d, t = symbols("a b c d t", real=True, positive=True)
        function = a * (1 / (1 + b / (c + t))) * exp(-d * t)

    elif model == "dijkstra":
        # === DIJKSTRA ===
        a, b, c, d, t = symbols("a b c d t", real=True, positive=True)
        function = a * exp((b * (1 - exp(-c * t)) / c) - d * t)

    elif model == "prasad":
        # === PRASAD ===
        a, b, c, d, t = symbols("a b c d t", real=True, positive=True)
        function = a + b * t + c * t**2 + d / t

    elif model == "milkbot":
        # === MILKBOT ===
        a, b, c, d, t = symbols("a b c d t", real=True, positive=True)
        function = a * (1 - exp((c - t) / b) / 2) * exp(-d * t)

    else:
        raise Exception("Unknown model")

    # find derivative
    fdiff = diff(function, t)
    # solve derivative for when it is zero to find the function for time of peak
    tpeak = solve(fdiff, t)

    # define the end of lactation
    T = lactation_length  # days in milk

    # Persistency = average slope after peak, does not work for all models so therefore try except
    persistency = None
    try:
        if tpeak:
            tmp = (function.subs(t, T) - function.subs(t, tpeak[0])) / (T - tpeak[0])
            tmp = tmp.cancel()  # light simplification
            if is_valid_sympy_expr(tmp):
                persistency = tmp
    except Exception:
        persistency = None

    if characteristic != "cumulative_milk_yield":
        if tpeak:  # Check if the list is not empty
            peak_expr = simplify(function.subs(t, tpeak[0]))
        else:
            raise Exception("No positive real solution for time to peak and peak yield found")

    # find function for cumulative milk yield of the lactation.
    # A lactation length of 305 is the default, but this value can
    # also be provided as a parameter in the characteristic function.
    cum_my_expr = integrate(function, (t, 0, lactation_length))

    # Sorted parameter list (exclude t)
    params = tuple(
        sorted([s for s in function.free_symbols if s.name != "t"], key=lambda x: x.name)
    )

    # ----------------------------------------------------
    # Select requested characteristic
    # ----------------------------------------------------
    if characteristic == "time_to_peak":
        expr = tpeak[0]
    elif characteristic == "peak_yield":
        expr = peak_expr
    elif characteristic == "cumulative_milk_yield":
        expr = cum_my_expr
    elif characteristic == "persistency":
        if persistency is None:
            raise Exception("Persistency could not be computed symbolically")
        expr = persistency
    else:
        # Return all four if None or 'all'
        expr = {
            "time_to_peak": tpeak[0],
            "peak_yield": peak_expr,
            "persistency": persistency,  # possibly None
            "cumulative_milk_yield": cum_my_expr,
        }

    # ----------------------------------------------------
    # Build fast numeric function with lambdify
    # ----------------------------------------------------
    if isinstance(expr, dict):
        func = {
            name: lambdify(params, ex, modules=["numpy", "scipy"])
            for name, ex in expr.items()
            if ex is not None
        }
    else:
        func = lambdify(params, expr, modules=["numpy", "scipy"])

    # ----------------------------------------------------
    # Store in cache
    # ----------------------------------------------------
    _LCC_CACHE[storage] = {"expr": expr, "params": params, "func": func}

    return expr, params, func


def calculate_characteristic(
    dim,
    milkrecordings,
    model="wood",
    characteristic="cumulative_milk_yield",
    fitting="frequentist",
    key=None,
    parity=3,
    breed="H",
    continent="USA",
    custom_priors=None,
    milk_unit="kg",
    persistency_method="derived",
    lactation_length=305,
) -> float:
    """Evaluate a lactation curve characteristic from observed test-day data.

    This function fits the requested model (frequentist or Bayesian via MilkBot),
    retrieves model parameters, and evaluates the requested characteristic using the
    symbolic expression (if available), falling back to numeric methods when needed.

    Args:
        dim (Int): Days in milk (DIM).
        milkrecordings (Float): Milk recordings (kg or lbs) for each DIM.
        model (str): Model name. Supported for this function:
            'milkbot', 'wood', 'wilmink', 'ali_schaeffer', 'fischer'.
        characteristic (str): One of 'time_to_peak', 'peak_yield',
            'cumulative_milk_yield', 'persistency'.
        fitting (str): 'frequentist' (default) or 'bayesian'.
        key (str | None): API key for MilkBot Bayesian fitting.
        parity (Int): Parity of the cow; values above 3 are considered as 3 (Bayesian).
        breed (str): 'H' (Holstein) or 'J' (Jersey) (Bayesian).
        custom_priors: provide your own priors for Bayesian fitting.
            provide as dictionary using build_prior() function
            to create the priors in the right format.
            Alternative use priors from the literature provided by the string command 'CHEN'
        milk_unit: Unit of milk recordings ('kg' or 'lb') for Bayesian fitting.
        continent (str): 'USA' or 'EU' (Bayesian).
        persistency_method (str): 'derived' (average slope after peak; default) or 'literature'
            (only for Wood and MilkBot).
        lactation_length (Int | str): Horizon for persistency calculation:
            305 (default), 'max' (use max DIM in data), or an integer.

    Returns:
        float: The requested characteristic value.

    Raises:
        Exception: If inputs are invalid, the model/characteristic is unsupported,
            an API key is missing for Bayesian fitting, or the characteristic cannot be computed.
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
        persistency_method=persistency_method,
        lactation_length=lactation_length,
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
    persistency_method = inputs.persistency_method
    lactation_length = inputs.lactation_length

    # After validation, these fields are guaranteed non-None for the paths below
    assert model is not None
    assert fitting is not None

    if model not in ["milkbot", "wood", "wilmink", "ali_schaeffer", "fischer"]:
        msg = (
            "this function currently only works for"
            " the milkbot, wood, wilmink,"
            " ali_schaeffer and fischer models"
        )
        raise Exception(msg)

    characteristic_options: list[str] = [
        "time_to_peak",
        "peak_yield",
        "cumulative_milk_yield",
        "persistency",
    ]

    if characteristic in characteristic_options:
        if fitting == "frequentist":
            # Get fitted parameters from your fitting function
            fitted_params = get_lc_parameters(dim, milkrecordings, model)
            assert fitted_params is not None

            if characteristic != "persistency":
                # Try symbolic formula first
                assert isinstance(lactation_length, int)
                expr, params, fn = lactation_curve_characteristic_function(
                    model, characteristic, lactation_length
                )
                with np.errstate(
                    divide="ignore", invalid="ignore"
                ):  # get rid of warnings for invalid operations
                    value = fn(*fitted_params)

                # If symbolic formula fails or is invalid (use numeric approach)
                if (
                    value is None
                    or not np.isfinite(value)
                    or (characteristic == "time_to_peak" and value <= 0)
                ):
                    assert parity is not None
                    assert breed is not None
                    assert continent is not None
                    if characteristic == "time_to_peak":
                        value = numeric_time_to_peak(
                            dim,
                            milkrecordings,
                            model,
                            fitting=fitting,
                            key=key,
                            parity=parity,
                            breed=breed,
                            continent=continent,
                        )
                    elif characteristic == "peak_yield":
                        value = numeric_peak_yield(
                            dim,
                            milkrecordings,
                            model,
                            fitting=fitting,
                            key=key,
                            parity=parity,
                            breed=breed,
                            continent=continent,
                        )
                    elif characteristic == "cumulative_milk_yield":
                        value = numeric_cumulative_yield(
                            dim,
                            milkrecordings,
                            model,
                            fitting=fitting,
                            lactation_length=lactation_length,
                            key=key,
                            parity=parity,
                            breed=breed,
                            continent=continent,
                        )

            else:
                if persistency_method == "derived":
                    # find lactation length from data
                    if lactation_length == "max":
                        lactation_length = max(dim)
                    elif isinstance(lactation_length, int):
                        lactation_length = lactation_length
                    else:
                        lactation_length = 305
                    value = persistency_fitted_curve(
                        dim,
                        milkrecordings,
                        model,
                        fitting="frequentist",
                        lactation_length=lactation_length,
                    )

                else:
                    if model == "wood":
                        value = persistency_wood(fitted_params[1], fitted_params[2])

                    elif model == "milkbot":
                        assert len(fitted_params) >= 4
                        value = persistency_milkbot(fitted_params[3])

                    else:
                        msg = (
                            "Currently only the Wood model"
                            " and MilkBot model have a"
                            " separate model function from"
                            " the literature integrated for"
                            " persistency. if"
                            ' persistency="derived" is'
                            " selected, persistency can be"
                            " calculated for every model as"
                            " the average slope of the"
                            " lactation after the peak."
                        )
                        raise Exception(msg)
            try:
                return float(value)
            except ValueError:
                raise Exception(
                    "Could not compute characteristic, possibly due to invalid fitted parameters"
                )

        else:
            if model == "milkbot":
                if key is None:
                    raise Exception("Key needed to use Bayesian fitting engine MilkBot")
                else:
                    assert parity is not None
                    assert breed is not None
                    assert continent is not None
                    assert isinstance(milk_unit, str)
                    fitted_params_bayes = bayesian_fit_milkbot_single_lactation(
                        dim,
                        milkrecordings,
                        key=key,
                        parity=parity,
                        breed=breed,
                        custom_priors=custom_priors,
                        continent=continent,
                        milk_unit=milk_unit,
                    )
                    fitted_params_bayes = (
                        fitted_params_bayes["scale"],
                        fitted_params_bayes["ramp"],
                        fitted_params_bayes["offset"],
                        fitted_params_bayes["decay"],
                    )

                    if characteristic != "persistency":
                        # Get the symbolic expression and model parameters
                        expr, param_symbols, fn = lactation_curve_characteristic_function(
                            model, characteristic
                        )
                        value = fn(*fitted_params_bayes)

                    else:
                        if persistency_method == "derived":
                            # find lactation length from data
                            if lactation_length == "max":
                                lactation_length = max(dim)
                            elif isinstance(lactation_length, int):
                                lactation_length = lactation_length
                            else:
                                lactation_length = 305
                            assert isinstance(milk_unit, str)
                            value = persistency_fitted_curve(
                                dim,
                                milkrecordings,
                                model,
                                fitting="bayesian",
                                key=key,
                                parity=parity,
                                breed=breed,
                                custom_priors=custom_priors,
                                milk_unit=milk_unit,
                                continent=continent,
                                lactation_length=lactation_length,
                            )

                        else:
                            value = persistency_milkbot(fitted_params_bayes[3])

                    return float(value)
            else:
                raise Exception("Bayesian fitting is currently only implemented for MilkBot models")

    else:
        raise Exception("Unknown characteristic")


# also define numeric approaches as back up if symbolic functions fail or throw invalid results
def numeric_time_to_peak(
    dim,
    milkrecordings,
    model,
    fitting="frequentist",
    key=None,
    parity=3,
    breed="H",
    custom_priors=None,
    milk_unit="kg",
    continent="USA",
) -> int:
    """Compute time to peak using a numeric approach.

    Fits the curve (frequentist or Bayesian), evaluates the predicted yields,
    and returns the DIM corresponding to the maximum predicted yield.

    Args:
        dim: DIM values.
        milkrecordings: Milk recordings (kg).
        model: Model name.
        fitting: 'frequentist' or 'bayesian'.
        key: API key for MilkBot (Bayesian).
        parity: Parity for Bayesian fitting.
        breed: Breed for Bayesian fitting ('H' or 'J').
        custom_priors: provide your own priors for Bayesian fitting.
            provide as dictionary using build_prior() function to set
            the priors in the right format. Alternative use priors from
            the literature provided by the string command 'CHEN'
        milk_unit: Unit of milk recordings ('kg' or 'lb') for Bayesian fitting.
        continent: Prior source for Bayesian ('USA', 'EU').


    Returns:
        int: DIM at which the curve attains its maximum (1-indexed).
    """
    # Fit the curve to get predicted milk yields
    yields = fit_lactation_curve(
        dim,
        milkrecordings,
        model,
        fitting=fitting,
        key=key,
        parity=parity,
        breed=breed,
        custom_priors=custom_priors,
        milk_unit=milk_unit,
        continent=continent,
    )
    # Find the index of the peak yield
    peak_idx = np.argmax(yields)
    # Return the corresponding DIM
    return int(peak_idx + 1)  # +1 because DIM starts at 1, not 0


def numeric_cumulative_yield(
    dim, milkrecordings, model, fitting="frequentist", lactation_length=305, **kwargs
) -> float:
    """Compute cumulative milk yield numerically over a given horizon.

    Adds up the fitted milk yield for the first `lactation_length` days of the
    predicted yield curve.

    Args:
        dim: DIM values.
        milkrecordings: Milk recordings (kg).
        model: Model name.
        fitting: 'frequentist' or 'bayesian'.
        lactation_length: Number of days to integrate (default 305).
        **kwargs: Additional arguments forwarded to `fit_lactation_curve`.

    Returns:
        float: Cumulative milk yield over the specified horizon.
    """
    y = fit_lactation_curve(dim, milkrecordings, model, fitting=fitting, **kwargs)
    return float(np.trapezoid(y[:lactation_length], dx=1))


def numeric_peak_yield(
    dim,
    milkrecordings,
    model,
    fitting="frequentist",
    key=None,
    parity=3,
    breed="H",
    custom_priors=None,
    milk_unit="kg",
    continent="USA",
) -> float:
    """Compute peak yield numerically from the fitted curve.

    Args:
        dim: DIM values.
        milkrecordings: Milk recordings (kg).
        model: Model name.
        fitting: 'frequentist' or 'bayesian'.
        key: API key for MilkBot (Bayesian).
        parity: Parity for Bayesian fitting.
        breed: Breed for Bayesian fitting ('H' or 'J').
        continent: Prior source for Bayesian ('USA', 'EU', 'CHEN').

    Returns:
        float: Maximum predicted milk yield.
    """
    # Fit the curve to get predicted milk yields
    yields = fit_lactation_curve(
        dim,
        milkrecordings,
        model,
        fitting=fitting,
        key=key,
        parity=parity,
        breed=breed,
        custom_priors=custom_priors,
        milk_unit=milk_unit,
        continent=continent,
    )
    # Find the peak yield
    peak_yield = np.max(yields)
    return peak_yield


def persistency_wood(b, c) -> float:
    """Persistency from Wood et al. (1984): `Persistency = -(b + 1) * ln(c)`.

    Args:
        b (float): Parameter `b` of the Wood model.
        c (float): Parameter `c` of the Wood model.

    Returns:
        float: Persistency value from the Wood formula.
    """
    return float(-(b + 1) * ln(c))


def persistency_milkbot(d) -> float:
    """Persistency from the MilkBot model (Ehrlich, 2013): `Persistency = 0.693 / d`.

    Args:
        d (float): Parameter `d` of the MilkBot model.

    Returns:
        float: Persistency value from the MilkBot formula.
    """
    return 0.693 / d


def persistency_fitted_curve(
    dim,
    milkrecordings,
    model,
    fitting="frequentist",
    key=None,
    parity=3,
    breed="H",
    custom_priors=None,
    milk_unit="kg",
    continent="USA",
    lactation_length=305,
) -> float:
    """Persistency as the average slope after peak until end of lactation (numeric).

    This is the default approach because symbolic derivation is not feasible for
    all models. It computes:
        `(yield_at_end - yield_at_peak) / (lactation_length - time_to_peak)`

    Args:
        dim: DIM values.
        milkrecordings: Milk recordings (kg).
        model: Model name. Options include 'milkbot' (Bayesian or frequentist),
            'wood', 'wilmink', 'ali_schaeffer', 'fischer'.
        fitting: 'frequentist' or 'bayesian'.
        key: API key (only for Bayesian fitting).
        parity: Parity of the cow; values above 3 treated as 3 (Bayesian).
        breed: 'H' or 'J' (Bayesian).
        custom_priors: provide your own priors for Bayesian fitting.
            provide as dictionary using build_prior() function to
            create the priors in the right format. Alternative use
            priors from the literature provided by the string
            command 'CHEN'
        milk_unit: Unit of milk recordings ('kg' or 'lb') for Bayesian fitting.
        continent: 'USA', 'EU', or 'CHEN' (Bayesian).
        lactation_length (int | str): 305 (default), 'max' (use max DIM), or integer.

    Returns:
        float: Average slope after the peak until end of lactation.
    """
    # calculate time to peak
    t_peak = numeric_time_to_peak(
        dim,
        milkrecordings,
        model,
        fitting=fitting,
        key=key,
        parity=parity,
        breed=breed,
        custom_priors=custom_priors,
        milk_unit=milk_unit,
        continent=continent,
    )

    # determine lactation length
    if lactation_length == "max":
        lactation_length = max(dim)
    elif isinstance(lactation_length, int):
        lactation_length = lactation_length
    else:
        lactation_length = 305

    # calculate milk yield at peak
    yields = fit_lactation_curve(
        dim,
        milkrecordings,
        model,
        fitting=fitting,
        key=key,
        parity=parity,
        breed=breed,
        custom_priors=custom_priors,
        milk_unit=milk_unit,
        continent=continent,
    )
    peak_yield = yields[int(t_peak) - 1]  # -1 to prevent index error
    # calculate milk yield at end of lactation
    end_yield = yields[int(lactation_length) - 1]  # -1 to prevent index error

    # calculate persistency
    persistency = (end_yield - peak_yield) / (lactation_length - t_peak)
    return persistency
