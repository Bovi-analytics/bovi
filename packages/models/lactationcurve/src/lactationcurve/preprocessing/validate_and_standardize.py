"""
# Utility functions
Input validation and tabular schema normalization for lactation curve workflows.

This module has two small utilities that are used by
`lactationcurve.fitting.lactation_curve_fitting` and by
'lactationcurve.characteristics.lactation_curve_characteristics'
to ensure consistent input handling:

1) `validate_and_prepare_inputs` consolidates routine checks for DIM and test‑day
   milk records, normalizes optional options (e.g., fitting method, breed, priors),
   drops rows with missing or non‑finite values, and returns a structured `PreparedInputs`
   bundle. This keeps the core fitting and characteristic functions focused on their main logic,
   and ensures that all inputs are clean and consistent.

2) `standardize_lactation_columns` aligns a flexible DataFrame schema to a small,
   canonical set of column names (`DaysInMilk`, `MilkingYield`, `TestId`) and trims
   records outside a user‑defined DIM horizon. This is handy prior to 305‑day
   calculations and when users provide varied source column names. (currently not yet implemented)

Design goals:
- Keep pre‑flight checks and schema handling **centralized** so model‑fitting and
  characteristic functions can assume clean, typed inputs.
- Keep behavior predictable across modules without hard‑coding assumptions in the
  fitting code.

Conventions:
- DIM is in days; milk yield is in kg or lb.


Author: Meike van Leerdam
Last update: 13 feb 2026
"""

from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
import pandas as pd


class ParameterPrior(TypedDict):
    mean: float
    sd: float


class MilkBotPriors(TypedDict):
    scale: ParameterPrior
    ramp: ParameterPrior
    decay: ParameterPrior
    offset: ParameterPrior
    seMilk: float


@dataclass
class PreparedInputs:
    """Normalized, ready‑to‑fit inputs.

    This container is returned by `validate_and_prepare_inputs` and is the single
    hand‑off object expected by the fitting routines. Arrays are finite and 1‑dimensional;
    categorical fields are lower/upper‑cased as appropriate and may be `None` if omitted.

    Attributes:
        dim: 1D NumPy array of day‑in‑milk values (finite; same length as `milkrecordings`).
        milkrecordings: 1D NumPy array of test‑day milk yields aligned to `dim`.
        model: Lowercased model identifier or `None` if not provided.
        fitting: `"frequentist"` or `"bayesian"` (lowercased) or `None`.
        breed: `"H"` or `"J"` or `None`.
        parity: Lactation number as `int`, if provided; otherwise `None`.
        continent: Prior source for MilkBot API (`"USA"`, `"EU"`), or `None`.
        persistency_method: Either `"derived"` or `"literature"`, or `None`.
        lactation_length: Integer horizon (e.g., 305), the string `"max"`, or `None`.
        milk_unit: Either `"kg"` or `"lb"`, defaulting to `"kg"`.
        custom_priors: Either a dict of priors, the string `"CHEN"` to use Chen et al. priors,
            or `None` if not provided.
    """

    dim: np.ndarray
    milkrecordings: np.ndarray
    model: str | None = None
    fitting: str | None = None
    breed: str | None = None
    parity: int | None = None
    continent: str | None = None
    persistency_method: str | None = None
    lactation_length: int | str | None = None
    milk_unit: str | None = None
    custom_priors: MilkBotPriors | str | None = None


def validate_and_prepare_inputs(
    dim,
    milkrecordings,
    model=None,
    fitting=None,
    *,
    breed=None,
    parity=None,
    continent=None,
    persistency_method=None,
    lactation_length=None,
    milk_unit="kg",
    custom_priors=None,
) -> PreparedInputs:
    """
    Validate, normalize, and clean input data for lactation curve fitting.

    This function performs basic consistency checks on the provided
    days-in-milk (DIM) and milk recording data, normalizes optional
    categorical parameters, and removes observations with missing or
    non-finite values. The cleaned and validated inputs are returned
    in a structured :class:`PreparedInputs` object.

    Parameters
    ----------
    dim : array-like
        Days in milk corresponding to each milk recording.
    milkrecordings : array-like
        Milk yield measurements corresponding to `dim`.
    model : str or None, optional
        Name of the lactation curve model. If provided, the name is
        stripped of whitespace and converted to lowercase.
    fitting : str or None, optional
        Fitting approach to be used. Must be either ``"frequentist"``
        or ``"bayesian"`` if provided.
    breed : str or None, optional
        Cow breed identifier. Must be ``"H"`` (Holstein) or ``"J"``
        (Jersey) if provided. Case-insensitive.
    parity : int or None, optional
        Lactation number (parity). If provided, it is coerced to an
        integer.
    continent : str or None, optional
        Geographic region identifier. Must be one of ``"USA"`` or
        ``"EU"`` if provided. Case-insensitive.
    milk_unit : str, optional
        Unit of milk yield measurements. Must be either ``"kg"`` or ``"lb"``. Default is ``"kg"``.
    custom_priors : dict or str or None, optional
        Custom prior distributions for Bayesian fitting. If a dict is provided,
        it must be a dictionary of prior distributions for each parameter in the model.
        If the string ``"CHEN"`` is provided, the default Chen et al. priors are used.

    Extra input for persistency calculation:
        persistency_method (String): way of calculating
            persistency, options: 'derived' which gives the
            average slope of the lactation after the peak until
            the end of lactation (default) or 'literature' for
            the wood and milkbot model.
        Lactation_length: string or int: length of the lactation
            in days to calculate persistency over, options:
            305 = default or 'max' uses the maximum DIM in the
            data, or an integer value to set the desired
            lactation length.

    Returns
    -------
    PreparedInputs
        A dataclass containing the cleaned numeric arrays (`dim`,
        `milkrecordings`) and the normalized optional parameters.

    Raises
    ------
    ValueError
        If input arrays have different lengths, contain insufficient
        valid observations, or if categorical parameters are invalid.

    Notes
    -----
    Observations with missing or non-finite values in either `dim` or
    `milkrecordings` are removed prior to model fitting. At least two
    valid observations are required to proceed.
    """
    if len(dim) != len(milkrecordings):
        raise ValueError("dim and milkrecordings must have the same length")

    model = model.strip().lower() if model else None

    if parity is not None:
        parity = int(parity)

    if fitting is not None:
        fitting = fitting.lower()
        if fitting not in {"frequentist", "bayesian"}:
            raise ValueError("Fitting method must be either frequentist or bayesian")

    if breed is not None:
        breed = breed.upper()
        if breed not in {"H", "J"}:
            raise ValueError("Breed must be either Holstein = 'H' or Jersey 'J'")

    if continent is not None:
        continent = continent.upper()
        if continent not in {"USA", "EU"}:
            raise ValueError("continent must be 'USA' or 'EU'")

    dim = np.asarray(dim, dtype=float)
    milkrecordings = np.asarray(milkrecordings, dtype=float)

    mask = np.isfinite(dim) & np.isfinite(milkrecordings)
    dim = dim[mask]
    milkrecordings = milkrecordings[mask]

    if len(dim) < 2:
        raise ValueError("At least two non missing points are required to fit a lactation curve")

    if persistency_method is not None:
        persistency_method = persistency_method.lower()
        if persistency_method not in {"derived", "literature"}:
            raise ValueError("persistency_method must be either 'derived' or 'literature'")

    if lactation_length is not None:
        if isinstance(lactation_length, str):
            if lactation_length.lower() != "max":
                raise ValueError("lactation_length string option must be 'max'")
        else:
            lactation_length = int(lactation_length)

    if milk_unit not in ["kg", "lb"]:
        raise ValueError("milk_unit must be 'kg' or 'lb'")

    if custom_priors is not None and not isinstance(custom_priors, (dict, str)):
        raise ValueError("custom_priors must be a dict, a string, or None")

    if isinstance(custom_priors, str):
        custom_priors = custom_priors.upper()
        if custom_priors != "CHEN":
            raise ValueError(
                "custom_priors string option must be"
                " 'CHEN', self defined priors can be"
                " provided as a dictionary through"
                " the build_prior function"
            )

    if isinstance(custom_priors, dict):
        custom_priors = cast(MilkBotPriors, custom_priors)

    return PreparedInputs(
        dim=dim,
        milkrecordings=milkrecordings,
        model=model or None,
        fitting=fitting,
        breed=breed,
        parity=parity,
        continent=continent,
        persistency_method=persistency_method,
        lactation_length=lactation_length,
        milk_unit=milk_unit,
        custom_priors=custom_priors,
    )


def standardize_lactation_columns(
    df: pd.DataFrame,
    *,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    default_test_id=0,
    max_dim: int = 305,
) -> pd.DataFrame:
    """
    Standardize column names and structure for lactation data.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with standardized columns:
        - DaysInMilk
        - MilkingYield
        - TestId
    """

    df = df.copy()

    # Accepted aliases (case-insensitive)
    aliases = {
        "DaysInMilk": ["daysinmilk", "dim", "testday"],
        "MilkingYield": [
            "milkingyield",
            "testdaymilkyield",
            "milkyield",
            "yield",
            "milkproduction",
            "milk_yield",
        ],
        "TestId": ["testid", "animalid", "id"],
    }

    # Lowercase lookup → actual column name
    col_lookup = {col.lower(): col for col in df.columns}

    def resolve_col(override, possible_names):
        if override:
            return col_lookup.get(override.lower())
        for name in possible_names:
            if name in col_lookup:
                return col_lookup[name]
        return None

    # Resolve columns
    dim_col = resolve_col(days_in_milk_col, aliases["DaysInMilk"])
    if not dim_col:
        raise ValueError("No DaysInMilk column found.")

    yield_col = resolve_col(milking_yield_col, aliases["MilkingYield"])
    if not yield_col:
        raise ValueError("No MilkingYield column found.")

    id_col = resolve_col(test_id_col, aliases["TestId"])

    # Create TestId if missing
    if not id_col:
        df["TestId"] = default_test_id
        id_col = "TestId"

    # Rename to standardized names
    df = df.rename(
        columns={
            dim_col: "DaysInMilk",
            yield_col: "MilkingYield",
            id_col: "TestId",
        }
    )

    # Filter DIM
    df = pd.DataFrame(df[df["DaysInMilk"] <= max_dim])

    return df
