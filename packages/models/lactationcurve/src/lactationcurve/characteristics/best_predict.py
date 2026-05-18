"""Best prediction for cumulative 305-day milk yield.

Purpose
-------
This module implements the best-prediction approach described by VanRaden
(1997) for ICAR Procedure 2, Section 2 (computing accumulated lactation
yield).

Method Summary
--------------
Best prediction combines a population-level standard lactation curve with
correlation-based corrections derived from observed test-day deviations.
The method projects observed deviations onto the full 305-day curve using a
covariance structure estimated from reference data.

Key Entry Points
----------------
- ``best_predict_method``: Apply best prediction per ``TestId``.
- ``best_predict_method_single_lac``: Predict one lactation.
- ``fit_autocorrelation_matrix``: Fit covariance structure from reference data.

Defaults
--------
- ``STANDARD_CURVE``: Baseline expected lactation curve for days 1..305 (Wood).
- ``COV_MATRIX``: Default day-to-day covariance structure used for projection.

Notes
-----
- The default assets are loaded from the package ``data`` directory.
- Users can fit curve and covariance ingredients from their own reference
    population.

Author: Meike van Leerdam,
Date: 24-04-2026
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize

from lactationcurve.fitting import fit_lactation_curve
from lactationcurve.preprocessing import standardize_lactation_columns

# get the standard lactation curve ingredients back from the data storage:
DATA_DIR = Path(__file__).resolve().parent / "data"
COV_MATRIX = np.load(DATA_DIR / "covariance_matrix_best_predict.npy")
STANDARD_CURVE = np.load(DATA_DIR / "standard_lc_wood.npy")

# functions to fit you own standard curve and covariance matrix


def pivot_milk_recordings_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert long-format recordings to a fixed 305-day matrix.

    Rows represent lactations (``TestId``) and columns represent days in milk
    from 1 through 305. Missing observations are kept as ``NaN``.

    Args:
        df: Dataframe with ``TestId``, ``DaysInMilk``, and ``MilkingYield``.

    Returns:
        A NumPy array of shape ``(n_lactations, 305)``.
    """
    # ensure sorting
    df = df.sort_values(["TestId", "DaysInMilk"])

    # pivot to wide format
    milk_recordings_pivot = df.pivot_table(
        index="TestId", columns="DaysInMilk", values="MilkingYield"
    )

    # enforce fixed 305-day grid alignment used by best-prediction
    milk_recordings_pivot = milk_recordings_pivot.reindex(columns=range(1, 306))

    # convert to numpy matrix
    Y = milk_recordings_pivot.to_numpy()
    return Y


def fit_standard_lc(df: pd.DataFrame) -> np.ndarray:
    """Fit a population-level standard lactation curve.

    The curve is fit with the package's frequentist Wood model and returned on
    the fixed day-1..305 grid.

    Args:
        df: Reference dataframe containing ``DaysInMilk`` and ``MilkingYield``.

    Returns:
        A NumPy array of expected daily milk yield for days 1..305.

    Notes:
        This mean curve acts as the baseline in best prediction. Individual
        lactations are represented as deviations around this population profile.
    """
    standard_lc = pd.Series(
        fit_lactation_curve(
            df["DaysInMilk"].values,
            df["MilkingYield"].values,
            model="wood",
            fitting="frequentist",
        ),
        index=range(1, 306),
    )

    return standard_lc.to_numpy(dtype=float)


def center_lactation_data(
    milk_matrix: np.ndarray,
    standard_lc: np.ndarray,
    day_mean_method: str = "standard_lc",
) -> np.ndarray:
    """Center lactation yields before covariance estimation.

    Args:
        milk_matrix: Yield matrix with lactations in rows and days in columns.
        standard_lc: Expected day-wise milk yield profile.
        day_mean_method: Mean-centering strategy. Supported values are
            ``"standard_lc"`` (default) and ``"data"``.

    Returns:
        A centered matrix with the same shape as ``milk_matrix``.

    Raises:
        ValueError: If ``day_mean_method`` is not supported.
    """
    if day_mean_method == "standard_lc":
        day_mean = standard_lc
    elif day_mean_method == "data":
        day_mean = np.nanmean(milk_matrix, axis=0)
    else:
        raise ValueError("day_mean_method must be 'standard_lc' or 'data'.")

    return milk_matrix - day_mean


def build_covariance_matrix(rho: float, size: int) -> np.ndarray:
    """Construct a covariance matrix.

    Cole et al. (2007) estimated correlations among test-day yields using a
    simplified model with an identity matrix (I) for daily measurement error
    and an autoregressive matrix (E) for biological change. E is defined as
    ``Eij = r ** |i-j|`` where ``i`` and ``j`` are test-day DIM and
    ``0 < r < 1``.

    Element ``(i, j)`` is ``rho ** abs(i - j)``.

    Args:
        rho: AR(1) correlation parameter.
        size: Matrix dimension.

    Returns:
        A ``(size, size)`` AR(1) correlation matrix.
    """
    idx = np.arange(size)
    M = np.abs(idx[:, None] - idx[None, :])
    return rho**M


def fit_autocorrelation_matrix(
    df: pd.DataFrame, standard_lc: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Estimate covariance parameters for best prediction.

    The model is ``B = b1 * I + b2 * E`` where ``E`` is an AR(1) correlation
    matrix. Parameters are optimized in transformed space and mapped back to
    enforce ``b1 > 0``, ``b2 > 0``, and ``0 < rho < 1``.

    Args:
        df: Reference milk-recording dataframe.
        standard_lc: Population mean curve used for centering.

    Returns:
        Dictionary with:
        - ``"B_hat"``: fitted covariance matrix.
        - ``"R_hat"``: correlation matrix derived from ``B_hat``.
        - ``"b1"``, ``"b2"``, ``"rho"``: fitted scalar parameters.
    """
    milk_matrix = pivot_milk_recordings_to_matrix(df)
    centered_matrix = center_lactation_data(milk_matrix, standard_lc)
    n_lactations, n_days = centered_matrix.shape
    observed_indices = [np.where(~np.isnan(centered_matrix[i]))[0] for i in range(n_lactations)]

    def negative_log_likelihood(params: np.ndarray) -> float:
        p_b1, p_b2, p_rho = params
        b1 = float(np.exp(p_b1))
        b2 = float(np.exp(p_b2))
        rho = float(1 / (1 + np.exp(-p_rho)))  # now rho in (0,1)
        correlation_matrix = build_covariance_matrix(rho, n_days)

        total = 0.0
        for lactation_idx, day_indices in enumerate(observed_indices):
            observation_count = len(day_indices)
            if observation_count == 0:
                continue

            observations = centered_matrix[lactation_idx, day_indices]
            correlation_subset = correlation_matrix[np.ix_(day_indices, day_indices)]
            sigma = b1 * np.eye(observation_count) + b2 * correlation_subset

            # Numerical safeguards: try Cholesky and penalize non-PD parameters.
            try:
                cholesky_factor, lower = cho_factor(sigma, check_finite=False)
                solution = cho_solve((cholesky_factor, lower), observations, check_finite=False)
            except LinAlgError:
                # penalty for non-PD
                return float(1e12 + np.sum(np.abs(params)))

            quadratic_form = float(observations @ solution)
            log_determinant = 2.0 * np.sum(np.log(np.diag(cholesky_factor)))
            total += 0.5 * (
                log_determinant + quadratic_form + observation_count * np.log(2 * np.pi)
            )

        # return total negative log-likelihood
        return float(total)

    # initial guesses and optimization. A 50/50 split in variance is assumed as starting point
    initial_variance = max(float(np.nanvar(centered_matrix)), 1e-6)
    initial_params = [
        np.log(0.5 * initial_variance),
        np.log(0.5 * initial_variance),
        0.5,
    ]

    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-8},
    )

    if not result.success:
        print(f"Optimization warning: {result.message}")

    log_b1_hat, log_b2_hat, logit_rho_hat = result.x
    b1_hat = float(np.exp(log_b1_hat))
    b2_hat = float(np.exp(log_b2_hat))
    rho_hat = float(1 / (1 + np.exp(-logit_rho_hat)))
    correlation_matrix = build_covariance_matrix(rho_hat, n_days)
    covariance_matrix = b1_hat * np.eye(n_days) + b2_hat * correlation_matrix

    # convert to correlation matrix
    std = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std, std)

    return {
        "B_hat": covariance_matrix,
        "R_hat": correlation_matrix,
        "b1": b1_hat,
        "b2": b2_hat,
        "rho": rho_hat,
    }


# Functions for best predict that also work with the provided standard curve and covariance matrix.


def preprocess_measured_data(lactation: pd.DataFrame, standard_lc: np.ndarray) -> pd.Series:
    """Build a 305-day deviation vector for a single lactation.

    For observed days, this computes ``MilkingYield - standard_lc[day]``.
    The result is reindexed to days 1..305 with unobserved days filled as zero.

    Args:
        lactation: Single-lactation dataframe with ``DaysInMilk`` and
            ``MilkingYield``.
        standard_lc: Expected daily milk yield profile.

    Returns:
        A Series indexed by day 1..305 containing milk-yield deviations.
    """

    # calculate the difference between the expected (population mean) and measured milk yield

    # extract the expected milk yields for the measured DaysInMilk in the df
    day_idx = lactation["DaysInMilk"].to_numpy(dtype=int) - 1
    expected = np.asarray(standard_lc, dtype=float)[day_idx]

    # Subtract
    lactation["MilkDifference"] = lactation["MilkingYield"].to_numpy(dtype=float) - expected

    # Create a Series of length 305 with missing values = 0
    milk_difference = cast(pd.Series, lactation.set_index("DaysInMilk")["MilkDifference"])
    corrected_series = milk_difference.reindex(range(1, 306), fill_value=0)

    return corrected_series


def best_predict_method_single_lac(
    lactation: pd.DataFrame,
    standard_lc: np.ndarray = STANDARD_CURVE,
    covariance_matrix: np.ndarray = COV_MATRIX,
) -> float:
    """Predict 305-day cumulative yield for one lactation.

    Observed test-day deviations are projected over all 305 days using the
    covariance structure and then added to the baseline cumulative standard
    curve.

    Args:
        lactation: Observed records for one lactation.
        standard_lc: Population mean daily yield profile.
        covariance_matrix: Day-to-day covariance matrix on the 305-day grid.

    Returns:
        Predicted cumulative 305-day milk yield.

    Notes:
        Duplicate day records are resolved with ``keep="last"`` before
        prediction. If no valid observations remain in days 1..305, the method
        returns the cumulative standard curve.
    """
    filtered_lactation = lactation.loc[
        (lactation["DaysInMilk"] >= 1) & (lactation["DaysInMilk"] <= 305)
    ].copy()
    filtered_lactation = filtered_lactation.drop_duplicates(subset=["DaysInMilk"], keep="last")
    filtered_lactation = filtered_lactation.sort_values("DaysInMilk")

    corrected_series = preprocess_measured_data(
        filtered_lactation,
        standard_lc=standard_lc,
    )

    if filtered_lactation.empty:
        return float(np.sum(standard_lc))

    obs_idx_1based = filtered_lactation["DaysInMilk"].to_numpy(dtype=int)  # DaysInMilk: 1-305
    obs_idx_0based = obs_idx_1based - 1  # Convert to 0-based matrix indices: 0-304
    y_obs = corrected_series.loc[obs_idx_1based].to_numpy(
        dtype=float
    )  # corrected_series is indexed by DaysInMilk (1-305)

    # Extract covariance blocks
    B_oo = covariance_matrix[
        np.ix_(obs_idx_0based, obs_idx_0based)
    ]  # Use 0-based indices for matrix
    B_mo = covariance_matrix[:, obs_idx_0based]  # Use 0-based indices for matrix

    # solve
    c, lower = cho_factor(B_oo)
    alpha = cho_solve((c, lower), y_obs)

    # Predict full deviation curve
    y_estimate = B_mo @ alpha

    # Total milk = baseline + deviation
    deviation = np.sum(y_estimate)

    total = np.sum(standard_lc) + deviation

    return total


def best_predict_method(
    df: pd.DataFrame,
    standard_lc: np.ndarray = STANDARD_CURVE,
    days_in_milk_col: str | None = None,
    milking_yield_col: str | None = None,
    test_id_col: str | None = None,
    default_test_id: int = 0,
    covariance_matrix: np.ndarray | None = COV_MATRIX,
    fit_standard_lc_from_data: bool = False,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply best prediction to one or more lactations.

    Args:
        df: Input observations. If ``TestId`` is missing, all rows are treated
            as one lactation.
        standard_lc: Expected daily milk yield profile on days 1..305.
        covariance_matrix: Optional prefit covariance matrix. If omitted,
            ``reference_df`` is used to fit one.
        reference_df: Reference dataframe used when ``covariance_matrix`` is
            not provided.

    Returns:
        Dataframe with columns ``TestId`` and ``LactationMilkYield``.

    Raises:
        ValueError: If neither ``covariance_matrix`` nor ``reference_df`` is
            provided.
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

    # Fit covariance if not provided
    if fit_standard_lc_from_data:
        if reference_df is None:
            raise ValueError("Provide reference_df to fit your own standard lactation curve.")
        reference_df = standardize_lactation_columns(
            reference_df,
            days_in_milk_col=days_in_milk_col,
            milking_yield_col=milking_yield_col,
            test_id_col=test_id_col,
            default_test_id=default_test_id,
            max_dim=305,
        )
        covariance_matrix = cast(
            np.ndarray, fit_autocorrelation_matrix(reference_df, standard_lc)["B_hat"]
        )

    covariance_matrix_array = cast(np.ndarray, covariance_matrix)

    df = df.copy()

    results = []

    for test_id, lactation in df.groupby("TestId"):
        pred = best_predict_method_single_lac(
            lactation,
            standard_lc,
            covariance_matrix_array,
        )
        results.append({"TestId": test_id, "LactationMilkYield": pred})

    return pd.DataFrame(results, columns=pd.Index(["TestId", "LactationMilkYield"]))


# demo function so I can see if this script runs as expected


def demo() -> None:
    """Run a minimal example of best prediction with mock data."""

    # --- Single + multiple lactations example ---
    test_df = pd.DataFrame(
        {
            "TestId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "DaysInMilk": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
            "MilkingYield": [30, 35, 40, 38, 36, 28, 33, 37, 39, 34],
        }
    )

    result_cov = best_predict_method(
        test_df, standard_lc=STANDARD_CURVE, covariance_matrix=COV_MATRIX
    )

    print("Predictions with provided covariance matrix:")
    print(result_cov)


if __name__ == "__main__":
    demo()
