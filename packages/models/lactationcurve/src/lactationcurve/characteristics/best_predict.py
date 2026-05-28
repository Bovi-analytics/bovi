"""Best prediction for cumulative 305-day milk yield.

Purpose
-------
This module implements the best-prediction approach described by VanRaden
(1997) for ICAR Procedure 2, Section 2 (computing accumulated lactation
yield).

Method Summary
--------------
Adapted from the Best Predict Manual by Cole and VanRaden (2015)

Best prediction combines a population-level standard lactation curve with
correlation-based corrections derived from observed test-day deviations.
The method projects observed deviations onto the full 305-day curve using a
covariance structure estimated from reference data.

Individual daily yield can be modeled as the expected value of a management
group plus a deviation from that mean:
yi = E(yi) + ti
where yi
is an individual yield on test day i, E(yi) is the expected yield for an
animal in the same management group
(Wiggans, Misztal, and Van Vleck 1988) on the same test day, and ti
is a deviation from the group mean on the same
test day. Suppose that µ is a vector of expected values for each day of
lactation for a single trait, t is a vector of 305
test day deviations for the trait, and tm is a vector of only the measured
deviations. The means and variances of t
and tm are assumed known with V (t) = V and V (tm) = Vm. The covariance
between t and tm, C, is also assumed
known.

**Lactation yield**: A cow’s true 305-d yield (y) is the sum of the expected
values for each day (1′µ) plus the sum of her
305 deviations from expectations (1′t), where 1′
is a vector of 1s of length 305. The cow’s true yield, and the best
prediction of that yield (yˆ), are:
y = 1′t
yˆ = 1′CVm^−1 * tm


Key Entry Points
----------------
- ``best_predict_method``: Apply best prediction per ``TestId``.
- ``best_predict_method_single_lac``: Predict one lactation.
- ``fit_autocorrelation_matrix``: Fit covariance structure from reference data.

Column Flexibility
------------------
The functions accept several case-insensitive column name aliases and can
create a default ``TestId`` if one is missing. Recognized aliases:

- Days in Milk: `["daysinmilk", "dim", "testday"]`
- Milk Yield: `["milkingyield", "testdaymilkyield", "milkyield", "yield"]`
- Test Id: `["animalid", "testid", "id"]`

It is also possible to provide your own column names so the function
can be applied to dataframes with different column naming conventions.

Defaults
--------
- ``STANDARD_CURVE``: Baseline expected lactation curve for days 1..305 (Wood).
- ``COV_MATRIX``: Default day-to-day covariance structure used for projection.

Notes
-----
- The default assets are loaded from the package ``data`` directory.
- Users can fit curve and covariance ingredients from their own reference
    population.
- The method can be applied to lactations without any measurements,
    in which case the result will be the population mean from the standard curve.
- Currently it is not yet possible to predict lactation yields for lactation windows
    other than 305 days, but this is on the roadmap for future updates.
- The method currently assumes that the standard curves and covariance structure is the same
    for all lactations,
    but future updates may allow using different standard curves and covariance structures for
    different subgroups of lactations (e.g., by breed or parity).
- For the used standard lactation curve currently the Wood LC model is used, it is possible to
    use other methods aswell.
- Strengths of Best Predction includes its ability to leverage the full covariance structure
    of the lactation curve and to therefore potentially provide more accurate predictions
    especially for lactations with few test days. This is because the inherent shape of the
    lactation curve is taken into account in the projection of
    observed deviations to unobserved days.
    It has fewer in between steps then ISLC and is therefore easier to use.
- Disadvantages of Best Prediction include its computational intensity,
    especially when fitting the covariance structure from data.
    And the method is not as easy to understand
    as a simpler method such as the test interval method,
    which can make it less transparent to users.
    The best results are obtained when the standard curve and covariance matrix
    are from the same population as the data,
    which can be a barrier for users without access to a large reference dataset.
    This also causes inconsistencies in cumulative milk yield results
    depending on which standard curves are used.
    A cow with the exact same test-day records
    can have a different cumulative milk yield estimates depending
    on the standard curve used, which can be considered unfair.

References
---------
VanRaden, P. M. (1997). Lactation yields and accuracies computed from test
day yields and (co) variances by best prediction.
Journal of dairy science, 80(11), 3015-3022.

A Manual for Use of BESTPRED: A Program for Estimation of Lactation Yield
and Persistency Using Best Prediction
Release 2.0 rc 7
J. B. Cole and P. M. VanRaden
August 12, 2009
Revised April 27, 2015
Animal Genomics and Improvement Laboratory, Agricultural Research Service, United States
Department of Agriculture, Room 306 Bldg 005 BARC-West, 10300 Baltimore Avenue,
Beltsville, MD 20705-2350

Original code for best predict can be found on [GitHub](https://github.com/wintermind/bestpred)

Author: Meike van Leerdam,
Date: 24-04-2026
Last update: 21-May-2026
"""

import inspect
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


# Lightweight repr-only wrapper for cleaner generated signatures in docs
class _DocDefault:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - docs-only
        return self.label


_DOC_STANDARD_CURVE = _DocDefault("STANDARD_CURVE")
_DOC_COV_MATRIX = _DocDefault("COV_MATRIX")

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


def fit_standard_lc(df: pd.DataFrame, lc_model: str = "Wood") -> np.ndarray:
    """Fit a population-level standard lactation curve.

    The curve is fit with the package's frequentist Wood model and returned on
    the fixed day-1..305 grid.

    Args:
        df: Reference dataframe containing ``DaysInMilk`` and ``MilkingYield``.
        lc_model: The lactation curve model to fit.
        Default is the "Wood" lactation curve model.

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
            model=lc_model,
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

    By default this function uses the package-provided standard curve and covariance matrix.
    But it is also possible to provide your own standard curve and covariance matrix,
    for example when you want to fit these ingredients from your own reference population.


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

    By default this function uses the package-provided standard curve and covariance matrix.
    But it is also possible to provide your own standard curve and covariance matrix,
    for example when you want to fit these ingredients from your own reference population.
    This can be done in two ways: either by fitting the covariance matrix and standard curve
    directly from a reference dataset by providing a pandas dataframe at 'reference_df ='
    when ``fit_standard_lc_from_data`` is True.
    Alternative for customization is to set standard_lc_305 and covariance_matrix
    directly in the function call.

    Args:
        df: Input observations. If ``TestId`` is missing, all rows are treated
            as one lactation.
        standard_lc: Expected daily milk yield lactation curve on days 1..305.
            If not provided, the package's default curve is used.
            Or fit your own standard curve from a reference dataset by
            providing a pandas dataframe at 'reference_df ='
            when ``fit_standard_lc_from_data`` is True.
        days_in_milk_col: Optional input column name for days in milk. If
            provided, it is mapped to ``DaysInMilk``.
        milking_yield_col: Optional input column name for milk yield. If
            provided, it is mapped to ``MilkingYield``.
        test_id_col: Optional input column name for lactation/test identifier.
            If provided, it is mapped to ``TestId``.
        default_test_id: Fallback test id used when no test-id column is
            available.
        covariance_matrix: Optional prefit covariance matrix. If omitted,
            the default matrix is used or
             ``reference_df`` can be used to fit one for your own data.
        fit_standard_lc_from_data: Whether to fit covariance information from
            ``reference_df`` instead of using a provided covariance matrix.
        reference_df: Reference dataframe used when ``covariance_matrix`` and
            ``standard_lc`` are not provided and ``fit_standard_lc_from_data``
            is True.

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

    return pd.DataFrame(results)


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


def _set_doc_signatures() -> None:
    """Override displayed defaults in docs without changing runtime behavior."""
    doc_defaults = {
        "standard_lc": _DOC_STANDARD_CURVE,
        "covariance_matrix": _DOC_COV_MATRIX,
    }

    for func in (best_predict_method_single_lac, best_predict_method):
        signature = inspect.signature(func)
        params = [
            param.replace(default=doc_defaults[param.name]) if param.name in doc_defaults else param
            for param in signature.parameters.values()
        ]
        func.__signature__ = signature.replace(parameters=params)


_set_doc_signatures()

if __name__ == "__main__":
    demo()
