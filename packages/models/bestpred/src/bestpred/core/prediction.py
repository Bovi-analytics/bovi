"""Prediction linear algebra helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PredictionLinearResult:
    """Result of `covariance_to_targets * inv(observation_covariance) * deviations`."""

    weights: FloatArray
    predictions: FloatArray
    reliability_covariance: FloatArray


def solve_prediction_system(
    *,
    covariance_to_targets: FloatArray,
    observation_covariance: FloatArray,
    deviations: FloatArray,
    covariance_to_targets_transposed: FloatArray | None = None,
) -> PredictionLinearResult:
    """Compute the core BESTPRED matrix products without explicit inversion.

    Fortran calls `invrt2(var, ...)` and then multiplies:

    - `covar = cov * var`
    - `vari = covar * covp`
    - `multi/single = covar * dev`

    Here `var` is kept as the original observation covariance matrix and solved
    with `np.linalg.solve`.
    """

    _validate_matrix_shapes(covariance_to_targets, observation_covariance, deviations)
    covp = (
        covariance_to_targets.T
        if covariance_to_targets_transposed is None
        else covariance_to_targets_transposed
    )

    solved = np.asarray(np.linalg.solve(observation_covariance, deviations), dtype=np.float64)
    predictions = np.asarray(covariance_to_targets @ solved, dtype=np.float64)
    weights = np.asarray(
        np.linalg.solve(observation_covariance.T, covariance_to_targets.T).T,
        dtype=np.float64,
    )
    reliability_covariance = np.asarray(weights @ covp, dtype=np.float64)

    return PredictionLinearResult(
        weights=weights,
        predictions=predictions,
        reliability_covariance=reliability_covariance,
    )


def _validate_matrix_shapes(
    covariance_to_targets: FloatArray,
    observation_covariance: FloatArray,
    deviations: FloatArray,
) -> None:
    if (
        observation_covariance.ndim != 2
        or observation_covariance.shape[0] != observation_covariance.shape[1]
    ):
        raise ValueError("observation_covariance must be a square matrix")
    if covariance_to_targets.ndim != 2:
        raise ValueError("covariance_to_targets must be a matrix")
    if deviations.ndim != 2 or deviations.shape[1] != 1:
        raise ValueError("deviations must be a column vector")
    size = observation_covariance.shape[0]
    if covariance_to_targets.shape[1] != size:
        raise ValueError(
            "covariance_to_targets column count must match observation covariance size"
        )
    if deviations.shape[0] != size:
        raise ValueError("deviations row count must match observation covariance size")
