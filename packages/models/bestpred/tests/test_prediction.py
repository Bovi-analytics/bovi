from __future__ import annotations

import numpy as np

from bestpred.core.prediction import solve_prediction_system


def test_solve_prediction_system_matches_explicit_fortran_algebra() -> None:
    observation_covariance = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    covariance_to_targets = np.array([[2.0, 0.5], [0.25, 1.5]], dtype=np.float64)
    deviations = np.array([[10.0], [5.0]], dtype=np.float64)

    result = solve_prediction_system(
        covariance_to_targets=covariance_to_targets,
        observation_covariance=observation_covariance,
        deviations=deviations,
    )
    inverse = np.linalg.inv(observation_covariance)

    np.testing.assert_allclose(result.weights, covariance_to_targets @ inverse)
    np.testing.assert_allclose(result.predictions, covariance_to_targets @ inverse @ deviations)
    np.testing.assert_allclose(
        result.reliability_covariance,
        covariance_to_targets @ inverse @ covariance_to_targets.T,
    )


def test_solve_prediction_system_validates_shapes() -> None:
    import pytest

    with pytest.raises(ValueError):
        solve_prediction_system(
            covariance_to_targets=np.ones((1, 2), dtype=np.float64),
            observation_covariance=np.ones((2, 3), dtype=np.float64),
            deviations=np.ones((2, 1), dtype=np.float64),
        )
