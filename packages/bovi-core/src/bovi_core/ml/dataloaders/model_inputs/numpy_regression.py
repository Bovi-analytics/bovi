"""Prepare NumPy feature matrices and target vectors for scalar regression."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from .scalar_regression import _inputs, _validate_columns, _validate_labels, _validate_shapes


def prepare_numpy_regression_inputs(
    batch: Mapping[str, Any], feature_names: Sequence[str], *, dtype: DTypeLike = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite X (samples, features) and y (samples,) in model feature order.

    Named columns are ordered by feature_names; dense matrices must already
    use that order. Precision defaults to float64 and can be chosen explicitly.
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("Regression dtype must be floating point")
    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [np.asarray(features[name], dtype=dtype) for name in feature_names]
        _validate_columns(columns)
        x = np.stack(columns, axis=1)
    else:
        x = np.asarray(features, dtype=dtype)

    y = np.asarray(labels, dtype=dtype)
    _validate_labels(y)
    y = y.reshape(-1)
    _validate_shapes(x, y, feature_names)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regression batches must contain finite values")
    return x, y
