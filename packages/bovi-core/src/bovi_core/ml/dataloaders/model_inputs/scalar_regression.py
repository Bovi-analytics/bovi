"""Shared shape and field rules for numeric scalar-regression model inputs.

Features are named scalar columns or a dense (samples, features) matrix.
Labels must contain one number per sample. Native conversion and finite-value
checks live in the framework-specific modules, without a NumPy round trip.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def _inputs(batch: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[Any, Any]:
    """Require supervised data and ensure configured feature names exist."""
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be nonempty and unique")
    features = batch.get("features")
    if features is None or batch.get("labels") is None:
        raise ValueError("Regression batches require features and labels")
    if isinstance(features, Mapping):
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Batch is missing configured features: {missing}")
    return features, batch["labels"]


def _validate_shapes(x: Any, y: Any, feature_names: Sequence[str]) -> None:
    """Check the final feature matrix against the flattened target vector."""
    if len(x.shape) != 2 or tuple(x.shape) != (len(y), len(feature_names)):
        raise ValueError("Feature and label shapes do not match the model")
    if not len(y):
        raise ValueError("Regression batches must be nonempty")


def _validate_columns(columns: list[Any]) -> None:
    """Reject vector-valued or unequal-length feature columns before stacking."""
    if any(len(column.shape) != 1 or column.shape != columns[0].shape for column in columns):
        raise ValueError("Regression features must be equal-length scalar columns")


def _validate_labels(labels: Any) -> None:
    """Accept a vector or a single-column matrix, never silently flatten targets."""
    if len(labels.shape) != 1 and not (len(labels.shape) == 2 and labels.shape[1] == 1):
        raise ValueError("Regression labels must have shape (samples,) or (samples, 1)")
