"""Ordered scalar-regression batches, with optional native framework adapters.

These helpers validate supervised numeric batches; they are not the contract
for every dataset. Images, structured targets and ragged sequences use their
own adapters. Framework imports occur only when that adapter is selected.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _inputs(batch: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[Any, Any]:
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
    if len(x.shape) != 2 or tuple(x.shape) != (len(y), len(feature_names)):
        raise ValueError("Feature and label shapes do not match the model")
    if not len(y):
        raise ValueError("Regression batches must be nonempty")


def _validate_columns(columns: list[Any]) -> None:
    if any(len(column.shape) != 1 or column.shape != columns[0].shape for column in columns):
        raise ValueError("Regression features must be equal-length scalar columns")


def _validate_labels(labels: Any) -> None:
    if len(labels.shape) != 1 and not (len(labels.shape) == 2 and labels.shape[1] == 1):
        raise ValueError("Regression labels must have shape (samples,) or (samples, 1)")


def numpy_regression_batch(
    batch: Mapping[str, Any], feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [np.asarray(features[name], dtype=np.float64) for name in feature_names]
        _validate_columns(columns)
        x = np.stack(columns, axis=1)
    else:
        x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _validate_labels(y)
    y = y.reshape(-1)
    _validate_shapes(x, y, feature_names)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regression batches must contain finite values")
    return x, y


def pytorch_regression_batch(
    batch: Mapping[str, Any], feature_names: Sequence[str]
) -> tuple[Any, Any]:
    import torch

    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [torch.as_tensor(features[name], dtype=torch.float32) for name in feature_names]
        _validate_columns(columns)
        x = torch.stack(columns, dim=1)
    else:
        x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    _validate_labels(y)
    y = y.reshape(-1)
    _validate_shapes(x, y, feature_names)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Regression batches must contain finite values")
    return x, y


def tensorflow_regression_batch(
    batch: Mapping[str, Any], feature_names: Sequence[str]
) -> tuple[Any, Any]:
    import tensorflow as tf

    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [
            tf.cast(tf.convert_to_tensor(features[name]), tf.float32) for name in feature_names
        ]
        _validate_columns(columns)
        x = tf.stack(columns, axis=1)
    else:
        x = tf.cast(tf.convert_to_tensor(features), tf.float32)
    y = tf.cast(tf.convert_to_tensor(labels), tf.float32)
    _validate_labels(y)
    y = tf.reshape(y, (-1,))
    _validate_shapes(x, y, feature_names)
    if not tf.reduce_all(tf.math.is_finite(x)) or not tf.reduce_all(tf.math.is_finite(y)):
        raise ValueError("Regression batches must contain finite values")
    return x, y
