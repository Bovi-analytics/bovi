"""Prepare eager TensorFlow scalar-regression inputs without a NumPy round trip."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .scalar_regression import _inputs, _validate_columns, _validate_labels, _validate_shapes

if TYPE_CHECKING:
    import tensorflow as tf


def prepare_tensorflow_regression_inputs(
    batch: Mapping[str, Any], feature_names: Sequence[str], *, dtype: tf.DType | None = None
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return finite X (samples, features) and y (samples,); default float32.

    For eager trainer/evaluator loops, not symbolic tf.function inputs. Named
    columns follow feature_names; dense matrices must already use that order.
    """
    import tensorflow as tf

    dtype = tf.float32 if dtype is None else tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("Regression dtype must be floating point")
    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [tf.cast(tf.convert_to_tensor(features[name]), dtype) for name in feature_names]
        _validate_columns(columns)
        x = tf.stack(columns, axis=1)
    else:
        x = tf.cast(tf.convert_to_tensor(features), dtype)

    y = tf.cast(tf.convert_to_tensor(labels), dtype)
    _validate_labels(y)
    y = tf.reshape(y, (-1,))
    _validate_shapes(x, y, feature_names)
    if not tf.reduce_all(tf.math.is_finite(x)) or not tf.reduce_all(tf.math.is_finite(y)):
        raise ValueError("Regression batches must contain finite values")
    return x, y
