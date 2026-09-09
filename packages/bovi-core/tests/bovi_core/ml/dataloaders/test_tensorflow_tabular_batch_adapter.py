"""TensorFlow batch validation runs in the affected runner's TF process."""

import pytest
from bovi_core.ml.dataloaders.model_inputs.tensorflow_regression import (
    prepare_tensorflow_regression_inputs,
)

pytestmark = pytest.mark.tensorflow


def test_native_columns_follow_model_feature_order():
    import tensorflow as tf

    x, y = prepare_tensorflow_regression_inputs(
        {"features": {"b": tf.constant([3, 4]), "a": tf.constant([1, 2])}, "labels": [5, 6]},
        ("a", "b"),
    )
    tf.debugging.assert_equal(x, tf.constant([[1, 3], [2, 4]], dtype=tf.float32))
    tf.debugging.assert_equal(y, tf.constant([5, 6], dtype=tf.float32))


def test_invalid_native_regression_batches_fail_explicitly(invalid_regression_batch):
    with pytest.raises(ValueError):
        prepare_tensorflow_regression_inputs(*invalid_regression_batch)


def test_precision_is_explicit():
    import tensorflow as tf

    batch = {"features": [[1, 2]], "labels": [3]}
    x, y = prepare_tensorflow_regression_inputs(batch, ("a", "b"), dtype=tf.float64)
    assert x.dtype == y.dtype == tf.float64
    with pytest.raises(ValueError, match="floating point"):
        prepare_tensorflow_regression_inputs(batch, ("a", "b"), dtype=tf.int64)
