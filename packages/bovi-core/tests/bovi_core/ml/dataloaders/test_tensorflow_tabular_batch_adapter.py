"""TensorFlow batch validation runs in the affected runner's TF process."""

import pytest
from bovi_core.ml.dataloaders.adapters.tabular import tensorflow_regression_batch

pytestmark = pytest.mark.tensorflow


def test_native_columns_follow_model_feature_order():
    import tensorflow as tf

    x, y = tensorflow_regression_batch(
        {"features": {"b": tf.constant([3, 4]), "a": tf.constant([1, 2])}, "labels": [5, 6]},
        ("a", "b"),
    )
    tf.debugging.assert_equal(x, tf.constant([[1, 3], [2, 4]], dtype=tf.float32))
    tf.debugging.assert_equal(y, tf.constant([5, 6], dtype=tf.float32))


def test_invalid_native_regression_batches_fail_explicitly(invalid_regression_batch):
    with pytest.raises(ValueError):
        tensorflow_regression_batch(*invalid_regression_batch)
