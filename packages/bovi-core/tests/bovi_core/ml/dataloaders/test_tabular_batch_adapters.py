"""NumPy scalar-regression batch validation."""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.adapters import tabular


@pytest.fixture
def regression_adapter():
    return tabular.numpy_regression_batch


def test_columns_follow_model_feature_order(regression_adapter):
    x, y = regression_adapter(
        {"features": {"b": [3, 4], "a": [1, 2]}, "labels": [5, 6]}, ("a", "b")
    )
    np.testing.assert_array_equal(np.asarray(x), [[1, 3], [2, 4]])
    np.testing.assert_array_equal(np.asarray(y), [5, 6])


def test_invalid_regression_batches_fail_explicitly(regression_adapter, invalid_regression_batch):
    batch, names = invalid_regression_batch
    with pytest.raises(ValueError):
        regression_adapter(batch, names)


def test_dense_batches_remain_supported(regression_adapter):
    x, y = regression_adapter({"features": [[1, 2]], "labels": [3]}, ("a", "b"))
    np.testing.assert_array_equal(np.asarray(x), [[1, 2]])
    np.testing.assert_array_equal(np.asarray(y), [3])
