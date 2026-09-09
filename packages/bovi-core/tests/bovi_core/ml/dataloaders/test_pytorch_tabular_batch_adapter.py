"""PyTorch batch validation runs in the affected runner's Torch process."""

import pytest
from bovi_core.ml.dataloaders.model_inputs.pytorch_regression import (
    prepare_pytorch_regression_inputs,
)

pytestmark = pytest.mark.torch


def test_native_columns_follow_model_feature_order():
    import torch

    x, y = prepare_pytorch_regression_inputs(
        {"features": {"b": torch.tensor([3, 4]), "a": torch.tensor([1, 2])}, "labels": [5, 6]},
        ("a", "b"),
    )
    torch.testing.assert_close(x, torch.tensor([[1, 3], [2, 4]], dtype=torch.float32))
    torch.testing.assert_close(y, torch.tensor([5, 6], dtype=torch.float32))


def test_invalid_native_regression_batches_fail_explicitly(invalid_regression_batch):
    with pytest.raises(ValueError):
        prepare_pytorch_regression_inputs(*invalid_regression_batch)


def test_precision_is_explicit():
    import torch

    batch = {"features": [[1, 2]], "labels": [3]}
    x, y = prepare_pytorch_regression_inputs(batch, ("a", "b"), dtype=torch.float64)
    assert x.dtype == y.dtype == torch.float64
    with pytest.raises(ValueError, match="floating point"):
        prepare_pytorch_regression_inputs(batch, ("a", "b"), dtype=torch.int64)
