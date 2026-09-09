"""PyTorch batch validation runs in the affected runner's Torch process."""

import pytest
from bovi_core.ml.dataloaders.adapters.tabular import pytorch_regression_batch

pytestmark = pytest.mark.torch


def test_native_columns_follow_model_feature_order():
    import torch

    x, y = pytorch_regression_batch(
        {"features": {"b": torch.tensor([3, 4]), "a": torch.tensor([1, 2])}, "labels": [5, 6]},
        ("a", "b"),
    )
    torch.testing.assert_close(x, torch.tensor([[1, 3], [2, 4]], dtype=torch.float32))
    torch.testing.assert_close(y, torch.tensor([5, 6], dtype=torch.float32))


def test_invalid_native_regression_batches_fail_explicitly(invalid_regression_batch):
    with pytest.raises(ValueError):
        pytorch_regression_batch(*invalid_regression_batch)
