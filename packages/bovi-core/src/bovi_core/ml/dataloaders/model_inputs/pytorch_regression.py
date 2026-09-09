"""Prepare native PyTorch model inputs without converting tensors to NumPy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .scalar_regression import _inputs, _validate_columns, _validate_labels, _validate_shapes

if TYPE_CHECKING:
    import torch


def prepare_pytorch_regression_inputs(
    batch: Mapping[str, Any], feature_names: Sequence[str], *, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite X (samples, features) and y (samples,); default float32.

    Named columns follow feature_names; dense matrices must already use that
    order. Existing tensors retain their device; this does not place the model
    or batch on a GPU.
    """
    import torch

    dtype = torch.float32 if dtype is None else dtype
    if not dtype.is_floating_point:
        raise ValueError("Regression dtype must be floating point")
    features, labels = _inputs(batch, feature_names)
    if isinstance(features, Mapping):
        columns = [torch.as_tensor(features[name], dtype=dtype) for name in feature_names]
        _validate_columns(columns)
        x = torch.stack(columns, dim=1)
    else:
        x = torch.as_tensor(features, dtype=dtype)

    y = torch.as_tensor(labels, dtype=dtype)
    _validate_labels(y)
    y = y.reshape(-1)
    _validate_shapes(x, y, feature_names)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Regression batches must contain finite values")
    return x, y
