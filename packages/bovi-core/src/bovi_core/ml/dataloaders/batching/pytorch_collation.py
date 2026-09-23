"""Combine NumPy samples and convert numeric batch fields to CPU tensors."""

from collections.abc import Collection
from typing import Any

import numpy as np

from .numpy_collation import collate_numpy_samples


def collate_pytorch_samples(
    samples: list[dict[str, Any]], preserve_keys: Collection[str] = ("metadata",)
) -> dict[str, Any]:
    """Preserve values, dtypes and axis order; only batch and tensorize.

    Metadata and ragged fields follow the NumPy collator's list policy.
    Image preprocessing must be supplied explicitly before collation.
    """
    import torch

    if not samples:
        return {}
    batch = collate_numpy_samples(samples, preserve_keys)
    if not isinstance(batch, dict):
        raise ValueError("PyTorch samples must have matching mapping keys")

    def tensorize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: item if key in preserve_keys else tensorize(item)
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            return torch.as_tensor(value)
        return value

    return tensorize(batch)
