"""Combine dataset samples into batches without model-specific preprocessing."""

from .numpy_collation import collate_numpy_samples
from .pytorch_collation import collate_pytorch_samples

__all__ = ["collate_numpy_samples", "collate_pytorch_samples"]
