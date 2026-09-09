"""Batch NumPy samples, preserving nested fields and per-sample metadata."""

from collections.abc import Collection, Mapping
from numbers import Number
from typing import Any

import numpy as np


def collate_numpy_samples(
    samples: list[Any], preserve_keys: Collection[str] = ("metadata",)
) -> Any:
    """Stack compatible numeric values; keep ragged or opaque values as lists.

    Matching mappings are combined recursively. Preserved fields remain one
    record per sample, even inside nested mappings. No scaling or axis changes
    are performed. An empty input produces an empty list.
    """
    if not samples:
        return []

    first = samples[0]
    if isinstance(first, Mapping):
        if not all(isinstance(sample, Mapping) for sample in samples):
            return samples
        if any(sample.keys() != first.keys() for sample in samples):
            return samples

        batch = {}
        for key in first:
            values = [sample[key] for sample in samples]
            batch[key] = (
                values if key in preserve_keys else collate_numpy_samples(values, preserve_keys)
            )
        return batch

    if first is None or isinstance(first, (str, bytes)):
        return samples
    if isinstance(first, (Number, np.bool_)):
        return np.asarray(samples)
    if isinstance(first, (np.ndarray, list, tuple)) or hasattr(first, "__array__"):
        try:
            arrays = [np.asarray(sample) for sample in samples]
            if any(array.dtype.kind not in "biufc" for array in arrays):
                return samples
            return np.stack(arrays)
        except (TypeError, ValueError):
            return samples
    return samples
