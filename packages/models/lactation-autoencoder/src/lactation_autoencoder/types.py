"""
Shared type definitions for lactation model.

These TypedDicts define the data structures used across the lactation model:
- LactationFeatures: Input features to the autoencoder
- LactationItem: Output from LactationDataset.__getitem__
- LactationRawData: Raw data before transforms (from data source)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt


class LactationFeatures(TypedDict):
    """Features dict structure from LactationDataset.

    These are the processed features ready for model input.
    """

    milk: npt.NDArray[np.float32]  # shape: (304,), normalized 0-1
    events: npt.NDArray[np.int32]  # shape: (304,), tokenized event indices
    parity: npt.NDArray[np.float32]  # shape: (1,), lactation number
    herd_stats: npt.NDArray[np.float32]  # shape: (10,), normalized statistics


class LactationItem(TypedDict):
    """Single item structure from LactationDataset.__getitem__."""

    features: LactationFeatures
    labels: npt.NDArray[np.float32]  # shape: (304,), same as features["milk"]
    metadata: dict[str, object]


class LactationRawData(TypedDict, total=False):
    """Raw data structure before transforms.

    Uses total=False since not all fields are present at all stages.
    This is the structure passed between transforms.
    """

    # Core data fields (from data source)
    milk: npt.NDArray[np.float32] | list[float] | None
    events: npt.NDArray[np.int32] | list[str] | list[int]
    parity: float | int | str
    herd_stats: npt.NDArray[np.float32] | list[float]

    # Mapping for event tokenization
    event_to_idx: dict[str, int]

    # Nested features structure (used by dataset)
    features: LactationFeatures

    # After tokenization
    events_encoded: npt.NDArray[np.int32]
