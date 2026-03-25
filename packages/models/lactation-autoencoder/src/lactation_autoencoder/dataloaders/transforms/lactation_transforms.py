"""
Lactation-specific transforms: Event tokenization, Milk normalization, etc.

These transforms use domain-specific knowledge about lactation data and
are designed to work with the lactation data structure.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numpy.typing as npt
from bovi_core.ml.dataloaders.base.universal_transform import UniversalTransform
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry
from typing_extensions import override

logger = logging.getLogger(__name__)


@TransformRegistry.register("event_tokenization")
class EventTokenizationTransform(UniversalTransform):
    """
    Convert event strings to integer indices.

    Requires event_to_idx_dict to be present in the data dict
    (typically provided by LactationJSONSource).

    Handles:
    - Lowercase conversion for case-insensitive matching
    - Unknown event fallback to "unknown" index
    - Consistent ordering in output
    """

    unknown_event: str

    def __init__(self, unknown_event: str = "unknown") -> None:
        """
        Args:
            unknown_event: Event name for unknown/unseen events (default: "unknown")
        """
        self.unknown_event = unknown_event

    @override
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply event tokenization transform to data.

        Args:
            data: Dictionary with 'events' and 'event_to_idx' fields

        Returns:
            Transformed data dictionary with encoded events
        """
        # Handle nested "features" structure (from LactationDataset)
        features: dict[str, object] | None = None
        if "features" in data:
            features = cast(dict[str, object], data["features"])
            if "events" not in features:
                logger.warning("'events' field not found in data['features']")
                return data
            events = features["events"]
        elif "events" in data:
            events = data["events"]
        else:
            logger.warning("'events' field not found in data")
            return data

        # Get event_to_idx mapping from data dict
        if "event_to_idx" not in data:
            logger.warning("'event_to_idx' mapping not found in data dict")
            return data

        event_to_idx = cast(dict[str, int], data["event_to_idx"])

        # Convert events to list if needed
        if isinstance(events, np.ndarray):
            events_list = cast(list[str | int | None], events.tolist())
        elif isinstance(events, list):
            events_list = cast(list[str | int | None], events)
        else:
            events_list = cast(list[str | int | None], list(cast(list[object], events)))

        # Convert to lowercase and map
        unknown_idx = event_to_idx.get(self.unknown_event, len(event_to_idx))

        events_encoded: list[int] = []
        for event in events_list:
            if event is None:
                events_encoded.append(unknown_idx)
            else:
                event_str = str(event).lower()
                idx = event_to_idx.get(event_str, unknown_idx)
                events_encoded.append(idx)

        events_encoded_array: npt.NDArray[np.int32] = np.array(events_encoded, dtype=np.int32)

        # Store in the same location as input
        if features is not None:
            features["events"] = events_encoded_array
        else:
            data["events_encoded"] = events_encoded_array

        return data

    @override
    def get_params(self) -> dict[str, object]:
        return {
            "name": "event_tokenization",
            "unknown_event": self.unknown_event,
        }


@TransformRegistry.register("milk_normalization")
class MilkNormalizationTransform(UniversalTransform):
    """
    Normalize milk to 0-1 range by dividing by max expected milk.

    Default: 80 kg/day (typical maximum for dairy cows).

    This normalization is lactation-specific based on domain knowledge
    about reasonable milk production values.
    """

    max_milk: float

    def __init__(self, max_milk: float = 80.0) -> None:
        """
        Args:
            max_milk: Maximum expected milk value (default: 80 kg/day)
        """
        if max_milk <= 0:
            raise ValueError(f"max_milk must be positive, got {max_milk}")
        self.max_milk = max_milk

    @override
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply milk normalization transform to data.

        Args:
            data: Dictionary with 'milk' field

        Returns:
            Transformed data dictionary with normalized milk
        """
        # Handle nested "features" structure (from LactationDataset)
        features: dict[str, object] | None = None
        if "features" in data:
            features = cast(dict[str, object], data["features"])
            if "milk" not in features:
                logger.warning("'milk' field not found in data['features']")
                return data
            milk = features["milk"]
        elif "milk" in data:
            milk = data["milk"]
        else:
            logger.warning("'milk' field not found in data")
            return data

        # Convert to numpy array if needed
        if milk is None:
            logger.warning("'milk' field is None")
            return data

        if isinstance(milk, list):
            milk_array: npt.NDArray[np.float32] = np.array(milk, dtype=np.float32)
        else:
            milk_array = np.asarray(milk, dtype=np.float32)

        # Replace NaN with 0 before normalization
        milk_array = np.where(np.isnan(milk_array), 0.0, milk_array).astype(np.float32)

        # Normalize
        milk_normalized: npt.NDArray[np.float32] = milk_array / self.max_milk

        # Store in the same location as input
        if features is not None:
            features["milk"] = milk_normalized
        else:
            data["milk"] = milk_normalized

        return data

    @override
    def get_params(self) -> dict[str, object]:
        return {
            "name": "milk_normalization",
            "max_milk": self.max_milk,
        }


@TransformRegistry.register("herd_stats_normalization")
class HerdStatsNormalizationTransform(UniversalTransform):
    """
    Normalize herd statistics.

    Applies z-score normalization per field to standardize
    herd statistics across different herds and parities.
    """

    method: str
    epsilon: float

    def __init__(self, method: str = "zscore", epsilon: float = 1e-6) -> None:
        """
        Args:
            method: Normalization method (zscore, minmax)
            epsilon: Small value to prevent division by zero
        """
        if method not in ["zscore", "minmax"]:
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.epsilon = epsilon

    @override
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply herd stats normalization transform to data.

        Args:
            data: Dictionary with 'herd_stats' field

        Returns:
            Transformed data dictionary with normalized herd_stats
        """
        # Handle nested "features" structure (from LactationDataset)
        features: dict[str, object] | None = None
        if "features" in data:
            features = cast(dict[str, object], data["features"])
            if "herd_stats" not in features:
                logger.warning("'herd_stats' field not found in data['features']")
                return data
            herd_stats = features["herd_stats"]
        elif "herd_stats" in data:
            herd_stats = data["herd_stats"]
        else:
            logger.warning("'herd_stats' field not found in data")
            return data

        # Convert to numpy array if needed
        if isinstance(herd_stats, list):
            stats_array: npt.NDArray[np.float32] = np.array(herd_stats, dtype=np.float32)
        else:
            stats_array = np.asarray(herd_stats, dtype=np.float32)

        # Replace NaN with 0
        stats_array = np.where(np.isnan(stats_array), 0.0, stats_array).astype(np.float32)

        if self.method == "zscore":
            mean = float(np.mean(stats_array))
            std = float(np.std(stats_array))
            if std > self.epsilon:
                stats_array = (stats_array - mean) / (std + self.epsilon)
            else:
                stats_array = stats_array - mean

        elif self.method == "minmax":
            min_val = float(np.min(stats_array))
            max_val = float(np.max(stats_array))
            if max_val - min_val > self.epsilon:
                stats_array = (stats_array - min_val) / (max_val - min_val + self.epsilon)

        # Store in the same location as input
        stats_array = stats_array.astype(np.float32)
        if features is not None:
            features["herd_stats"] = stats_array
        else:
            data["herd_stats"] = stats_array

        return data

    @override
    def get_params(self) -> dict[str, object]:
        return {
            "name": "herd_stats_normalization",
            "method": self.method,
            "epsilon": self.epsilon,
        }
