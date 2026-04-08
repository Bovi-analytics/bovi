"""
Lactation-specific transforms: Herd stats enrichment, Event tokenization, Milk normalization, etc.

These transforms use domain-specific knowledge about lactation data and
are designed to work with the lactation data structure.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from bovi_core.ml.dataloaders.base.universal_transform import UniversalTransform
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry
from typing_extensions import override

logger = logging.getLogger(__name__)


@TransformRegistry.register("herd_stats_enrichment")
class HerdStatsEnrichmentTransform(UniversalTransform):
    """
    Enrich data dicts with herd statistics using hierarchical fallback.

    Loads pkl reference data at init and adds ``herd_stats`` to data dicts
    based on ``herd_id`` + ``parity`` with a 4-level hierarchical fallback:

    - Level 1: herd_id + parity (most specific)
    - Level 2: herd_id only (parity average for herd)
    - Level 3: parity only (herd average for parity)
    - Level 4: global average (least specific)

    Args:
        herd_stats_dir: Directory containing herd statistics pickle files.
    """

    herd_stats_dir: Path
    idx_to_herd_par: dict[int, str]
    herd_stats_per_parity: dict[int, dict[str, npt.NDArray[np.float32] | dict[str, float]]]
    herd_stats_per_herd: dict[int, npt.NDArray[np.float32] | dict[str, float]]
    herd_stats_per_parity_global: dict[str, npt.NDArray[np.float32] | dict[str, float]]
    herd_stats_global: npt.NDArray[np.float32] | dict[str, float]

    def __init__(self, herd_stats_dir: str | Path) -> None:
        """
        Args:
            herd_stats_dir: Directory containing herd statistics pickle files.
        """
        self.herd_stats_dir = Path(herd_stats_dir)
        if not self.herd_stats_dir.exists():
            raise ValueError(f"Herd stats directory not found: {self.herd_stats_dir}")
        self._load_herd_stats()

    # ------------------------------------------------------------------
    # Loading helpers (local for now; swap for blob later)
    # ------------------------------------------------------------------

    def _load_pkl(self, path: Path) -> object:
        """Load a pickle file from local disk.

        This method exists as a seam so it can later be replaced with
        a loader that fetches from blob storage.

        Args:
            path: Local path to the pickle file.

        Returns:
            Deserialized Python object.
        """
        import warnings

        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                return pickle.load(f, encoding="latin1")

    def _load_herd_stats(self) -> None:
        """Load all herd statistics pickle files."""
        logger.info(f"Loading herd statistics from {self.herd_stats_dir}")

        # Index to herd parameter mapping
        self.idx_to_herd_par = cast(
            dict[int, str],
            self._load_pkl(self.herd_stats_dir / "idx_to_herd_par_dict.pkl"),
        )
        logger.info(f"  - Loaded herd parameter index: {len(self.idx_to_herd_par)} parameters")

        # Herd stats per parity (Level 1: most specific)
        self.herd_stats_per_parity = cast(
            dict[int, dict[str, npt.NDArray[np.float32] | dict[str, float]]],
            self._load_pkl(self.herd_stats_dir / "herd_stats_per_parity_dict.pkl"),
        )
        logger.info(f"  - Loaded herd stats per parity: {len(self.herd_stats_per_parity)} herds")

        # Herd stats means per herd (Level 2)
        self.herd_stats_per_herd = cast(
            dict[int, npt.NDArray[np.float32] | dict[str, float]],
            self._load_pkl(self.herd_stats_dir / "herd_stats_means_per_herd.pkl"),
        )
        logger.info(f"  - Loaded herd stats means: {len(self.herd_stats_per_herd)} herds")

        # Herd stats means per parity global (Level 3)
        self.herd_stats_per_parity_global = cast(
            dict[str, npt.NDArray[np.float32] | dict[str, float]],
            self._load_pkl(self.herd_stats_dir / "herd_stat_means_per_parity.pkl"),
        )
        logger.info(f"  - Loaded parity means: {len(self.herd_stats_per_parity_global)} parities")

        # Global means (Level 4)
        self.herd_stats_global = cast(
            npt.NDArray[np.float32] | dict[str, float],
            self._load_pkl(self.herd_stats_dir / "herd_stat_means_global.pkl"),
        )
        logger.info("  - Loaded global means")

    # ------------------------------------------------------------------
    # Fallback logic
    # ------------------------------------------------------------------

    def _get_herd_stats_with_fallback(
        self,
        herd_id: int | None,
        parity: int | None,
    ) -> npt.NDArray[np.float32]:
        """Get herd statistics with 4-level hierarchical fallback.

        Tries to find stats in this order:
        1. herd_id + parity (most specific)
        2. herd_id only (parity average for herd)
        3. parity only (herd average for parity)
        4. global average (least specific)

        Returns:
            np.array of herd statistics, ordered by idx_to_herd_par.
        """
        stats_dict: npt.NDArray[np.float32] | dict[str, float] | None = None
        fallback_level = 4

        # Level 1: Most specific (herd_id + parity)
        if herd_id is not None and parity is not None:
            try:
                parity_str = str(parity)
                if (
                    herd_id in self.herd_stats_per_parity
                    and parity_str in self.herd_stats_per_parity[herd_id]
                ):
                    stats_dict = self.herd_stats_per_parity[herd_id][parity_str]
                    fallback_level = 1
            except (KeyError, TypeError, ValueError):
                pass

        # Level 2: Herd average across parities
        if stats_dict is None and herd_id is not None:
            try:
                if herd_id in self.herd_stats_per_herd:
                    stats_dict = self.herd_stats_per_herd[herd_id]
                    fallback_level = 2
            except (KeyError, TypeError, ValueError):
                pass

        # Level 3: Parity average across herds
        if stats_dict is None and parity is not None:
            try:
                parity_str = str(parity)
                if parity_str in self.herd_stats_per_parity_global:
                    stats_dict = self.herd_stats_per_parity_global[parity_str]
                    fallback_level = 3
            except (KeyError, TypeError, ValueError):
                pass

        # Level 4: Global average
        if stats_dict is None:
            if isinstance(self.herd_stats_global, np.ndarray):
                stats_dict = self.herd_stats_global.copy()
            else:
                stats_dict = dict(self.herd_stats_global)
            fallback_level = 4

        return self._convert_stats_to_array(stats_dict, fallback_level)

    def _convert_stats_to_array(
        self,
        stats_dict: npt.NDArray[np.float32] | dict[str, float] | None,
        fallback_level: int,
    ) -> npt.NDArray[np.float32]:
        """Convert stats dict or array to ordered numpy array."""
        try:
            if isinstance(stats_dict, np.ndarray):
                if len(stats_dict) == len(self.idx_to_herd_par):
                    return stats_dict.astype(np.float32)
                else:
                    logger.warning(
                        f"Array length mismatch: {len(stats_dict)} vs "
                        f"{len(self.idx_to_herd_par)}. Using fallback level {fallback_level}"
                    )
                    return np.zeros(len(self.idx_to_herd_par), dtype=np.float32)
            elif isinstance(stats_dict, dict):
                stats_array: npt.NDArray[np.float32] = np.array(
                    [
                        stats_dict.get(self.idx_to_herd_par[i], 0.0)
                        for i in range(len(self.idx_to_herd_par))
                    ],
                    dtype=np.float32,
                )
                return stats_array
            else:
                logger.warning(
                    f"Unexpected stats type {type(stats_dict)}. Expected dict or ndarray. "
                    f"Using fallback level {fallback_level}"
                )
                return np.zeros(len(self.idx_to_herd_par), dtype=np.float32)
        except Exception as e:
            logger.warning(
                f"Error converting stats to array: {e}. Using fallback level {fallback_level}"
            )
            return np.zeros(len(self.idx_to_herd_par), dtype=np.float32)

    # ------------------------------------------------------------------
    # Transform interface
    # ------------------------------------------------------------------

    @override
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """Add herd_stats to data dict based on herd_id and parity.

        Args:
            data: Dictionary with 'herd_id' and 'parity' fields.

        Returns:
            Data dict with 'herd_stats' added.
        """
        herd_id_raw = data.get("herd_id")
        parity_raw = data.get("parity")

        if herd_id_raw is not None and parity_raw is not None:
            herd_id_int = herd_id_raw if isinstance(herd_id_raw, int) else int(str(herd_id_raw))
            parity_int = parity_raw if isinstance(parity_raw, int) else int(str(parity_raw))
            herd_stats = self._get_herd_stats_with_fallback(herd_id_int, parity_int)
        else:
            logger.warning("Missing herd_id or parity in data dict")
            herd_stats = self._get_herd_stats_with_fallback(None, None)

        data["herd_stats"] = herd_stats
        return data

    @override
    def get_params(self) -> dict[str, object]:
        return {
            "name": "herd_stats_enrichment",
            "herd_stats_dir": str(self.herd_stats_dir),
        }


@TransformRegistry.register("event_tokenization")
class EventTokenizationTransform(UniversalTransform):
    """
    Convert event strings to integer indices.

    Loads its own ``event_to_idx`` mapping from a pickle file at init.
    Falls back to reading ``event_to_idx`` from the data dict for
    backward compatibility.

    Handles:
    - Lowercase conversion for case-insensitive matching
    - Unknown event fallback to "unknown" index
    - Consistent ordering in output
    """

    unknown_event: str
    event_to_idx: dict[str, int] | None

    def __init__(
        self,
        event_to_idx_path: str | Path | None = None,
        unknown_event: str = "unknown",
    ) -> None:
        """
        Args:
            event_to_idx_path: Path to pickle file with event-to-index mapping.
                If not provided, falls back to ``data["event_to_idx"]``.
            unknown_event: Event name for unknown/unseen events (default: "unknown").
        """
        self.unknown_event = unknown_event
        self.event_to_idx = None
        if event_to_idx_path is not None:
            self.event_to_idx = self._load_pkl(Path(event_to_idx_path))

    def _load_pkl(self, path: Path) -> dict[str, int]:
        """Load a pickle file from local disk.

        This method exists as a seam so it can later be replaced
        with a loader that fetches from blob storage.

        Args:
            path: Local path to the pickle file.

        Returns:
            Event-to-index mapping dict.
        """
        import warnings

        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                return pickle.load(f, encoding="latin1")

    @override
    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        """
        Apply event tokenization transform to data.

        Args:
            data: Dictionary with 'events' field and optionally 'event_to_idx'.

        Returns:
            Transformed data dictionary with encoded events.
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

        # Resolve event_to_idx: prefer self, fall back to data dict
        event_to_idx = self.event_to_idx
        if event_to_idx is None:
            if "event_to_idx" not in data:
                logger.warning("'event_to_idx' mapping not found (neither init nor data dict)")
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
