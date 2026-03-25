"""
LactationPKLSource: Load lactation data from pickle + JSON files.

Implements hierarchical fallback for herd statistics to ensure the model
never fails due to missing metadata.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
from bovi_core.ml.dataloaders.base.data_source import DataSource
from typing_extensions import override

logger = logging.getLogger(__name__)


class LactationPKLSource(DataSource[dict[str, object]]):
    """
    Load lactation data from pickle files with hierarchical herd statistics.

    Data Flow:
    1. Scan JSON files to build flat index: (file, animal_id, parity) → lactation record
    2. Load herd statistics pickle files at initialization
    3. For each item: return raw data with hierarchical fallback for herd stats

    Hierarchical Fallback (4 levels):
    - Level 1: herd_id + parity (most specific)
    - Level 2: herd_id only (parity average for herd)
    - Level 3: parity only (herd average for parity)
    - Level 4: global average (least specific)

    Args:
        json_root_dir: Directory containing animal_*.json lactation records
        herd_stats_dir: Directory containing pickle files with herd statistics metadata
        file_pattern: Glob pattern for JSON files (default: "*.json")
        keep_in_memory: Keep all data in memory vs. load per sample (default: True)
    """

    # Class attributes with type hints
    json_root_dir: Path
    herd_stats_dir: Path
    file_pattern: str
    keep_in_memory: bool
    index: list[Path]
    data_cache: dict[int, dict[str, object]] | None

    # Herd statistics loaded from pickle files
    event_to_idx: dict[str, int]
    idx_to_herd_par: dict[int, str]
    herd_stats_per_parity: dict[int, dict[str, npt.NDArray[np.float32] | dict[str, float]]]
    herd_stats_per_herd: dict[int, npt.NDArray[np.float32] | dict[str, float]]
    herd_stats_per_parity_global: dict[str, npt.NDArray[np.float32] | dict[str, float]]
    herd_stats_global: npt.NDArray[np.float32] | dict[str, float]

    def __init__(
        self,
        json_root_dir: str | Path,
        herd_stats_dir: str | Path,
        file_pattern: str = "animal_*.json",
        keep_in_memory: bool = True,
    ) -> None:
        self.json_root_dir = Path(json_root_dir)
        self.herd_stats_dir = Path(herd_stats_dir)
        self.file_pattern = file_pattern
        self.keep_in_memory = keep_in_memory

        # Validate directories
        if not self.json_root_dir.exists():
            raise ValueError(f"JSON directory not found: {self.json_root_dir}")
        if not self.herd_stats_dir.exists():
            raise ValueError(f"Herd stats directory not found: {self.herd_stats_dir}")

        # Load herd statistics pickle files
        self._load_herd_stats()

        # Build index and optionally load data
        self.index = []
        self.data_cache = {} if keep_in_memory else None

        self._build_index()

        logger.info(
            f"LactationJSONSource initialized: {len(self)} lactations from "
            f"{len(set(self.index))} JSON files"
        )

    def _load_herd_stats(self) -> None:
        """Load all 5 herd statistics pickle files."""
        logger.info(f"Loading herd statistics from {self.herd_stats_dir}")

        # Event mapping
        event_file = self.herd_stats_dir / "event_to_idx_dict.pkl"
        with open(event_file, "rb") as f:
            self.event_to_idx = pickle.load(f, encoding="latin1")
        logger.info(f"  - Loaded event mapping: {len(self.event_to_idx)} events")

        # Index to herd parameter mapping
        herd_par_file = self.herd_stats_dir / "idx_to_herd_par_dict.pkl"
        with open(herd_par_file, "rb") as f:
            self.idx_to_herd_par = pickle.load(f, encoding="latin1")
        logger.info(f"  - Loaded herd parameter index: {len(self.idx_to_herd_par)} parameters")

        # Herd stats per parity (Level 1: most specific)
        herd_per_par_file = self.herd_stats_dir / "herd_stats_per_parity_dict.pkl"
        with open(herd_per_par_file, "rb") as f:
            self.herd_stats_per_parity = pickle.load(f, encoding="latin1")
        logger.info(f"  - Loaded herd stats per parity: {len(self.herd_stats_per_parity)} herds")

        # Herd stats means per herd (Level 2)
        herd_means_file = self.herd_stats_dir / "herd_stats_means_per_herd.pkl"
        with open(herd_means_file, "rb") as f:
            self.herd_stats_per_herd = pickle.load(f, encoding="latin1")
        logger.info(f"  - Loaded herd stats means: {len(self.herd_stats_per_herd)} herds")

        # Herd stats means per parity global (Level 3)
        parity_means_file = self.herd_stats_dir / "herd_stat_means_per_parity.pkl"
        with open(parity_means_file, "rb") as f:
            self.herd_stats_per_parity_global = pickle.load(f, encoding="latin1")
        logger.info(f"  - Loaded parity means: {len(self.herd_stats_per_parity_global)} parities")

        # Global means (Level 4)
        global_means_file = self.herd_stats_dir / "herd_stat_means_global.pkl"
        with open(global_means_file, "rb") as f:
            self.herd_stats_global = pickle.load(f, encoding="latin1")
        logger.info("  - Loaded global means")

    def _build_index(self) -> None:
        """Build flat index of all lactations."""
        json_files = sorted(self.json_root_dir.glob(self.file_pattern))
        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data: dict[str, object] = json.load(f)

                # Capture index before appending
                idx = len(self.index)
                self.index.append(json_file)

                if self.keep_in_memory and self.data_cache is not None:
                    self.data_cache[idx] = data

            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")

    @override
    def __len__(self) -> int:
        """Return number of lactation records."""
        return len(self.index)

    @override
    def load_item(self, key: int | str) -> dict[str, object]:
        """
        Load raw data for a single lactation record.

        Args:
            key: Index or file path

        Returns:
            Dict with raw data from JSON + hierarchical herd stats
        """
        if isinstance(key, str):
            # Load by file path
            json_file = Path(key)
            if json_file not in self.index:
                raise ValueError(f"File not in index: {key}")
            index = self.index.index(json_file)
        else:
            # Load by index
            index = key
            if index < 0 or index >= len(self.index):
                raise IndexError(f"Index {index} out of range [0, {len(self) - 1}]")
            json_file = self.index[index]

        # Load JSON data
        if self.keep_in_memory and self.data_cache is not None and index in self.data_cache:
            data = dict(self.data_cache[index])  # Make a copy
        else:
            with open(json_file, "r") as f:
                data = json.load(f)

        # Add herd stats with hierarchical fallback
        herd_id_raw = data.get("herd_id")
        parity_raw = data.get("parity")

        if herd_id_raw is not None and parity_raw is not None:
            # Convert to int, handling both int and str types from JSON
            herd_id_int = herd_id_raw if isinstance(herd_id_raw, int) else int(str(herd_id_raw))
            parity_int = parity_raw if isinstance(parity_raw, int) else int(str(parity_raw))
            herd_stats = self._get_herd_stats_with_fallback(herd_id_int, parity_int)
        else:
            logger.warning(f"Missing herd_id or parity in {json_file}")
            herd_stats = self._get_herd_stats_with_fallback(None, None)

        data["herd_stats"] = herd_stats

        # Add event mapping for transforms to use
        data["event_to_idx"] = self.event_to_idx

        return data

    def _get_herd_stats_with_fallback(
        self,
        herd_id: int | None,
        parity: int | None,
    ) -> npt.NDArray[np.float32]:
        """
        Get herd statistics with 4-level hierarchical fallback.

        Tries to find stats in this order:
        1. herd_id + parity (most specific)
        2. herd_id only (parity average for herd)
        3. parity only (herd average for parity)
        4. global average (least specific)

        Returns:
            np.array of 10 herd statistics, ordered by idx_to_herd_par
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

        # Convert stats to ordered numpy array using idx_to_herd_par mapping
        return self._convert_stats_to_array(stats_dict, fallback_level)

    def _convert_stats_to_array(
        self,
        stats_dict: npt.NDArray[np.float32] | dict[str, float] | None,
        fallback_level: int,
    ) -> npt.NDArray[np.float32]:
        """Convert stats dict or array to ordered numpy array."""
        try:
            # Handle both dict and array types from pickle files
            if isinstance(stats_dict, np.ndarray):
                # Already an array, ensure correct dtype and length
                if len(stats_dict) == len(self.idx_to_herd_par):
                    return stats_dict.astype(np.float32)
                else:
                    logger.warning(
                        f"Array length mismatch: {len(stats_dict)} vs {len(self.idx_to_herd_par)}. "
                        f"Using fallback level {fallback_level}"
                    )
                    return np.zeros(len(self.idx_to_herd_par), dtype=np.float32)
            elif isinstance(stats_dict, dict):
                # Convert dict to ordered array using idx_to_herd_par mapping
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
            # Return zero array as fallback
            return np.zeros(len(self.idx_to_herd_par), dtype=np.float32)

    @override
    def get_metadata(self, key: int | str) -> dict[str, object]:
        """Get metadata for an item."""
        if isinstance(key, int):
            json_file = self.index[key]
        else:
            json_file = Path(key)

        return {
            "file": str(json_file),
            "file_name": json_file.name,
        }

    @override
    def get_keys(self) -> list[int | str]:
        """Get all available keys (indices)."""
        return list(range(len(self.index)))

    @override
    def close(self) -> None:
        """Clean up resources."""
        if self.data_cache is not None:
            self.data_cache.clear()
