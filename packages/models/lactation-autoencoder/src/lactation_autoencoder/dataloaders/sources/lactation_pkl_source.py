"""
LactationPKLSource: Load lactation data from JSON files.

A pure data source that loads lactation records from JSON files.
Enrichment (herd stats, event tokenization) is handled by transforms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bovi_core.ml.dataloaders.base.data_source import DataSource
from typing_extensions import override

logger = logging.getLogger(__name__)


class LactationPKLSource(DataSource[dict[str, object]]):
    """
    Load lactation data from JSON files.

    Scans a directory for JSON files matching a glob pattern,
    builds a flat index, and loads individual records on demand.

    Args:
        json_root_dir: Directory containing animal_*.json lactation records.
        file_pattern: Glob pattern for JSON files (default: "animal_*.json").
        keep_in_memory: Keep all data in memory vs. load per sample (default: True).
    """

    json_root_dir: Path
    file_pattern: str
    keep_in_memory: bool
    index: list[Path]
    data_cache: dict[int, dict[str, object]] | None

    def __init__(
        self,
        json_root_dir: str | Path,
        file_pattern: str = "animal_*.json",
        keep_in_memory: bool = True,
    ) -> None:
        self.json_root_dir = Path(json_root_dir)
        self.file_pattern = file_pattern
        self.keep_in_memory = keep_in_memory

        # Validate directory
        if not self.json_root_dir.exists():
            raise ValueError(f"JSON directory not found: {self.json_root_dir}")

        # Build index and optionally load data
        self.index = []
        self.data_cache = {} if keep_in_memory else None

        self._build_index()

        logger.info(
            f"LactationPKLSource initialized: {len(self)} lactations from "
            f"{len(set(self.index))} JSON files"
        )

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
            key: Index or file path.

        Returns:
            Dict with raw JSON data (no enrichment).
        """
        if isinstance(key, str):
            json_file = Path(key)
            if json_file not in self.index:
                raise ValueError(f"File not in index: {key}")
            index = self.index.index(json_file)
        else:
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

        return data

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
