"""Finite, in-memory source for a JSON array of records."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .dict_source import DictSource


class JSONRecordsSource(DictSource):
    """Read UTF-8 JSON objects once; expose independent records and provenance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ValueError("Expected a JSON array of records (objects)")
        super().__init__(records)

    def load_item(self, key: int | str) -> dict[str, Any]:
        return deepcopy(super().load_item(key))

    def get_metadata(self, key: int | str) -> dict[str, Any]:
        index = int(key)
        record = self._items[index]
        if index < 0:
            index += len(self)
        metadata: dict[str, Any] = {"index": index, "path": str(self.path)}
        if record.get("sample_id") is not None:
            metadata["sample_id"] = record["sample_id"]
        return metadata
