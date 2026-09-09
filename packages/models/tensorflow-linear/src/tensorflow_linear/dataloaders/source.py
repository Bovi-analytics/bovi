"""JSON input kept small enough for a CPU tutorial."""

import json
from pathlib import Path
from typing import Any

from bovi_core.ml import DataSource


class LinearJSONSource(DataSource[dict[str, Any]]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        records = json.loads(self.path.read_text())
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError("Expected a JSON array of records")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def load_item(self, key: int | str) -> dict[str, Any]:
        return dict(self.records[int(key)])

    def get_metadata(self, key: int | str) -> dict[str, object]:
        return {"index": int(key), "path": str(self.path)}

    def get_keys(self) -> list[int | str]:
        return list(range(len(self)))
