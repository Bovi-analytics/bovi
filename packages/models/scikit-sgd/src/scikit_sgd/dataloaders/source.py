"""Small JSON-record data source used by the scikit SGD example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bovi_core.ml import DataSource


class RegressionJSONSource(DataSource[dict[str, Any]]):
    """Load a JSON array of regression records once and expose it by index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Regression data file does not exist: {self.path}")

        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
            raise ValueError("Regression data must be a JSON array of objects")
        self._records = loaded

    def __len__(self) -> int:
        return len(self._records)

    def load_item(self, key: int | str) -> dict[str, Any]:
        return dict(self._records[int(key)])

    def get_metadata(self, key: int | str) -> dict[str, Any]:
        record = self._records[int(key)]
        return {"index": int(key), "sample_id": record.get("sample_id")}

    def get_keys(self) -> list[int | str]:
        return list(range(len(self._records)))
