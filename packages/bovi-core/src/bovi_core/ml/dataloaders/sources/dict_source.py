"""In-memory data source for serving dicts directly."""

from __future__ import annotations

from bovi_core.ml.dataloaders.base.data_source import DataSource


class DictSource(DataSource[dict[str, object]]):
    """Serve pre-built dicts as a DataSource.

    For inference from HTTP requests, test data, or any scenario
    where data is already in memory.

    Args:
        items: List of data dicts to serve.

    """

    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def load_item(self, key: int | str) -> dict[str, object]:
        return self._items[int(key)]

    def get_metadata(self, key: int | str) -> dict[str, object]:
        return {"index": int(key)}

    def get_keys(self) -> list[int | str]:
        return list(range(len(self._items)))
