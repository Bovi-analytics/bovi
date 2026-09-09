"""Configured numeric feature selection for tabular regression samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .feature_vector_dataset import FeatureVectorDataset

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import DataSource


class TabularDataset(FeatureVectorDataset):
    """Select named numeric features and an optional scalar target from records.

    Apply record transforms to the source before constructing this dataset.
    Feature mappings follow ``feature_names`` order; framework adapters choose
    the model's matrix representation. Metadata is supplied by the source.
    """

    def __init__(
        self,
        source: DataSource[dict[str, Any]],
        feature_names: Sequence[str],
        target_name: str | None = "y",
        config: Config | None = None,
    ) -> None:
        names = tuple(feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain non-empty field names")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        super().__init__(source=source, config=config)
        self.feature_names = names
        self.target_name = target_name

    def _get_features(self, raw_data: dict[str, Any]) -> dict[str, float]:
        try:
            return {name: float(raw_data[name]) for name in self.feature_names}
        except KeyError as exc:
            raise ValueError(f"Missing configured feature: {exc.args[0]}") from exc

    def _get_labels(self, raw_data: dict[str, Any]) -> float | None:
        if self.target_name is None:
            return None
        try:
            return float(raw_data[self.target_name])
        except KeyError as exc:
            raise ValueError(f"Missing configured target: {self.target_name}") from exc

    def _get_metadata(self, raw_data: dict[str, Any], index: int) -> dict[str, Any]:
        return self.source.get_metadata(index)
