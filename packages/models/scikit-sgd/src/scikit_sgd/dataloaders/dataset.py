"""Feature-vector dataset for the scikit SGD regression example."""

from __future__ import annotations

from typing import Any

from bovi_core.config import Config
from bovi_core.ml import DataSource
from bovi_core.ml.dataloaders.datasets import FeatureVectorDataset


class ScikitRegressionDataset(FeatureVectorDataset):
    """Select configured numeric features and one target from source records."""

    def __init__(
        self,
        source: DataSource[dict[str, Any]],
        feature_names: tuple[str, ...],
        target_name: str,
        config: Config | None = None,
    ) -> None:
        super().__init__(source=source, config=config, feature_keys=list(feature_names))
        self.feature_names = feature_names
        self.target_name = target_name

    def _get_features(self, raw_data: dict[str, Any]) -> dict[str, float]:
        try:
            return {name: float(raw_data[name]) for name in self.feature_names}
        except KeyError as exc:
            raise ValueError(f"Missing configured feature: {exc.args[0]}") from exc

    def _get_labels(self, raw_data: dict[str, Any]) -> float:
        try:
            return float(raw_data[self.target_name])
        except KeyError as exc:
            raise ValueError(f"Missing configured target: {self.target_name}") from exc

    def _get_metadata(self, raw_data: dict[str, Any], index: int) -> dict[str, Any]:
        return {"index": index, "sample_id": raw_data.get("sample_id")}
