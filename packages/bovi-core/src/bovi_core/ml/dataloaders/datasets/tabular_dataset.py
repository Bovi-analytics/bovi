"""Turn dictionary records into named numeric features and a scalar target.

Each dataset item represents one record, not a table or batch. This is a
concrete FeatureVectorDataset for scalar regression: field names determine
what goes into the model and what the model should predict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .feature_vector_dataset import FeatureVectorDataset

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.sources.base_source import DataSource


class TabularDataset(FeatureVectorDataset):
    """Select numeric input fields and an optional target from each source record.

    FeatureVectorDataset implements __len__ and __getitem__: it loads a record,
    calls the three extraction methods below, and assembles the sample. This
    class only defines which fields to select and converts their values to
    Python floats. It does not batch records or create framework tensors.

    Args:
        source: Source returning one dictionary per record, such as DictSource
            or JSONRecordsSource. Extra fields are ignored.
        feature_names: Non-empty, unique input field names in model feature
            order. Each selected value must be convertible to a scalar float.
        target_name: Field the model should predict; defaults to "y". Use None
            when no target is available, for example during inference.
        config: Optional config retained by the base dataset. Field names are
            passed explicitly; this class does not look them up in YAML.

    Returns:
        Accessing dataset[index] returns a dictionary with:
        - "features": Selected field names mapped to float values.
        - "labels": The target as a float, or None when target_name is None.
        - "metadata": Metadata returned by source.get_metadata(index).

    Example:
        >>> from bovi_core.ml.dataloaders.sources import DictSource
        >>> source = DictSource([{"x": 2, "y": 5, "note": "unused"}])
        >>> dataset = TabularDataset(source, feature_names=("x",), target_name="y")
        >>> dataset[0]
        {'features': {'x': 2.0}, 'labels': 5.0, 'metadata': {'index': 0}}
        >>> TabularDataset(source, feature_names=("x",), target_name=None)[0]["labels"] is None
        True

    Apply record transforms with TransformedSource before feature selection.
    Loaders combine samples into batches; model_inputs helpers subsequently
    turn feature columns into a model's matrix representation. Categorical
    encoding, missing-value handling and multi-output targets are not provided
    here; use explicit transforms or a different FeatureVectorDataset subclass.
    """

    def __init__(
        self,
        source: DataSource[dict[str, Any]],
        feature_names: Sequence[str],
        target_name: str | None = "y",
        config: Config | None = None,
    ) -> None:
        """Validate feature names and retain the source without reading records."""
        names = tuple(feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain non-empty field names")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        super().__init__(source=source, config=config)
        self.feature_names = names
        self.target_name = target_name

    def _get_features(self, raw_data: dict[str, Any]) -> dict[str, float]:
        """Select input fields in the configured order and convert them to floats.

        Missing fields raise ValueError. Values that cannot be converted to
        float propagate the conversion error; no imputation is performed.
        """
        try:
            return {name: float(raw_data[name]) for name in self.feature_names}
        except KeyError as exc:
            raise ValueError(f"Missing configured feature: {exc.args[0]}") from exc

    def _get_labels(self, raw_data: dict[str, Any]) -> float | None:
        """Read the target field, or return None when targets are disabled.

        A configured but missing target is an error, not an unlabeled sample.
        """
        if self.target_name is None:
            return None
        try:
            return float(raw_data[self.target_name])
        except KeyError as exc:
            raise ValueError(f"Missing configured target: {self.target_name}") from exc

    def _get_metadata(self, raw_data: dict[str, Any], index: int) -> dict[str, Any]:
        """Use source metadata, not arbitrary extra fields from the record."""
        return self.source.get_metadata(index)
