"""
FeatureVectorDataset: Base class for tabular/time-series data.

Standard Output Structure:
{
    "features": Dict[str, Any],  # Input features as dict
    "labels": Any                # Target/ground truth (optional)
    "metadata": Dict[str, Any]   # Optional metadata
}

This structure makes it crystal clear:
- "features" = What goes into the model
- "labels" = What the model should predict
- "metadata" = Additional info (indices, ids, etc)

Note: Transforms are NOT applied in datasets - they are applied in DataLoaders
or manually in preprocessing. Datasets always return raw NumPy data.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..base.data_source import DataSource

from ..base.dataset import Dataset


class FeatureVectorDataset(Dataset):
    """
    Base class for feature-based datasets (tabular, time-series).

    Enforces standard structure:
    - "features": Dict of input features (raw NumPy)
    - "labels": Target/ground truth
    - "metadata": Optional additional info

    Subclasses must implement:
    - _get_features(raw_data): Define what constitutes features
    - _get_labels(raw_data): Define what constitutes labels/target

    Note: Transforms are NOT applied here. Use UniversalTransform in
    preprocessing or in DataLoaders.

    Example:
        class AgeWeightDataset(FeatureVectorDataset):
            def _get_features(self, raw):
                return {
                    "age": raw["age"],
                    "weight": raw["weight"]
                }

            def _get_labels(self, raw):
                return raw["diagnosis"]

        # Usage:
        item = dataset[0]
        X = item["features"]  # {"age": 25, "weight": 70}
        y = item["labels"]     # "healthy"
        metadata = item["metadata"]  # {"index": 0, ...}
    """

    def __init__(
        self,
        source: "DataSource",
        config: Optional[Any] = None,
        feature_keys: Optional[List[str]] = None,
    ):
        """
        Initialize feature vector dataset.

        Args:
            source: DataSource to load raw data
            config: Optional config instance
            feature_keys: Optional list of feature keys to include
                         (filters output of _get_features)
        """
        super().__init__(source, config)
        self.feature_keys = feature_keys

    def __len__(self) -> int:
        """Return number of items in dataset"""
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get item with standard structure.

        Returns:
            Dict with keys:
            - "features": Dict of input features
            - "labels": Target/ground truth (or None)
            - "metadata": Additional info (optional)
        """
        # Normalize negative indices
        if index < 0:
            index = len(self.source) + index

        # Load raw data from source
        raw_data = self.source.load_item(index)

        # Get features (subclass defines mapping)
        features = self._get_features(raw_data)

        # Filter features if specified
        if self.feature_keys is not None:
            features = {k: features[k] for k in self.feature_keys if k in features}

        # Get labels (subclass defines mapping)
        labels = self._get_labels(raw_data)

        # Get metadata (optional) - pass normalized index
        metadata = self._get_metadata(raw_data, index)

        # Return standard structure
        result = {
            "features": features,
            "labels": labels,
        }

        if metadata is not None:
            result["metadata"] = metadata

        return result

    @abstractmethod
    def _get_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from raw data.

        Subclasses implement this to define:
        - Which fields are features
        - How to structure them
        - Any feature-specific processing

        Args:
            raw_data: Raw data dict from source (after transforms)

        Returns:
            Dict of features (input to model)

        Example:
            def _get_features(self, raw):
                return {
                    "milk": raw["milk"],
                    "events": raw["events_encoded"],
                    "parity": raw["parity"]
                }
        """
        pass

    @abstractmethod
    def _get_labels(self, raw_data: Dict[str, Any]) -> Any:
        """
        Extract labels/target from raw data.

        Subclasses implement this to define:
        - What is the prediction target
        - Can return None for unsupervised learning

        Args:
            raw_data: Raw data dict from source (after transforms)

        Returns:
            Labels/target (can be None)

        Example:
            # Autoencoder (reconstruct input)
            def _get_labels(self, raw):
                return raw["milk"]

            # Classifier (predict class)
            def _get_labels(self, raw):
                return raw["disease_category"]

            # Unsupervised
            def _get_labels(self, raw):
                return None
        """
        pass

    def _get_metadata(self, raw_data: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from raw data.

        Optional method - subclasses can override to provide additional metadata.
        By default, returns None (no extra metadata).

        Args:
            raw_data: Raw data dict from source (after transforms)
            index: Item index in dataset

        Returns:
            Dict of metadata or None

        Example:
            def _get_metadata(self, raw, index):
                return {
                    "index": index,
                    "animal_id": raw.get("animal_id"),
                    "herd_id": raw.get("herd_id")
                }
        """
        return None
