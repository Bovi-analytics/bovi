"""Tests for FeatureVectorDataset base class."""

import pytest
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path

from bovi_core.ml.dataloaders.datasets.feature_vector_dataset import FeatureVectorDataset
from bovi_core.ml.dataloaders.base.data_source import DataSource


class MockDataSource(DataSource):
    """Mock data source for testing."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self._data = {
            i: {
                "value": float(i),
                "squared": float(i ** 2),
                "id": f"sample_{i}"
            }
            for i in range(num_samples)
        }

    def __len__(self) -> int:
        return self.num_samples

    def load_item(self, index: Union[int, str]) -> Dict[str, Any]:
        """Load raw data for a single item."""
        if isinstance(index, str):
            index = int(index)
        # Handle negative indexing
        if index < 0:
            index = self.num_samples + index
        # Bounds check
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_samples}")
        return self._data[index]

    def get_metadata(self, index: Union[int, str]) -> Dict[str, Any]:
        """Get metadata for a single item."""
        if isinstance(index, str):
            index = int(index)
        return {"id": f"sample_{index}", "index": index}

    def get_keys(self) -> List[Union[int, str]]:
        """Get list of all available keys."""
        return list(range(self.num_samples))


class SimpleFeatureDataset(FeatureVectorDataset):
    """Simple implementation for testing."""

    def _get_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "value": np.array([raw_data["value"]], dtype=np.float32),
            "squared": np.array([raw_data["squared"]], dtype=np.float32),
        }

    def _get_labels(self, raw_data: Dict[str, Any]) -> np.ndarray:
        return np.array([raw_data["value"]], dtype=np.float32)

    def _get_metadata(self, raw_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "id": raw_data["id"],
            "index": index,
        }


class SequenceFeatureDataset(FeatureVectorDataset):
    """Sequence implementation for testing time-series."""

    def __init__(self, data_source: DataSource, sequence_length: int = 10):
        super().__init__(data_source)
        self.sequence_length = sequence_length

    def _get_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        base_value = raw_data["value"]
        # Create sequence: [base_value, base_value+1, base_value+2, ...]
        sequence = np.arange(
            base_value,
            base_value + self.sequence_length,
            dtype=np.float32
        )
        return {
            "time_series": sequence,
            "base_value": np.array([base_value], dtype=np.float32),
        }

    def _get_labels(self, raw_data: Dict[str, Any]) -> np.ndarray:
        # Label is the sum of the sequence
        base_value = raw_data["value"]
        sequence = np.arange(base_value, base_value + self.sequence_length)
        return np.array([sequence.sum()], dtype=np.float32)


@pytest.fixture
def mock_data_source():
    """Create mock data source."""
    return MockDataSource(num_samples=10)


@pytest.fixture
def simple_dataset(mock_data_source):
    """Create simple feature dataset."""
    return SimpleFeatureDataset(mock_data_source)


@pytest.fixture
def sequence_dataset(mock_data_source):
    """Create sequence feature dataset."""
    return SequenceFeatureDataset(mock_data_source, sequence_length=10)


class TestFeatureVectorDatasetBasics:
    """Test basic FeatureVectorDataset functionality."""

    def test_dataset_length(self, simple_dataset):
        """Test dataset returns correct length."""
        assert len(simple_dataset) == 10

    def test_dataset_getitem_basic(self, simple_dataset):
        """Test basic getitem returns correct structure."""
        item = simple_dataset[0]

        assert isinstance(item, dict)
        assert "features" in item
        assert "labels" in item

    def test_dataset_getitem_features_structure(self, simple_dataset):
        """Test features are in correct structure."""
        item = simple_dataset[0]

        features = item["features"]
        assert isinstance(features, dict)
        assert "value" in features
        assert "squared" in features

    def test_dataset_getitem_features_dtype(self, simple_dataset):
        """Test feature dtypes are correct."""
        item = simple_dataset[0]

        features = item["features"]
        assert features["value"].dtype == np.float32
        assert features["squared"].dtype == np.float32

    def test_dataset_getitem_features_values(self, simple_dataset):
        """Test feature values are correct."""
        item = simple_dataset[0]

        features = item["features"]
        assert np.allclose(features["value"], [0.0])
        assert np.allclose(features["squared"], [0.0])

    def test_dataset_getitem_labels(self, simple_dataset):
        """Test labels are in correct format."""
        item = simple_dataset[0]

        assert "labels" in item
        labels = item["labels"]
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.float32

    def test_dataset_getitem_labels_values(self, simple_dataset):
        """Test label values are correct."""
        item = simple_dataset[3]

        assert np.allclose(item["labels"], [3.0])

    def test_dataset_getitem_with_metadata(self, simple_dataset):
        """Test metadata is included."""
        item = simple_dataset[5]

        assert "metadata" in item
        assert isinstance(item["metadata"], dict)

    def test_dataset_getitem_metadata_content(self, simple_dataset):
        """Test metadata content is correct."""
        item = simple_dataset[5]

        metadata = item["metadata"]
        assert metadata["id"] == "sample_5"
        assert metadata["index"] == 5

    def test_dataset_iteration(self, simple_dataset):
        """Test iterating over dataset."""
        items = list(simple_dataset)

        assert len(items) == 10
        for i, item in enumerate(items):
            assert "features" in item
            assert "labels" in item
            assert item["metadata"]["index"] == i

    def test_dataset_negative_indexing(self, simple_dataset):
        """Test negative indexing works."""
        item_last = simple_dataset[-1]
        item_9 = simple_dataset[9]

        assert item_last["metadata"]["index"] == 9
        assert item_last["labels"] == item_9["labels"]

    def test_dataset_out_of_bounds(self, simple_dataset):
        """Test out of bounds indexing raises error."""
        with pytest.raises(IndexError):
            simple_dataset[100]


class TestFeatureVectorDatasetSequence:
    """Test FeatureVectorDataset with sequence/time-series data."""

    def test_sequence_dataset_length(self, sequence_dataset):
        """Test sequence dataset length."""
        assert len(sequence_dataset) == 10

    def test_sequence_features_shape(self, sequence_dataset):
        """Test sequence features have correct shape."""
        item = sequence_dataset[0]

        features = item["features"]
        assert features["time_series"].shape == (10,)

    def test_sequence_features_values(self, sequence_dataset):
        """Test sequence features have correct values."""
        item = sequence_dataset[3]

        time_series = item["features"]["time_series"]
        expected = np.arange(3, 13, dtype=np.float32)
        assert np.allclose(time_series, expected)

    def test_sequence_labels_correct(self, sequence_dataset):
        """Test sequence labels are correct (sum of sequence)."""
        item = sequence_dataset[2]

        # Sequence is [2, 3, 4, ..., 11], sum is 65
        expected_sum = sum(range(2, 12))
        assert np.allclose(item["labels"], [expected_sum])

    def test_sequence_batch_consistency(self, sequence_dataset):
        """Test multiple items have consistent structure."""
        items = [sequence_dataset[i] for i in range(5)]

        for item in items:
            assert item["features"]["time_series"].shape == (10,)
            assert item["labels"].shape == (1,)


class TestFeatureVectorDatasetEdgeCases:
    """Test edge cases for FeatureVectorDataset."""

    def test_empty_dataset(self):
        """Test dataset with zero samples."""
        empty_source = MockDataSource(num_samples=0)
        dataset = SimpleFeatureDataset(empty_source)

        assert len(dataset) == 0

    def test_single_sample_dataset(self):
        """Test dataset with single sample."""
        source = MockDataSource(num_samples=1)
        dataset = SimpleFeatureDataset(source)

        assert len(dataset) == 1
        item = dataset[0]
        assert item["metadata"]["index"] == 0

    def test_large_dataset_index(self):
        """Test dataset with large number of samples."""
        source = MockDataSource(num_samples=1000)
        dataset = SimpleFeatureDataset(source)

        assert len(dataset) == 1000
        item = dataset[500]
        assert item["metadata"]["index"] == 500

    def test_multiple_feature_types(self):
        """Test dataset with multiple feature types."""
        class MultiTypeDataset(FeatureVectorDataset):
            def _get_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "float_feature": np.array([1.5], dtype=np.float32),
                    "int_feature": np.array([42], dtype=np.int32),
                    "array_feature": np.array([1, 2, 3], dtype=np.float32),
                }

            def _get_labels(self, raw_data: Dict[str, Any]) -> np.ndarray:
                return np.array([1.0], dtype=np.float32)

        source = MockDataSource(num_samples=5)
        dataset = MultiTypeDataset(source)
        item = dataset[0]

        features = item["features"]
        assert features["float_feature"].dtype == np.float32
        assert features["int_feature"].dtype == np.int32
        assert features["array_feature"].dtype == np.float32


class TestFeatureVectorDatasetMetadata:
    """Test metadata handling in FeatureVectorDataset."""

    def test_metadata_optional(self):
        """Test dataset where metadata is not implemented."""
        class NoMetadataDataset(FeatureVectorDataset):
            def _get_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"value": np.array([1.0], dtype=np.float32)}

            def _get_labels(self, raw_data: Dict[str, Any]) -> np.ndarray:
                return np.array([1.0], dtype=np.float32)

            # _get_metadata not overridden, so returns None

        source = MockDataSource(num_samples=5)
        dataset = NoMetadataDataset(source)
        item = dataset[0]

        # When _get_metadata returns None, metadata key should not be in result
        assert "metadata" not in item
        # But features and labels should still be present
        assert "features" in item
        assert "labels" in item

    def test_metadata_is_dict(self):
        """Test metadata is always a dictionary."""
        item = SimpleFeatureDataset(MockDataSource())[0]

        assert isinstance(item["metadata"], dict)

    def test_metadata_preserves_custom_values(self, simple_dataset):
        """Test custom metadata values are preserved."""
        item = simple_dataset[7]

        assert item["metadata"]["id"] == "sample_7"
        assert item["metadata"]["index"] == 7
