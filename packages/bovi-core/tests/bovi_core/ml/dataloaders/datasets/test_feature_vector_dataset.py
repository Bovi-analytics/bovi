"""Tests for FeatureVectorDataset base class."""

from typing import Any, Dict, List, Union

import numpy as np
import pytest
from bovi_core.ml.dataloaders.base.data_source import DataSource
from bovi_core.ml.dataloaders.datasets.feature_vector_dataset import FeatureVectorDataset


class MockDataSource(DataSource):
    """Mock data source for testing."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self._data = {
            i: {"value": float(i), "squared": float(i**2), "id": f"sample_{i}"}
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
        sequence = np.arange(base_value, base_value + self.sequence_length, dtype=np.float32)
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


def test_input_example_matches_numpy_loader_nested_batch(simple_dataset, mock_dataloader_config):
    from bovi_core.ml.dataloaders import SklearnDataLoader

    loader = SklearnDataLoader(
        simple_dataset, mock_dataloader_config, model_name="test_model", batch_size=3, shuffle=False
    )
    batch = next(iter(loader))
    example = simple_dataset.get_input_example(n_samples=3)

    assert set(example) == set(batch)
    for name in batch["features"]:
        np.testing.assert_array_equal(example["features"][name], batch["features"][name])
    np.testing.assert_array_equal(example["labels"], batch["labels"])
    assert example["metadata"] == batch["metadata"]


def test_tabular_dataset_selects_named_features_and_source_metadata(mock_data_source):
    from bovi_core.ml.dataloaders.datasets import TabularDataset

    dataset = TabularDataset(mock_data_source, ("squared", "value"), "value")
    assert isinstance(dataset, FeatureVectorDataset)
    assert len(dataset) == 10
    sample = dataset[3]
    assert list(sample["features"]) == ["squared", "value"]
    assert sample["features"] == {"squared": 9.0, "value": 3.0}
    assert sample["labels"] == 3.0
    assert sample["metadata"] == {"id": "sample_3", "index": 3}
    assert dataset[-1]["metadata"]["index"] == 9


@pytest.mark.parametrize(
    "features, target, message",
    [
        (("missing",), "value", "Missing configured feature"),
        (("value",), "missing", "Missing configured target"),
    ],
)
def test_tabular_dataset_rejects_missing_fields(mock_data_source, features, target, message):
    from bovi_core.ml.dataloaders.datasets import TabularDataset

    with pytest.raises(ValueError, match=message):
        TabularDataset(mock_data_source, features, target)[0]


@pytest.mark.parametrize("features", [(), ("",), ("value", "value")])
def test_tabular_dataset_rejects_invalid_feature_names(mock_data_source, features):
    from bovi_core.ml.dataloaders.datasets import TabularDataset

    with pytest.raises(ValueError, match="feature_names"):
        TabularDataset(mock_data_source, features)


def test_tabular_dataset_supports_unlabelled_records(mock_data_source):
    from bovi_core.ml.dataloaders.datasets import TabularDataset

    sample = TabularDataset(mock_data_source, ("value",), target_name=None)[0]
    assert sample["labels"] is None


@pytest.mark.parametrize("index", [-4, -3, 2, 10])
def test_tabular_dataset_rejects_out_of_range_indices_without_wrapping(index):
    from bovi_core.ml.dataloaders.datasets import TabularDataset
    from bovi_core.ml.dataloaders.sources import DictSource

    dataset = TabularDataset(DictSource([{"x": 1}, {"x": 2}]), ("x",), target_name=None)
    with pytest.raises(IndexError, match="out of range"):
        dataset[index]


def test_tabular_dataset_preserves_valid_negative_indices():
    from bovi_core.ml.dataloaders.datasets import TabularDataset
    from bovi_core.ml.dataloaders.sources import DictSource

    dataset = TabularDataset(DictSource([{"x": 1}, {"x": 2}]), ("x",), target_name=None)
    assert dataset[-2] == dataset[0]
    assert dataset[-1] == dataset[1]


@pytest.mark.parametrize("index", [-1, 0])
def test_empty_tabular_dataset_rejects_all_indices(index):
    from bovi_core.ml.dataloaders.datasets import TabularDataset
    from bovi_core.ml.dataloaders.sources import DictSource

    dataset = TabularDataset(DictSource([]), ("x",), target_name=None)
    with pytest.raises(IndexError, match="out of range"):
        dataset[index]
