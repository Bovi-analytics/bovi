"""Tests for LactationDataset."""

import json
import pickle

import numpy as np
import pytest
from bovi_core.ml.dataloaders.sources import TransformedSource

from lactation_autoencoder.dataloaders.datasets.lactation_dataset import LactationDataset
from lactation_autoencoder.dataloaders.sources.lactation_pkl_source import LactationPKLSource
from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
    HerdStatsEnrichmentTransform,
)


@pytest.fixture
def herd_stats_dir(tmp_path):
    """Create mock herd statistics directory."""
    stats_dir = tmp_path / "herd_stats"
    stats_dir.mkdir()

    # Create mock herd parameter index
    idx_to_herd_par = {i: f"param_{i}" for i in range(10)}
    with open(stats_dir / "idx_to_herd_par_dict.pkl", "wb") as f:
        pickle.dump(idx_to_herd_par, f)

    # Create herd stats per parity (CRITICAL: parity must be strings!)
    herd_stats_per_parity = {
        1001: {
            "1": {
                "param_0": 1.0,
                "param_1": 2.0,
                "param_2": 3.0,
                "param_3": 4.0,
                "param_4": 5.0,
                "param_5": 6.0,
                "param_6": 7.0,
                "param_7": 8.0,
                "param_8": 9.0,
                "param_9": 10.0,
            },
        },
    }
    with open(stats_dir / "herd_stats_per_parity_dict.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity, f)

    # Create herd stats means per herd
    herd_stats_per_herd = {
        1001: {
            "param_0": 1.25,
            "param_1": 2.25,
            "param_2": 3.25,
            "param_3": 4.25,
            "param_4": 5.25,
            "param_5": 6.25,
            "param_6": 7.25,
            "param_7": 8.25,
            "param_8": 9.25,
            "param_9": 10.25,
        },
    }
    with open(stats_dir / "herd_stats_means_per_herd.pkl", "wb") as f:
        pickle.dump(herd_stats_per_herd, f)

    # Create herd stats means per parity (parity must be strings!)
    herd_stats_per_parity_global = {
        "1": {
            "param_0": 1.5,
            "param_1": 2.5,
            "param_2": 3.5,
            "param_3": 4.5,
            "param_4": 5.5,
            "param_5": 6.5,
            "param_6": 7.5,
            "param_7": 8.5,
            "param_8": 9.5,
            "param_9": 10.5,
        },
    }
    with open(stats_dir / "herd_stat_means_per_parity.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity_global, f)

    # Create global herd stats means
    herd_stat_means_global = {
        "param_0": 1.75,
        "param_1": 2.75,
        "param_2": 3.75,
        "param_3": 4.75,
        "param_4": 5.75,
        "param_5": 6.75,
        "param_6": 7.75,
        "param_7": 8.75,
        "param_8": 9.75,
        "param_9": 10.75,
    }
    with open(stats_dir / "herd_stat_means_global.pkl", "wb") as f:
        pickle.dump(herd_stat_means_global, f)

    return stats_dir


@pytest.fixture
def json_data_dir(tmp_path):
    """Create mock JSON data directory."""
    json_dir = tmp_path / "jsons"
    json_dir.mkdir()

    lactation = {
        "animal_id": "cow_001",
        "herd_id": 1001,
        "parity": 1,
        "milk": [20.0, 21.0, 22.0, 23.0, 24.0] + [20.0] * 299,
        "events": ["milking"] * 304,
    }

    with open(json_dir / "animal_001.json", "w") as f:
        json.dump(lactation, f)

    return json_dir


@pytest.fixture
def source(json_data_dir, herd_stats_dir):
    """Create enriched source (raw source + herd stats enrichment transform)."""
    raw_source = LactationPKLSource(json_root_dir=json_data_dir)
    enrich = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
    return TransformedSource(raw_source, [enrich])


@pytest.fixture
def dataset(source):
    """Create LactationDataset."""
    return LactationDataset(source, max_days=304)


class TestLactationDatasetBasic:
    """Test basic LactationDataset functionality."""

    def test_dataset_length(self, dataset):
        """Test dataset returns correct length."""
        assert len(dataset) == 1

    def test_getitem_returns_dict(self, dataset):
        """Test getitem returns dictionary."""
        item = dataset[0]

        assert isinstance(item, dict)
        assert "features" in item
        assert "labels" in item

    def test_getitem_features_structure(self, dataset):
        """Test features have correct structure."""
        item = dataset[0]

        features = item["features"]
        assert isinstance(features, dict)
        assert "milk" in features
        assert "events" in features
        assert "parity" in features
        assert "herd_stats" in features

    def test_getitem_features_milk_shape(self, dataset):
        """Test milk feature has correct shape."""
        item = dataset[0]

        milk = item["features"]["milk"]
        assert milk.shape == (304,)
        assert milk.dtype == np.float32

    def test_getitem_features_events_shape(self, dataset):
        """Test events feature has correct shape."""
        item = dataset[0]

        events = item["features"]["events"]
        assert events.shape == (304,)
        assert events.dtype == np.int32

    def test_getitem_features_parity_shape(self, dataset):
        """Test parity feature has correct shape."""
        item = dataset[0]

        parity = item["features"]["parity"]
        assert parity.shape == (1,)
        assert parity.dtype == np.float32

    def test_getitem_features_herd_stats_shape(self, dataset):
        """Test herd_stats feature has correct shape."""
        item = dataset[0]

        herd_stats = item["features"]["herd_stats"]
        assert herd_stats.shape == (10,)
        assert herd_stats.dtype == np.float32

    def test_getitem_labels_shape(self, dataset):
        """Test labels have correct shape."""
        item = dataset[0]

        labels = item["labels"]
        assert labels.shape == (304,)
        assert labels.dtype == np.float32

    def test_getitem_labels_equals_features_milk(self, dataset):
        """Test labels equal features milk."""
        item = dataset[0]

        labels = item["labels"]
        milk = item["features"]["milk"]

        assert np.allclose(labels, milk)

    def test_getitem_metadata(self, dataset):
        """Test metadata is included."""
        item = dataset[0]

        assert "metadata" in item
        metadata = item["metadata"]
        assert isinstance(metadata, dict)

    def test_getitem_metadata_content(self, dataset):
        """Test metadata content."""
        item = dataset[0]

        metadata = item["metadata"]
        assert "animal_id" in metadata
        assert "herd_id" in metadata
        assert "parity" in metadata
        assert metadata["animal_id"] == "cow_001"


class TestLactationDatasetFeatureValues:
    """Test feature values are correct."""

    def test_milk_values_positive(self, dataset):
        """Test milk values are positive (normalized)."""
        item = dataset[0]

        milk = item["features"]["milk"]
        assert (milk >= 0).all()

    def test_milk_values_normalized(self, dataset):
        """Test milk values are within expected range."""
        item = dataset[0]

        milk = item["features"]["milk"]
        # Milk values are in raw range (20-24 in test data), will be normalized by transforms
        assert milk.max() <= 100.0  # Raw milk values are typically 0-100

    def test_events_are_integers(self, dataset):
        """Test events are integer indices."""
        item = dataset[0]

        events = item["features"]["events"]
        assert np.all(np.equal(np.mod(events, 1), 0))  # All integers

    def test_parity_value(self, dataset):
        """Test parity value."""
        item = dataset[0]

        parity = item["features"]["parity"]
        assert parity[0] == 1.0

    def test_herd_stats_are_positive(self, dataset):
        """Test herd stats are positive."""
        item = dataset[0]

        herd_stats = item["features"]["herd_stats"]
        assert (herd_stats > 0).all()


class TestLactationDatasetDataTypes:
    """Test data types are correct."""

    def test_milk_dtype(self, dataset):
        """Test milk is float32."""
        item = dataset[0]

        assert item["features"]["milk"].dtype == np.float32

    def test_events_dtype(self, dataset):
        """Test events is int32."""
        item = dataset[0]

        assert item["features"]["events"].dtype == np.int32

    def test_parity_dtype(self, dataset):
        """Test parity is float32."""
        item = dataset[0]

        assert item["features"]["parity"].dtype == np.float32

    def test_herd_stats_dtype(self, dataset):
        """Test herd_stats is float32."""
        item = dataset[0]

        assert item["features"]["herd_stats"].dtype == np.float32

    def test_labels_dtype(self, dataset):
        """Test labels are float32."""
        item = dataset[0]

        assert item["labels"].dtype == np.float32


class TestLactationDatasetMaxDays:
    """Test max_days parameter."""

    def test_max_days_default(self, source):
        """Test default max_days is 304."""
        dataset = LactationDataset(source)

        item = dataset[0]
        assert item["features"]["milk"].shape == (304,)

    def test_max_days_custom(self, source):
        """Test custom max_days."""
        dataset = LactationDataset(source, max_days=100)

        item = dataset[0]
        assert item["features"]["milk"].shape == (100,)

    def test_max_days_truncates(self, source):
        """Test max_days truncates sequences."""
        dataset_100 = LactationDataset(source, max_days=100)
        dataset_304 = LactationDataset(source, max_days=304)

        item_100 = dataset_100[0]
        item_304 = dataset_304[0]

        # First 100 values should match
        assert np.allclose(item_100["features"]["milk"], item_304["features"]["milk"][:100])

    def test_max_days_pads(self, source):
        """Test max_days pads short sequences."""
        dataset = LactationDataset(source, max_days=400)

        item = dataset[0]
        # Original is 304 days, should be padded to 400
        assert item["features"]["milk"].shape == (400,)


class TestLactationDatasetIndexing:
    """Test indexing behavior."""

    def test_negative_indexing(self, dataset):
        """Test negative indexing."""
        item = dataset[-1]

        assert "features" in item
        assert "labels" in item

    def test_out_of_bounds(self, dataset):
        """Test out of bounds indexing."""
        with pytest.raises(IndexError):
            dataset[100]

    def test_iteration(self, dataset):
        """Test iterating over dataset."""
        items = list(dataset)

        assert len(items) == 1
        assert "features" in items[0]


class TestLactationDatasetIntegration:
    """Test integration with transforms."""

    def test_dataset_without_transforms(self, dataset):
        """Test dataset works without transforms."""
        item = dataset[0]

        assert item["features"]["milk"].dtype == np.float32
        assert item["features"]["events"].dtype == np.int32

    def test_dataset_structure_consistency(self, dataset):
        """Test dataset structure is consistent across multiple calls."""
        item1 = dataset[0]
        item2 = dataset[0]

        # Keys should be the same
        assert set(item1["features"].keys()) == set(item2["features"].keys())

        # Shapes should be the same
        for key in item1["features"]:
            assert item1["features"][key].shape == item2["features"][key].shape


class TestLactationDatasetBatching:
    """Test dataset is suitable for batching."""

    def test_all_samples_same_shape(self, json_data_dir, herd_stats_dir):
        """Test all samples have same shape for batching."""
        # Create multiple lactations
        json_dir = json_data_dir
        for i in range(2, 4):
            lactation = {
                "animal_id": f"cow_{i:03d}",
                "herd_id": 1001,
                "parity": 1,
                "milk": [20.0 + i] * 304,
                "events": ["milking"] * 304,
            }
            with open(json_dir / f"animal_{i:03d}.json", "w") as f:
                json.dump(lactation, f)

        raw_source = LactationPKLSource(json_root_dir=json_dir)
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        source = TransformedSource(raw_source, [enrich])
        dataset = LactationDataset(source)

        # All samples should have same shape
        shapes = [dataset[i]["features"]["milk"].shape for i in range(len(dataset))]
        assert len(set(shapes)) == 1  # All shapes are the same
