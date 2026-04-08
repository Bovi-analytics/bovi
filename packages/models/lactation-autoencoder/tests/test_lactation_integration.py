"""Integration tests for lactation data pipeline."""

import json
import pickle

import numpy as np
import pytest
from bovi_core.ml.dataloaders.sources import TransformedSource
from bovi_core.ml.dataloaders.transforms.timeseries import (
    ImputationTransform,
    SequenceNormalizationTransform,
)

from lactation_autoencoder.dataloaders.datasets.lactation_dataset import LactationDataset
from lactation_autoencoder.dataloaders.sources.lactation_pkl_source import LactationPKLSource
from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
    EventTokenizationTransform,
    HerdStatsEnrichmentTransform,
    HerdStatsNormalizationTransform,
    MilkNormalizationTransform,
)


@pytest.fixture
def complete_herd_stats_dir(tmp_path):
    """Create complete herd statistics directory with all required pickle files."""
    stats_dir = tmp_path / "herd_stats"
    stats_dir.mkdir()

    # Herd parameter index
    idx_to_herd_par = {i: f"param_{i}" for i in range(10)}
    with open(stats_dir / "idx_to_herd_par_dict.pkl", "wb") as f:
        pickle.dump(idx_to_herd_par, f)

    # Herd stats per parity (Level 1) — parity keys are strings
    herd_stats_per_parity = {
        1001: {
            "1": {f"param_{i}": v for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])},
            "2": {f"param_{i}": v for i, v in enumerate([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])},
        },
        1002: {
            "1": {f"param_{i}": v for i, v in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])},
        },
    }
    with open(stats_dir / "herd_stats_per_parity_dict.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity, f)

    # Herd stats means per herd (Level 2)
    herd_stats_per_herd = {
        1001: {f"param_{i}": v for i, v in enumerate([1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25, 9.25, 10.25])},
        1002: {f"param_{i}": v for i, v in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])},
    }
    with open(stats_dir / "herd_stats_means_per_herd.pkl", "wb") as f:
        pickle.dump(herd_stats_per_herd, f)

    # Herd stats means per parity (Level 3) — string keys
    herd_stats_per_parity_global = {
        "1": {f"param_{i}": v for i, v in enumerate([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])},
        "2": {f"param_{i}": v for i, v in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])},
    }
    with open(stats_dir / "herd_stat_means_per_parity.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity_global, f)

    # Global means (Level 4)
    herd_stat_means_global = {f"param_{i}": v for i, v in enumerate([1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75, 10.75])}
    with open(stats_dir / "herd_stat_means_global.pkl", "wb") as f:
        pickle.dump(herd_stat_means_global, f)

    return stats_dir


@pytest.fixture
def event_to_idx_path(tmp_path):
    """Create event_to_idx pickle file and return its path."""
    event_to_idx = {
        "milking": 0,
        "vaccination": 1,
        "treatment": 2,
        "heat": 3,
        "calving": 4,
    }
    path = tmp_path / "event_to_idx_dict.pkl"
    with open(path, "wb") as f:
        pickle.dump(event_to_idx, f)
    return path


@pytest.fixture
def complete_json_dir(tmp_path):
    """Create complete JSON data directory with multiple lactations."""
    json_dir = tmp_path / "jsons"
    json_dir.mkdir()

    # Create 3 lactations with different characteristics
    lactations = [
        {
            "animal_id": "cow_001",
            "herd_id": 1001,
            "parity": 1,
            "milk": [20.0, 21.0, 22.0, 23.0, 24.0] + [20.0] * 299,
            "events": ["milking"] * 304,
        },
        {
            "animal_id": "cow_002",
            "herd_id": 1001,
            "parity": 2,
            "milk": [25.0, 26.0, 27.0, 28.0, 29.0] + [25.0] * 299,
            "events": ["milking"] * 200 + ["vaccination"] * 50 + ["milking"] * 54,
        },
        {
            "animal_id": "cow_003",
            "herd_id": 1002,
            "parity": 1,
            "milk": [30.0, 31.0, 32.0, 33.0, 34.0] + [30.0] * 299,
            "events": ["milking"] * 304,
        },
    ]

    for i, lactation in enumerate(lactations, 1):
        with open(json_dir / f"animal_{i:03d}.json", "w") as f:
            json.dump(lactation, f)

    return json_dir


@pytest.fixture
def source(complete_json_dir):
    """Create LactationPKLSource (raw JSON, no enrichment)."""
    return LactationPKLSource(json_root_dir=complete_json_dir)


@pytest.fixture
def enriched_source(source, complete_herd_stats_dir):
    """Create a TransformedSource that enriches raw data with herd stats."""
    enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)
    return TransformedSource(source, [enrich])


class TestLactationDataPipelineBasic:
    """Test basic end-to-end lactation pipeline."""

    def test_source_to_dataset(self, enriched_source):
        """Test creating dataset from enriched source."""
        dataset = LactationDataset(enriched_source)

        assert len(dataset) == 3

    def test_dataset_item_structure(self, enriched_source):
        """Test dataset item has correct structure."""
        dataset = LactationDataset(enriched_source)
        item = dataset[0]

        # Check structure
        assert "features" in item
        assert "labels" in item
        assert "metadata" in item

        # Check features
        features = item["features"]
        assert "milk" in features
        assert "events" in features
        assert "parity" in features
        assert "herd_stats" in features

    def test_dataset_shapes(self, enriched_source):
        """Test all items have consistent shapes."""
        dataset = LactationDataset(enriched_source)

        for i in range(len(dataset)):
            item = dataset[i]

            assert item["features"]["milk"].shape == (304,)
            assert item["features"]["events"].shape == (304,)
            assert item["features"]["parity"].shape == (1,)
            assert item["features"]["herd_stats"].shape == (10,)
            assert item["labels"].shape == (304,)


class TestLactationDataPipelineWithTransforms:
    """Test lactation pipeline with transforms."""

    def test_pipeline_with_tokenization(self, source, complete_herd_stats_dir, event_to_idx_path):
        """Test pipeline with event tokenization."""
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)
        tokenize = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        transformed_source = TransformedSource(source, [enrich, tokenize])
        dataset = LactationDataset(transformed_source)
        item = dataset[0]

        # Events should be tokenized to integers
        events = item["features"]["events"]
        assert events.dtype == np.int32

    def test_pipeline_with_normalization(self, enriched_source):
        """Test pipeline with milk normalization."""
        normalize = MilkNormalizationTransform(max_milk=80.0)

        transformed_source = TransformedSource(enriched_source, [normalize])
        dataset = LactationDataset(transformed_source)
        item = dataset[0]

        # Milk should be normalized to 0-1 range (approximately)
        milk = item["features"]["milk"]
        assert milk.max() <= 1.0

    def test_pipeline_with_herd_stats_normalization(self, enriched_source):
        """Test pipeline with herd stats normalization."""
        normalize = HerdStatsNormalizationTransform(method="zscore")

        transformed_source = TransformedSource(enriched_source, [normalize])
        dataset = LactationDataset(transformed_source)
        item = dataset[0]

        # Herd stats should be z-score normalized
        herd_stats = item["features"]["herd_stats"]
        assert np.isclose(herd_stats.mean(), 0.0, atol=1e-5)

    def test_full_transform_pipeline(self, source, complete_herd_stats_dir, event_to_idx_path):
        """Test full transform pipeline with all transforms."""
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)
        tokenize = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)
        milk_norm = MilkNormalizationTransform(max_milk=80.0)
        herd_norm = HerdStatsNormalizationTransform(method="zscore")

        # Apply enrichment first, then the rest via dataset
        enriched = TransformedSource(source, [enrich])
        dataset = LactationDataset(enriched)

        # Manually apply remaining transforms to first item
        item = dataset[0]
        item = tokenize(item)
        item = milk_norm(item)
        item = herd_norm(item)

        # Check results
        assert item["features"]["events"].dtype == np.int32
        assert item["features"]["milk"].max() <= 1.0
        assert np.isclose(item["features"]["herd_stats"].mean(), 0.0, atol=1e-5)


class TestLactationDataPipelineWithTimeSeriesTransforms:
    """Test lactation pipeline with time-series transforms."""

    def test_pipeline_with_imputation(self, enriched_source):
        """Test pipeline with imputation (though lactation data shouldn't have NaNs)."""
        impute = ImputationTransform(method="forward_fill")

        transformed_source = TransformedSource(enriched_source, [impute])
        dataset = LactationDataset(transformed_source)
        item = dataset[0]

        # Should still work without errors
        assert "features" in item

    def test_pipeline_with_sequence_normalization(self, enriched_source):
        """Test pipeline with sequence normalization."""
        normalize = SequenceNormalizationTransform(method="zscore")

        transformed_source = TransformedSource(enriched_source, [normalize])
        dataset = LactationDataset(transformed_source)
        item = dataset[0]

        # Milk sequence should be z-score normalized
        milk = item["features"]["milk"]
        # Note: This is a full sequence, not individual points
        assert milk.dtype == np.float32


class TestLactationDataPipelineMultipleSamples:
    """Test lactation pipeline with multiple samples."""

    def test_batching_consistency(self, enriched_source):
        """Test that dataset can be used for batching."""
        dataset = LactationDataset(enriched_source)

        # Get multiple items
        items = [dataset[i] for i in range(len(dataset))]

        # All items should have same structure
        assert len(items) == 3
        for item in items:
            assert item["features"]["milk"].shape == (304,)
            assert item["features"]["herd_stats"].shape == (10,)

    def test_different_parities(self, enriched_source):
        """Test dataset handles different parities."""
        dataset = LactationDataset(enriched_source)

        parities = [dataset[i]["features"]["parity"][0] for i in range(len(dataset))]

        # Should have at least parity 1 and 2
        assert 1.0 in parities
        assert 2.0 in parities

    def test_different_herds(self, enriched_source):
        """Test dataset handles different herds."""
        dataset = LactationDataset(enriched_source)

        herds = set()
        for i in range(len(dataset)):
            item = dataset[i]
            herds.add(item["metadata"]["herd_id"])

        # Should have at least herd 1001 and 1002
        assert len(herds) >= 2


class TestLactationDataPipelineDataTypes:
    """Test data types throughout the pipeline."""

    def test_all_dtypes_correct(self, enriched_source):
        """Test all output dtypes are correct."""
        dataset = LactationDataset(enriched_source)
        item = dataset[0]

        assert item["features"]["milk"].dtype == np.float32
        assert item["features"]["events"].dtype == np.int32
        assert item["features"]["parity"].dtype == np.float32
        assert item["features"]["herd_stats"].dtype == np.float32
        assert item["labels"].dtype == np.float32

    def test_dtypes_after_transforms(self, source, complete_herd_stats_dir, event_to_idx_path):
        """Test dtypes are preserved after transforms."""
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)
        tokenize = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)
        milk_norm = MilkNormalizationTransform(max_milk=80.0)

        enriched = TransformedSource(source, [enrich])
        dataset = LactationDataset(enriched)
        item = dataset[0]

        # Apply transforms
        item = tokenize(item)
        item = milk_norm(item)

        # Check dtypes
        assert item["features"]["events"].dtype == np.int32
        assert item["features"]["milk"].dtype == np.float32


class TestLactationDataPipelineEdgeCases:
    """Test edge cases in the lactation pipeline."""

    def test_single_sample(self, tmp_path, complete_herd_stats_dir):
        """Test pipeline with single sample."""
        # Create directory with single lactation
        json_dir = tmp_path / "single_json"
        json_dir.mkdir()

        lactation = {
            "animal_id": "cow_single",
            "herd_id": 1001,
            "parity": 1,
            "milk": [20.0] * 304,
            "events": ["milking"] * 304,
        }

        with open(json_dir / "animal_001.json", "w") as f:
            json.dump(lactation, f)

        source = LactationPKLSource(json_root_dir=json_dir)
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)
        enriched = TransformedSource(source, [enrich])
        dataset = LactationDataset(enriched)

        assert len(dataset) == 1
        item = dataset[0]
        assert item["features"]["milk"].shape == (304,)

    def test_consistent_output_shapes(self, enriched_source):
        """Test all samples have consistent output shapes."""
        dataset = LactationDataset(enriched_source)

        shapes = {
            "milk": [],
            "events": [],
            "parity": [],
            "herd_stats": [],
            "labels": [],
        }

        for i in range(len(dataset)):
            item = dataset[i]
            shapes["milk"].append(item["features"]["milk"].shape)
            shapes["events"].append(item["features"]["events"].shape)
            shapes["parity"].append(item["features"]["parity"].shape)
            shapes["herd_stats"].append(item["features"]["herd_stats"].shape)
            shapes["labels"].append(item["labels"].shape)

        # All shapes should be consistent
        for key in shapes:
            unique_shapes = set(shapes[key])
            assert len(unique_shapes) == 1, f"Inconsistent shapes for {key}: {unique_shapes}"

    def test_metadata_consistency(self, enriched_source):
        """Test metadata is consistent across samples."""
        dataset = LactationDataset(enriched_source)

        required_metadata = {"animal_id", "herd_id", "parity", "index"}

        for i in range(len(dataset)):
            item = dataset[i]
            metadata = item["metadata"]

            # Check all required keys present
            for key in required_metadata:
                assert key in metadata, f"Missing metadata key: {key}"

            # Check values are reasonable
            assert isinstance(metadata["animal_id"], str)
            assert isinstance(metadata["herd_id"], (int, np.integer))
            assert isinstance(metadata["parity"], (int, np.integer, float, np.floating))
            assert isinstance(metadata["index"], int)


class TestLactationDataPipelineMemory:
    """Test memory-related aspects of the pipeline."""

    def test_in_memory_vs_lazy_loading(self, complete_json_dir, complete_herd_stats_dir):
        """Test both memory modes produce same results."""
        enrich = HerdStatsEnrichmentTransform(herd_stats_dir=complete_herd_stats_dir)

        # In-memory
        source_mem = LactationPKLSource(
            json_root_dir=complete_json_dir,
            keep_in_memory=True,
        )
        enriched_mem = TransformedSource(source_mem, [enrich])
        dataset_mem = LactationDataset(enriched_mem)

        # Lazy-loading
        source_lazy = LactationPKLSource(
            json_root_dir=complete_json_dir,
            keep_in_memory=False,
        )
        enriched_lazy = TransformedSource(source_lazy, [enrich])
        dataset_lazy = LactationDataset(enriched_lazy)

        # Compare first item
        item_mem = dataset_mem[0]
        item_lazy = dataset_lazy[0]

        assert np.allclose(item_mem["features"]["milk"], item_lazy["features"]["milk"])
        assert np.allclose(item_mem["features"]["herd_stats"], item_lazy["features"]["herd_stats"])
