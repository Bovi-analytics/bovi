"""Tests for lactation-specific transforms."""

import pickle

import numpy as np
import pytest
from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
    EventTokenizationTransform,
    HerdStatsEnrichmentTransform,
    HerdStatsNormalizationTransform,
    MilkNormalizationTransform,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_to_idx_mapping():
    """Event to index mapping."""
    return {
        "milking": 0,
        "vaccination": 1,
        "treatment": 2,
        "heat": 3,
        "calving": 4,
    }


@pytest.fixture
def event_to_idx_path(tmp_path, event_to_idx_mapping):
    """Write event_to_idx mapping to a pkl file and return the path."""
    path = tmp_path / "event_to_idx_dict.pkl"
    with open(path, "wb") as f:
        pickle.dump(event_to_idx_mapping, f)
    return path


@pytest.fixture
def sample_data():
    """Create sample data for transforms (no event_to_idx in data dict)."""
    return {
        "features": {
            "milk": np.array([20.0, 25.0, 30.0], dtype=np.float32),
            "events": ["milking", "vaccination", "milking"],
            "parity": np.array([1], dtype=np.float32),
            "herd_stats": np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32
            ),
        },
    }


@pytest.fixture
def herd_stats_dir(tmp_path):
    """Create a herd stats directory with all required pickle files."""
    stats_dir = tmp_path / "herd_stats"
    stats_dir.mkdir()

    # idx_to_herd_par
    idx_to_herd_par = {i: f"param_{i}" for i in range(10)}
    with open(stats_dir / "idx_to_herd_par_dict.pkl", "wb") as f:
        pickle.dump(idx_to_herd_par, f)

    # Level 1: herd + parity (parity keys are strings)
    herd_stats_per_parity = {
        1001: {
            "1": {f"param_{i}": float(i + 1) for i in range(10)},
        },
    }
    with open(stats_dir / "herd_stats_per_parity_dict.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity, f)

    # Level 2: herd only
    herd_stats_per_herd = {
        1001: {f"param_{i}": float(i + 10) for i in range(10)},
        1002: {f"param_{i}": float(i + 20) for i in range(10)},
    }
    with open(stats_dir / "herd_stats_means_per_herd.pkl", "wb") as f:
        pickle.dump(herd_stats_per_herd, f)

    # Level 3: parity only (string keys)
    herd_stats_per_parity_global = {
        "1": {f"param_{i}": float(i + 100) for i in range(10)},
        "2": {f"param_{i}": float(i + 110) for i in range(10)},
    }
    with open(stats_dir / "herd_stat_means_per_parity.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity_global, f)

    # Level 4: global
    herd_stat_means_global = {f"param_{i}": float(i + 200) for i in range(10)}
    with open(stats_dir / "herd_stat_means_global.pkl", "wb") as f:
        pickle.dump(herd_stat_means_global, f)

    return stats_dir


# ===========================================================================
# HerdStatsEnrichmentTransform tests
# ===========================================================================


class TestHerdStatsEnrichmentTransformInit:
    """Test HerdStatsEnrichmentTransform initialization."""

    def test_init_loads_stats(self, herd_stats_dir):
        """Test transform loads all pkl files at init."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)

        assert transform.idx_to_herd_par is not None
        assert transform.herd_stats_per_parity is not None
        assert transform.herd_stats_per_herd is not None
        assert transform.herd_stats_per_parity_global is not None
        assert transform.herd_stats_global is not None

    def test_init_missing_dir(self):
        """Test error when directory does not exist."""
        with pytest.raises(ValueError, match="Herd stats directory not found"):
            HerdStatsEnrichmentTransform(herd_stats_dir="/nonexistent/path")

    def test_get_params(self, herd_stats_dir):
        """Test get_params returns correct config."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        params = transform.get_params()

        assert params["name"] == "herd_stats_enrichment"
        assert params["herd_stats_dir"] == str(herd_stats_dir)

    def test_transform_registry(self, herd_stats_dir):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create(
            "herd_stats_enrichment", herd_stats_dir=str(herd_stats_dir)
        )
        assert isinstance(transform, HerdStatsEnrichmentTransform)


class TestHerdStatsEnrichmentFallback:
    """Test all 4 hierarchical fallback levels."""

    def test_level_1_herd_and_parity(self, herd_stats_dir):
        """Test Level 1: herd_id + parity exact match."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        data: dict[str, object] = {"herd_id": 1001, "parity": 1}

        result = transform(data)

        expected = np.array([float(i + 1) for i in range(10)], dtype=np.float32)
        assert "herd_stats" in result
        assert np.allclose(result["herd_stats"], expected)

    def test_level_2_herd_only(self, herd_stats_dir):
        """Test Level 2: herd_id only (parity not in Level 1)."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        # herd 1002 exists in Level 2 but not in Level 1
        data: dict[str, object] = {"herd_id": 1002, "parity": 1}

        result = transform(data)

        expected = np.array([float(i + 20) for i in range(10)], dtype=np.float32)
        assert np.allclose(result["herd_stats"], expected)

    def test_level_3_parity_only(self, herd_stats_dir):
        """Test Level 3: parity only (unknown herd)."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        # herd 9999 doesn't exist anywhere
        data: dict[str, object] = {"herd_id": 9999, "parity": 2}

        result = transform(data)

        expected = np.array([float(i + 110) for i in range(10)], dtype=np.float32)
        assert np.allclose(result["herd_stats"], expected)

    def test_level_4_global(self, herd_stats_dir):
        """Test Level 4: global fallback (unknown herd and parity)."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        data: dict[str, object] = {"herd_id": 9999, "parity": 99}

        result = transform(data)

        expected = np.array([float(i + 200) for i in range(10)], dtype=np.float32)
        assert np.allclose(result["herd_stats"], expected)

    def test_missing_herd_id_and_parity(self, herd_stats_dir):
        """Test fallback when herd_id and parity are missing."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        data: dict[str, object] = {"animal_id": "cow_x"}

        result = transform(data)

        # Should fall through to Level 4 (global)
        expected = np.array([float(i + 200) for i in range(10)], dtype=np.float32)
        assert np.allclose(result["herd_stats"], expected)

    def test_output_shape_and_dtype(self, herd_stats_dir):
        """Test output array shape and dtype."""
        transform = HerdStatsEnrichmentTransform(herd_stats_dir=herd_stats_dir)
        data: dict[str, object] = {"herd_id": 1001, "parity": 1}

        result = transform(data)

        assert result["herd_stats"].shape == (10,)
        assert result["herd_stats"].dtype == np.float32


# ===========================================================================
# EventTokenizationTransform tests
# ===========================================================================


class TestEventTokenizationTransform:
    """Test EventTokenizationTransform."""

    def test_transform_initialization_with_path(self, event_to_idx_path):
        """Test transform loads event_to_idx from pkl at init."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        assert transform.event_to_idx is not None
        assert "milking" in transform.event_to_idx

    def test_transform_initialization_without_path(self):
        """Test transform initializes without path (backward compat)."""
        transform = EventTokenizationTransform()

        assert transform.event_to_idx is None

    def test_tokenize_with_path(self, event_to_idx_path):
        """Test tokenization using pkl-loaded mapping."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "vaccination", "milking"],
            },
        }

        result = transform(data)

        events = result["features"]["events"]
        assert events[0] == 0  # milking
        assert events[1] == 1  # vaccination
        assert events[2] == 0  # milking
        assert events.dtype == np.int32

    def test_tokenize_with_data_dict_fallback(self, event_to_idx_mapping):
        """Test tokenization falling back to data dict event_to_idx."""
        transform = EventTokenizationTransform()

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "vaccination", "milking"],
            },
            "event_to_idx": event_to_idx_mapping,
        }

        result = transform(data)

        events = result["features"]["events"]
        assert events[0] == 0
        assert events[1] == 1

    def test_tokenize_single_event(self, event_to_idx_path):
        """Test tokenizing single event."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        sample: dict[str, object] = {
            "features": {
                "events": ["milking"],
                "milk": np.array([20.0], dtype=np.float32),
            },
        }

        result = transform(sample)
        assert result["features"]["events"][0] == 0

    def test_tokenize_multiple_events(self, event_to_idx_path):
        """Test tokenizing multiple events."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "vaccination", "milking"],
            },
        }

        result = transform(data)

        events = result["features"]["events"]
        assert events[0] == 0  # milking
        assert events[1] == 1  # vaccination
        assert events[2] == 0  # milking

    def test_tokenization_preserves_shape(self, event_to_idx_path):
        """Test tokenization preserves shape."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "vaccination", "treatment"],
            },
        }

        result = transform(data)
        assert len(result["features"]["events"]) == 3

    def test_unknown_event_handling(self, event_to_idx_path):
        """Test handling of unknown events."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "unknown_event", "vaccination"],
            },
        }

        result = transform(data)
        assert len(result["features"]["events"]) == 3

    def test_all_events_tokenized(self, event_to_idx_path):
        """Test all mapped events are tokenized correctly."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": ["milking", "vaccination", "treatment", "heat", "calving"],
            },
        }

        result = transform(data)
        events = result["features"]["events"]

        expected = [0, 1, 2, 3, 4]
        assert list(events) == expected

    def test_no_mapping_returns_data_unchanged(self):
        """Test that without mapping (neither path nor data dict) data is returned unchanged."""
        transform = EventTokenizationTransform()

        data: dict[str, object] = {
            "features": {
                "events": ["milking"],
            },
        }

        result = transform(data)
        # No mapping available, should return unchanged
        assert result["features"]["events"] == ["milking"]

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("event_tokenization")
        assert isinstance(transform, EventTokenizationTransform)


# ===========================================================================
# MilkNormalizationTransform tests
# ===========================================================================


class TestMilkNormalizationTransform:
    """Test MilkNormalizationTransform."""

    def test_transform_initialization(self):
        """Test transform initializes correctly."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        assert transform is not None

    def test_normalize_basic(self, sample_data):
        """Test basic normalization."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        result = transform(sample_data)

        # Original: [20.0, 25.0, 30.0], max=80.0
        # Expected: [0.25, 0.3125, 0.375]
        expected = np.array([20.0 / 80.0, 25.0 / 80.0, 30.0 / 80.0], dtype=np.float32)
        assert np.allclose(result["features"]["milk"], expected)

    def test_normalize_zero_milk(self):
        """Test normalization with zero milk."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        data: dict[str, object] = {
            "features": {
                "milk": np.array([0.0, 10.0, 80.0], dtype=np.float32),
            }
        }

        result = transform(data)

        expected = np.array([0.0, 0.125, 1.0], dtype=np.float32)
        assert np.allclose(result["features"]["milk"], expected)

    def test_normalize_exceeds_max(self):
        """Test normalization when milk exceeds max."""
        transform = MilkNormalizationTransform(max_milk=30.0)

        data: dict[str, object] = {
            "features": {
                "milk": np.array([20.0, 30.0, 40.0], dtype=np.float32),
            }
        }

        result = transform(data)

        # Values can exceed 1.0
        expected = np.array([20.0 / 30.0, 1.0, 40.0 / 30.0], dtype=np.float32)
        assert np.allclose(result["features"]["milk"], expected)

    def test_normalize_preserves_shape(self, sample_data):
        """Test normalization preserves shape."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        input_shape = sample_data["features"]["milk"].shape
        result = transform(sample_data)
        output_shape = result["features"]["milk"].shape

        assert input_shape == output_shape

    def test_normalize_preserves_dtype(self, sample_data):
        """Test normalization preserves dtype."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        result = transform(sample_data)

        assert result["features"]["milk"].dtype == np.float32

    def test_custom_max_milk(self):
        """Test custom max_milk value."""
        transform = MilkNormalizationTransform(max_milk=50.0)

        data: dict[str, object] = {
            "features": {
                "milk": np.array([25.0], dtype=np.float32),
            }
        }

        result = transform(data)

        expected = 25.0 / 50.0  # 0.5
        assert np.allclose(result["features"]["milk"], [expected])

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("milk_normalization", max_milk=80.0)
        assert isinstance(transform, MilkNormalizationTransform)


# ===========================================================================
# HerdStatsNormalizationTransform tests
# ===========================================================================


class TestHerdStatsNormalizationTransform:
    """Test HerdStatsNormalizationTransform."""

    def test_transform_initialization(self):
        """Test transform initializes correctly."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        assert transform is not None

    def test_zscore_normalization(self, sample_data):
        """Test z-score normalization."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        result = transform(sample_data)

        herd_stats = result["features"]["herd_stats"]
        # Z-score normalized should have mean ~0 and std ~1
        assert np.isclose(herd_stats.mean(), 0.0, atol=1e-6)
        assert np.isclose(herd_stats.std(), 1.0, atol=1e-6)

    def test_minmax_normalization(self, sample_data):
        """Test minmax normalization."""
        transform = HerdStatsNormalizationTransform(method="minmax")

        result = transform(sample_data)

        herd_stats = result["features"]["herd_stats"]
        # Minmax normalized should be between 0 and 1
        assert herd_stats.min() >= -1e-6
        assert herd_stats.max() <= 1.0 + 1e-6

    def test_zscore_with_different_values(self):
        """Test z-score normalization with different value ranges."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        data: dict[str, object] = {
            "features": {
                "herd_stats": np.array(
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                    dtype=np.float32,
                ),
            }
        }

        result = transform(data)

        herd_stats = result["features"]["herd_stats"]
        assert np.isclose(herd_stats.mean(), 0.0, atol=1e-6)
        assert np.isclose(herd_stats.std(), 1.0, atol=1e-6)

    def test_minmax_with_different_values(self):
        """Test minmax normalization with different value ranges."""
        transform = HerdStatsNormalizationTransform(method="minmax")

        data: dict[str, object] = {
            "features": {
                "herd_stats": np.array(
                    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
                    dtype=np.float32,
                ),
            }
        }

        result = transform(data)

        herd_stats = result["features"]["herd_stats"]
        assert np.isclose(herd_stats.min(), 0.0, atol=1e-6)
        assert np.isclose(herd_stats.max(), 1.0, atol=1e-6)

    def test_normalization_preserves_shape(self, sample_data):
        """Test normalization preserves shape."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        input_shape = sample_data["features"]["herd_stats"].shape
        result = transform(sample_data)
        output_shape = result["features"]["herd_stats"].shape

        assert input_shape == output_shape

    def test_normalization_preserves_dtype(self, sample_data):
        """Test normalization preserves dtype."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        result = transform(sample_data)

        assert result["features"]["herd_stats"].dtype == np.float32

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("herd_stats_normalization", method="zscore")
        assert isinstance(transform, HerdStatsNormalizationTransform)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestLactationTransformsIntegration:
    """Test lactation transforms working together."""

    def test_all_transforms_pipeline(self, sample_data, event_to_idx_path):
        """Test applying all transforms in sequence."""
        # Tokenize events (from pkl)
        tokenize = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)
        data = tokenize(sample_data)

        # Normalize milk
        milk_norm = MilkNormalizationTransform(max_milk=80.0)
        data = milk_norm(data)

        # Normalize herd stats
        herd_norm = HerdStatsNormalizationTransform(method="zscore")
        result = herd_norm(data)

        # Check final output
        assert "features" in result
        assert "milk" in result["features"]
        assert "events" in result["features"]
        assert "herd_stats" in result["features"]

    def test_transform_order_independence(self, sample_data):
        """Test that normalization order doesn't matter."""
        # Path 1: milk then herd_stats
        data1 = sample_data.copy()
        milk_norm = MilkNormalizationTransform(max_milk=80.0)
        data1 = milk_norm(data1)
        herd_norm1 = HerdStatsNormalizationTransform(method="zscore")
        result1 = herd_norm1(data1)

        # Path 2: herd_stats then milk
        data2 = sample_data.copy()
        herd_norm2 = HerdStatsNormalizationTransform(method="zscore")
        data2 = herd_norm2(data2)
        milk_norm2 = MilkNormalizationTransform(max_milk=80.0)
        result2 = milk_norm2(data2)

        # Results should be the same (normalization is independent)
        assert np.allclose(result1["features"]["milk"], result2["features"]["milk"])
        assert np.allclose(result1["features"]["herd_stats"], result2["features"]["herd_stats"])


class TestLactationTransformsEdgeCases:
    """Test edge cases for lactation transforms."""

    def test_empty_events_array(self, event_to_idx_path):
        """Test tokenization with empty events."""
        transform = EventTokenizationTransform(event_to_idx_path=event_to_idx_path)

        data: dict[str, object] = {
            "features": {
                "events": [],
            },
        }

        result = transform(data)
        assert len(result["features"]["events"]) == 0

    def test_very_small_milk_values(self):
        """Test normalization with very small milk values."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        data: dict[str, object] = {
            "features": {
                "milk": np.array([0.001, 0.01, 0.1], dtype=np.float32),
            }
        }

        result = transform(data)
        # Should still normalize correctly
        assert result["features"]["milk"][0] < result["features"]["milk"][1]

    def test_constant_herd_stats(self):
        """Test herd_stats normalization with constant values."""
        transform = HerdStatsNormalizationTransform(method="zscore")

        data: dict[str, object] = {
            "features": {
                "herd_stats": np.array(
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32
                ),
            }
        }

        result = transform(data)
        # Constant values should result in 0 or NaN after zscore
        herd_stats = result["features"]["herd_stats"]
        assert np.all((np.isnan(herd_stats)) | (herd_stats == 0.0))
