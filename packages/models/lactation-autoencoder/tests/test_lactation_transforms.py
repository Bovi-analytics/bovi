"""Tests for lactation-specific transforms."""

import numpy as np
import pytest

from lactation_autoencoder.dataloaders.transforms.lactation_transforms import (
    EventTokenizationTransform,
    HerdStatsNormalizationTransform,
    MilkNormalizationTransform,
)


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
def sample_data(event_to_idx_mapping):
    """Create sample data for transforms."""
    return {
        "features": {
            "milk": np.array([20.0, 25.0, 30.0], dtype=np.float32),
            "events": ["milking", "vaccination", "milking"],
            "parity": np.array([1], dtype=np.float32),
            "herd_stats": np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32
            ),
        },
        "event_to_idx": event_to_idx_mapping,
    }


class TestEventTokenizationTransform:
    """Test EventTokenizationTransform."""

    def test_transform_initialization(self):
        """Test transform initializes correctly."""
        transform = EventTokenizationTransform()

        assert transform is not None
        assert hasattr(transform, "unknown_event")

    def test_tokenize_single_event(self, event_to_idx_mapping):
        """Test tokenizing single event."""
        transform = EventTokenizationTransform()

        # Modify sample to have single event
        sample = {
            "features": {
                "events": ["milking"],
                "milk": np.array([20.0], dtype=np.float32),
            },
            "event_to_idx": event_to_idx_mapping,
        }

        result = transform(sample)

        # Should be tokenized to 0
        assert result["features"]["events"][0] == 0

    def test_tokenize_multiple_events(self, sample_data):
        """Test tokenizing multiple events."""
        transform = EventTokenizationTransform()

        result = transform(sample_data)

        events = result["features"]["events"]
        assert events[0] == 0  # milking
        assert events[1] == 1  # vaccination
        assert events[2] == 0  # milking

    def test_tokenization_preserves_shape(self, sample_data):
        """Test tokenization preserves shape."""
        transform = EventTokenizationTransform()

        input_events = sample_data["features"]["events"]
        result = transform(sample_data)
        output_events = result["features"]["events"]

        assert len(input_events) == len(output_events)

    def test_unknown_event_handling(self, event_to_idx_mapping):
        """Test handling of unknown events."""
        transform = EventTokenizationTransform()

        data = {
            "features": {
                "events": ["milking", "unknown_event", "vaccination"],
            },
            "event_to_idx": event_to_idx_mapping,
        }

        # Should handle gracefully (may use a default or raise)
        try:
            result = transform(data)
            # If it doesn't raise, check output
            assert len(result["features"]["events"]) == 3
        except KeyError:
            # Expected if unknown events raise KeyError
            pass

    def test_all_events_tokenized(self, event_to_idx_mapping):
        """Test all mapped events are tokenized correctly."""
        transform = EventTokenizationTransform()

        data = {
            "features": {
                "events": ["milking", "vaccination", "treatment", "heat", "calving"],
            },
            "event_to_idx": event_to_idx_mapping,
        }

        result = transform(data)
        events = result["features"]["events"]

        expected = [0, 1, 2, 3, 4]
        assert list(events) == expected

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("event_tokenization")
        assert isinstance(transform, EventTokenizationTransform)


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

        data = {
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

        data = {
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

        data = {
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

        data = {
            "features": {
                "herd_stats": np.array(
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=np.float32
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

        data = {
            "features": {
                "herd_stats": np.array(
                    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0], dtype=np.float32
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


class TestLactationTransformsIntegration:
    """Test lactation transforms working together."""

    def test_all_transforms_pipeline(self, sample_data):
        """Test applying all transforms in sequence."""
        # Tokenize events
        tokenize = EventTokenizationTransform()
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

    def test_empty_events_array(self, event_to_idx_mapping):
        """Test tokenization with empty events."""
        transform = EventTokenizationTransform()

        data = {
            "features": {
                "events": [],
            },
            "event_to_idx": event_to_idx_mapping,
        }

        result = transform(data)
        assert len(result["features"]["events"]) == 0

    def test_very_small_milk_values(self):
        """Test normalization with very small milk values."""
        transform = MilkNormalizationTransform(max_milk=80.0)

        data = {
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

        data = {
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
