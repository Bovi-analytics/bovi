"""Tests for time-series transforms."""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.transforms.timeseries import (
    ImputationTransform,
    SequenceNormalizationTransform,
    SequencePaddingTransform,
    WindowingTransform,
)


class TestImputationTransform:
    """Test ImputationTransform."""

    @pytest.fixture
    def sequence_with_nan(self):
        """Create sequence with NaN values."""
        data = {
            "features": {"time_series": np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)}
        }
        return data

    @pytest.fixture
    def sequence_with_nans(self):
        """Create sequence with multiple NaNs."""
        data = {
            "features": {
                "time_series": np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan], dtype=np.float32)
            }
        }
        return data

    def test_forward_fill_imputation(self, sequence_with_nan):
        """Test forward fill imputation."""
        transform = ImputationTransform(method="forward_fill")
        result = transform(sequence_with_nan)

        expected = np.array([1.0, 1.0, 3.0, 3.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_backward_fill_imputation(self, sequence_with_nan):
        """Test backward fill imputation."""
        transform = ImputationTransform(method="backward_fill")
        result = transform(sequence_with_nan)

        expected = np.array([1.0, 3.0, 3.0, 5.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_linear_imputation(self, sequence_with_nan):
        """Test linear interpolation imputation."""
        transform = ImputationTransform(method="linear")
        result = transform(sequence_with_nan)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_zero_imputation(self, sequence_with_nan):
        """Test zero fill imputation."""
        transform = ImputationTransform(method="zero")
        result = transform(sequence_with_nan)

        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_mean_imputation(self):
        """Test mean fill imputation."""
        data = {"features": {"time_series": np.array([1.0, np.nan, 5.0], dtype=np.float32)}}
        transform = ImputationTransform(method="mean")
        result = transform(data)

        # Mean of non-nan values is 3.0
        expected = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_no_nans_forward_fill(self):
        """Test forward fill on sequence with no NaNs."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}
        transform = ImputationTransform(method="forward_fill")
        result = transform(data)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_all_nans_forward_fill(self):
        """Test forward fill when all values are NaN."""
        data = {"features": {"time_series": np.array([np.nan, np.nan, np.nan], dtype=np.float32)}}
        transform = ImputationTransform(method="forward_fill")
        result = transform(data)

        # Should remain NaN since there's nothing to forward fill from
        assert np.isnan(result["features"]["time_series"]).all()

    def test_multiple_consecutive_nans(self, sequence_with_nans):
        """Test handling of multiple consecutive NaNs."""
        transform = ImputationTransform(method="forward_fill")
        result = transform(sequence_with_nans)

        expected = np.array([1.0, 1.0, 1.0, 4.0, 5.0, 5.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("imputation", method="forward_fill")
        assert isinstance(transform, ImputationTransform)


class TestSequenceNormalizationTransform:
    """Test SequenceNormalizationTransform."""

    @pytest.fixture
    def sequence_data(self):
        """Create test sequence data."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}}
        return data

    def test_zscore_normalization(self, sequence_data):
        """Test z-score normalization."""
        transform = SequenceNormalizationTransform(method="zscore")
        result = transform(sequence_data)

        normalized = result["features"]["time_series"]
        # Z-score normalized should have mean ~0 and std ~1
        assert np.isclose(normalized.mean(), 0.0, atol=1e-6)
        assert np.isclose(normalized.std(), 1.0, atol=1e-6)

    def test_minmax_normalization(self, sequence_data):
        """Test minmax normalization."""
        transform = SequenceNormalizationTransform(method="minmax")
        result = transform(sequence_data)

        normalized = result["features"]["time_series"]
        # Minmax normalized should be between 0 and 1
        assert normalized.min() >= -1e-6  # Allow small floating point error
        assert normalized.max() <= 1.0 + 1e-6

    def test_maxabs_normalization(self):
        """Test maxabs normalization."""
        data = {"features": {"time_series": np.array([-5.0, -2.0, 3.0, 4.0], dtype=np.float32)}}
        transform = SequenceNormalizationTransform(method="maxabs")
        result = transform(data)

        normalized = result["features"]["time_series"]
        # Max absolute value should be 1
        assert np.isclose(np.abs(normalized).max(), 1.0)

    def test_scale_normalization(self):
        """Test scale normalization (divides by scalar)."""
        data = {
            "features": {"time_series": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)}
        }
        transform = SequenceNormalizationTransform(method="scale", scale=10.0)
        result = transform(data)

        normalized = result["features"]["time_series"]
        # Scale divides by the scale parameter
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert np.allclose(normalized, expected)

    def test_normalization_with_constant_sequence(self):
        """Test normalization with constant sequence."""
        data = {"features": {"time_series": np.array([3.0, 3.0, 3.0], dtype=np.float32)}}
        transform = SequenceNormalizationTransform(method="zscore")
        result = transform(data)

        normalized = result["features"]["time_series"]
        # Constant values after zscore normalization should be 0 or NaN
        # (handling edge case of std=0)
        assert np.all((np.isnan(normalized)) | (normalized == 0.0))

    def test_normalization_preserves_shape(self, sequence_data):
        """Test normalization preserves shape."""
        transform = SequenceNormalizationTransform(method="zscore")
        result = transform(sequence_data)

        assert (
            result["features"]["time_series"].shape
            == sequence_data["features"]["time_series"].shape
        )

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("sequence_normalization", method="zscore")
        assert isinstance(transform, SequenceNormalizationTransform)


class TestSequencePaddingTransform:
    """Test SequencePaddingTransform."""

    def test_pad_short_sequence(self):
        """Test padding a short sequence."""
        data = {"features": {"time_series": np.array([1.0, 2.0], dtype=np.float32)}}
        transform = SequencePaddingTransform(
            max_length=5, field="time_series", mode="post", pad_value=0.0
        )
        result = transform(data)

        expected = np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_truncate_long_sequence(self):
        """Test truncating a long sequence."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}}
        transform = SequencePaddingTransform(
            max_length=3, field="time_series", mode="post", pad_value=0.0
        )
        result = transform(data)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_pad_pre_mode(self):
        """Test padding with pre mode (pad at beginning)."""
        data = {"features": {"time_series": np.array([1.0, 2.0], dtype=np.float32)}}
        transform = SequencePaddingTransform(
            max_length=5, field="time_series", mode="pre", pad_value=0.0
        )
        result = transform(data)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_exact_length_no_change(self):
        """Test sequence with exact length doesn't change."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}
        transform = SequencePaddingTransform(
            max_length=3, field="time_series", mode="post", pad_value=0.0
        )
        result = transform(data)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_custom_pad_value(self):
        """Test padding with custom pad value."""
        data = {"features": {"time_series": np.array([1.0, 2.0], dtype=np.float32)}}
        transform = SequencePaddingTransform(
            max_length=4, field="time_series", mode="post", pad_value=-1.0
        )
        result = transform(data)

        expected = np.array([1.0, 2.0, -1.0, -1.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create(
            "sequence_padding", max_length=10, field="time_series", mode="post"
        )
        assert isinstance(transform, SequencePaddingTransform)


class TestWindowingTransform:
    """Test WindowingTransform.

    Note: The current implementation truncates to the first window_size elements,
    rather than creating multiple sliding windows. This is useful for data
    preparation where you want to limit sequence length.
    """

    def test_truncate_to_window_size(self):
        """Test that windowing truncates to window_size."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}}
        transform = WindowingTransform(window_size=3, stride=1)
        result = transform(data)

        # Implementation truncates to first window_size elements
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_windowing_with_field_param(self):
        """Test windowing with specific field."""
        data = {
            "features": {
                "time_series": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "other": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            }
        }
        transform = WindowingTransform(window_size=2, stride=1, field="time_series")
        result = transform(data)

        # Only time_series should be truncated
        assert len(result["features"]["time_series"]) == 2
        assert len(result["features"]["other"]) == 4

    def test_window_larger_than_sequence(self):
        """Test when window size is larger than sequence - no change."""
        data = {"features": {"time_series": np.array([1.0, 2.0], dtype=np.float32)}}
        transform = WindowingTransform(window_size=5, stride=1)
        result = transform(data)

        # Sequence is shorter than window_size, should remain unchanged
        expected = np.array([1.0, 2.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_exact_window_size(self):
        """Test when sequence is exactly window_size."""
        data = {"features": {"time_series": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}
        transform = WindowingTransform(window_size=3, stride=1)
        result = transform(data)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_transform_registry(self):
        """Test transform is properly registered."""
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        transform = TransformRegistry.create("windowing", window_size=5, stride=1)
        assert isinstance(transform, WindowingTransform)


class TestTimeSeriesTransformIntegration:
    """Test time-series transforms working together."""

    def test_imputation_then_normalization(self):
        """Test pipeline of imputation then normalization."""
        data = {"features": {"time_series": np.array([1.0, np.nan, 3.0], dtype=np.float32)}}

        # Impute
        impute = ImputationTransform(method="linear")
        data = impute(data)

        # Normalize
        normalize = SequenceNormalizationTransform(method="zscore")
        result = normalize(data)

        # After imputation: [1.0, 2.0, 3.0]
        # After normalization: mean ~0, std ~1
        normalized = result["features"]["time_series"]
        assert np.isclose(normalized.mean(), 0.0, atol=1e-6)
        assert np.isclose(normalized.std(), 1.0, atol=1e-6)

    def test_padding_then_windowing(self):
        """Test pipeline of padding then windowing."""
        data = {"features": {"time_series": np.array([1.0, 2.0], dtype=np.float32)}}

        # Pad to length 5
        pad = SequencePaddingTransform(
            max_length=5, field="time_series", mode="post", pad_value=0.0
        )
        data = pad(data)

        # After padding: [1.0, 2.0, 0.0, 0.0, 0.0]
        assert len(data["features"]["time_series"]) == 5

        # Truncate to window size
        window = WindowingTransform(window_size=3, stride=1)
        result = window(data)

        # After windowing: truncated to first 3 elements
        expected = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        assert np.allclose(result["features"]["time_series"], expected)

    def test_full_pipeline(self):
        """Test full pipeline: impute -> normalize -> pad -> window."""
        data = {
            "features": {"time_series": np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)}
        }

        # Impute
        impute = ImputationTransform(method="forward_fill")
        data = impute(data)

        # Normalize
        normalize = SequenceNormalizationTransform(method="zscore")
        data = normalize(data)

        # Pad
        pad = SequencePaddingTransform(
            max_length=10, field="time_series", mode="post", pad_value=0.0
        )
        data = pad(data)

        # Window (truncate to size 3)
        window = WindowingTransform(window_size=3, stride=1)
        result = window(data)

        # Should be truncated to window_size
        assert len(result["features"]["time_series"]) == 3
