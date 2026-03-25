"""Tests for GenericPredictionResult class."""

import numpy as np
import pytest

from bovi_core.ml.predictors.results import GenericPredictionResult


class TestGenericPredictionResult:
    """Test suite for GenericPredictionResult."""

    def test_simple_array_prediction(self):
        """Test with simple numpy array output."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = GenericPredictionResult(predictions=predictions)

        assert result.num_predictions == 2
        assert np.array_equal(result.predictions, predictions)

    def test_dict_prediction(self):
        """Test with dictionary of arrays."""
        predictions = {
            "class_probs": np.array([[0.1, 0.9]]),
            "embeddings": np.array([[0.5, 0.3, 0.2]]),
        }
        result = GenericPredictionResult(predictions=predictions)

        assert result.num_predictions == 1
        assert "class_probs" in result.predictions
        assert "embeddings" in result.predictions

    def test_from_raw_array(self):
        """Test from_raw class method with array."""
        raw_output = np.array([[0.5, 0.5], [0.7, 0.3]])
        result = GenericPredictionResult.from_raw(raw_output)

        assert result.num_predictions == 2
        assert np.array_equal(result.predictions, raw_output)

    def test_from_raw_dict(self):
        """Test from_raw class method with dict."""
        raw_output = {"predictions": np.array([1, 2, 3])}
        result = GenericPredictionResult.from_raw(raw_output)

        assert isinstance(result.predictions, dict)
        assert "predictions" in result.predictions

    def test_from_raw_with_metadata(self):
        """Test from_raw with additional metadata."""
        raw_output = np.array([1, 2, 3])
        result = GenericPredictionResult.from_raw(
            raw_output,
            metadata={"model_name": "my_model"}
        )

        assert result.metadata["model_name"] == "my_model"

    def test_from_raw_with_kwargs(self):
        """Test from_raw with kwargs that get added to metadata."""
        raw_output = np.array([1, 2, 3])
        result = GenericPredictionResult.from_raw(
            raw_output,
            threshold=0.5,
            model_version="v1.0"
        )

        assert result.metadata["threshold"] == 0.5
        assert result.metadata["model_version"] == "v1.0"

    def test_to_serializable_array(self):
        """Test serialization of array predictions."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = GenericPredictionResult(predictions=predictions)

        serialized = result.to_serializable()

        # Check structure
        assert "predictions" in serialized
        assert "num_predictions" in serialized
        assert "metadata" in serialized

        # Check values
        assert serialized["predictions"] == [[0.1, 0.9], [0.8, 0.2]]
        assert serialized["num_predictions"] == 2

        # Check metadata
        assert serialized["metadata"]["model_type"] == "generic"
        assert serialized["metadata"]["prediction_shape"] == [2, 2]
        assert "float" in serialized["metadata"]["prediction_dtype"]

    def test_to_serializable_dict(self):
        """Test serialization of dict predictions."""
        predictions = {
            "class_probs": np.array([[0.1, 0.9]]),
            "class_labels": np.array([1]),
        }
        result = GenericPredictionResult(predictions=predictions)

        serialized = result.to_serializable()

        # Should have keys from predictions dict
        assert "class_probs" in serialized
        assert "class_labels" in serialized
        assert "num_predictions" in serialized
        assert "metadata" in serialized

        # Check conversion to lists
        assert serialized["class_probs"] == [[0.1, 0.9]]
        assert serialized["class_labels"] == [1]

        # Check metadata
        assert serialized["metadata"]["model_type"] == "generic"
        assert serialized["metadata"]["output_keys"] == ["class_probs", "class_labels"]

    def test_num_predictions_1d_array(self):
        """Test num_predictions for 1D array."""
        predictions = np.array([0.1, 0.2, 0.3])
        result = GenericPredictionResult(predictions=predictions)

        assert result.num_predictions == 3

    def test_num_predictions_2d_array(self):
        """Test num_predictions for 2D array."""
        predictions = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = GenericPredictionResult(predictions=predictions)

        assert result.num_predictions == 2

    def test_num_predictions_scalar(self):
        """Test num_predictions for scalar (0D array)."""
        predictions = np.array(42)
        result = GenericPredictionResult(predictions=predictions)

        assert result.num_predictions == 1

    def test_num_predictions_dict(self):
        """Test num_predictions for dict with arrays."""
        predictions = {
            "output1": np.array([[1, 2, 3], [4, 5, 6]]),
            "output2": np.array([[7, 8], [9, 10]]),
        }
        result = GenericPredictionResult(predictions=predictions)

        # Should use first key's length
        assert result.num_predictions == 2

    def test_metadata_preservation(self):
        """Test that custom metadata is preserved."""
        predictions = np.array([1, 2, 3])
        custom_metadata = {"custom_key": "custom_value"}

        result = GenericPredictionResult(
            predictions=predictions,
            metadata=custom_metadata
        )

        serialized = result.to_serializable()
        assert "custom_key" in serialized["metadata"]
        assert serialized["metadata"]["custom_key"] == "custom_value"

    def test_serialization_no_numpy_arrays(self):
        """Test that serialized output contains no numpy arrays."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = GenericPredictionResult(predictions=predictions)

        serialized = result.to_serializable()

        # Check that predictions are converted to lists
        assert isinstance(serialized["predictions"], list)
        assert isinstance(serialized["predictions"][0], list)
        assert not isinstance(serialized["predictions"], np.ndarray)
