"""Tests for MLflow signature utilities."""

import numpy as np
import pytest

from bovi_core.ml.predictors.results import (
    GenericPredictionResult,
)
from bovi_core.ml.utils.signature_utils import output_to_serializable


class TestOutputToSerializable:
    """Test suite for output_to_serializable function."""

    def test_with_generic_result(self):
        """Test with GenericPredictionResult (Level 3)."""
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = GenericPredictionResult(predictions=predictions)

        serialized = output_to_serializable(result)

        # Should call to_serializable() method
        assert "predictions" in serialized
        assert "metadata" in serialized
        assert serialized["metadata"]["model_type"] == "generic"

    def test_with_dict_already_serializable(self):
        """Test with dict that's already in base format (Level 2)."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
            "scores": [0.95],
            "metadata": {"model_type": "yolo"}
        }

        serialized = output_to_serializable(base_dict)

        # Should return as-is since it has metadata
        assert serialized == base_dict

    def test_with_dict_no_metadata(self):
        """Test with dict without metadata."""
        output_dict = {
            "predictions": [[0.1, 0.9]],
            "labels": [1]
        }

        serialized = output_to_serializable(output_dict)

        # Should wrap it with metadata
        assert "predictions" in serialized
        assert "labels" in serialized
        assert "metadata" in serialized
        assert serialized["metadata"]["format"] == "dict"

    def test_with_numpy_array(self):
        """Test with raw numpy array (Level 1)."""
        raw_array = np.array([[0.1, 0.9], [0.8, 0.2]])

        serialized = output_to_serializable(raw_array)

        # Should convert to dict with predictions key
        assert "predictions" in serialized
        assert serialized["predictions"] == [[0.1, 0.9], [0.8, 0.2]]
        assert "num_predictions" in serialized
        assert serialized["num_predictions"] == 2
        assert "metadata" in serialized
        assert serialized["metadata"]["format"] == "array"
        assert serialized["metadata"]["shape"] == [2, 2]

    def test_with_list(self):
        """Test with raw list (Level 1)."""
        raw_list = [0.1, 0.2, 0.3]

        serialized = output_to_serializable(raw_list)

        # Should convert to dict with predictions key
        assert "predictions" in serialized
        assert serialized["predictions"] == [0.1, 0.2, 0.3]
        assert "num_predictions" in serialized
        assert serialized["num_predictions"] == 3
        assert "metadata" in serialized
        assert serialized["metadata"]["format"] == "list"

    def test_with_tuple(self):
        """Test with tuple."""
        raw_tuple = (0.1, 0.2, 0.3)

        serialized = output_to_serializable(raw_tuple)

        # Should convert tuple to list
        assert "predictions" in serialized
        assert serialized["predictions"] == [0.1, 0.2, 0.3]
        assert serialized["num_predictions"] == 3

    def test_with_scalar_int(self):
        """Test with scalar int value."""
        scalar = 42

        serialized = output_to_serializable(scalar)

        assert "prediction" in serialized
        assert serialized["prediction"] == 42
        assert serialized["num_predictions"] == 1
        assert serialized["metadata"]["format"] == "scalar"

    def test_with_scalar_float(self):
        """Test with scalar float value."""
        scalar = 3.14

        serialized = output_to_serializable(scalar)

        assert "prediction" in serialized
        assert serialized["prediction"] == 3.14
        assert serialized["num_predictions"] == 1

    def test_with_scalar_bool(self):
        """Test with scalar boolean value."""
        scalar = True

        serialized = output_to_serializable(scalar)

        assert "prediction" in serialized
        assert serialized["prediction"] is True
        assert serialized["num_predictions"] == 1

    def test_with_scalar_string(self):
        """Test with string value."""
        scalar = "prediction_class_A"

        serialized = output_to_serializable(scalar)

        assert "prediction" in serialized
        assert serialized["prediction"] == "prediction_class_A"
        assert serialized["num_predictions"] == 1

    def test_with_invalid_type(self):
        """Test error on unsupported type."""
        class CustomClass:
            pass

        custom_obj = CustomClass()

        with pytest.raises(TypeError) as exc_info:
            output_to_serializable(custom_obj)

        # Check error message is helpful
        error_msg = str(exc_info.value)
        assert "Cannot serialize output of type" in error_msg
        assert "CustomClass" in error_msg
        assert "BasePredictionResult" in error_msg

    def test_no_numpy_arrays_in_output(self):
        """Test that output contains no numpy arrays."""
        raw_array = np.array([[0.1, 0.9]])

        serialized = output_to_serializable(raw_array)

        # Check predictions are converted to lists
        assert isinstance(serialized["predictions"], list)
        assert isinstance(serialized["predictions"][0], list)
        assert not isinstance(serialized["predictions"], np.ndarray)

    def test_metadata_always_present(self):
        """Test that all outputs have metadata."""
        test_cases = [
            np.array([1, 2, 3]),
            [1, 2, 3],
            {"key": "value"},
            42,
            GenericPredictionResult(predictions=np.array([1, 2, 3])),
        ]

        for test_input in test_cases:
            serialized = output_to_serializable(test_input)
            assert "metadata" in serialized, f"metadata missing for {type(test_input)}"

    def test_preserves_metadata_from_result_objects(self):
        """Test that metadata from result objects is preserved."""
        result = GenericPredictionResult(
            predictions=np.array([[0.1, 0.9]]),
            metadata={"custom_key": "custom_value"},
        )

        serialized = output_to_serializable(result)

        assert "custom_key" in serialized["metadata"]
        assert serialized["metadata"]["custom_key"] == "custom_value"
