"""MLflow signature generation utilities.

This module provides helper functions to convert prediction outputs to
MLflow-compatible serializable dictionaries for signature inference.
"""

from typing import Any, Dict

import numpy as np


def output_to_serializable(output: Any) -> Dict[str, Any]:
    """
    Convert any prediction output to MLflow-serializable dict.

    This is the single point of conversion for MLflow signature generation.
    Handles all three levels of prediction outputs:
    - Level 1 (raw): Framework-native outputs (np.ndarray, lists, etc.)
    - Level 2 (base dict): Already serializable, returns as-is
    - Level 3 (result object): Calls .to_serializable() method

    Args:
        output: Prediction output in any format

    Returns:
        Serializable dict suitable for MLflow signature inference

    Raises:
        TypeError: If output type cannot be serialized

    Example:
        >>> # Level 3: Rich result object
        >>> yolo_result = YoloPredictionResult.from_raw(raw_output)
        >>> serialized = output_to_serializable(yolo_result)
        >>> print(serialized)
        {"boxes_xyxy": [...], "scores": [...], "metadata": {...}}

        >>> # Level 2: Already a dict
        >>> base_dict = yolo_result.to_serializable()
        >>> serialized = output_to_serializable(base_dict)
        >>> # Returns same dict

        >>> # Level 1: Raw numpy array
        >>> raw_array = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> serialized = output_to_serializable(raw_array)
        >>> print(serialized)
        {"predictions": [[0.1, 0.9], [0.8, 0.2]], "metadata": {...}}
    """
    # Import here to avoid circular imports
    from bovi_core.ml.predictors.results import BasePredictionResult

    # Level 3: Rich result object (has to_serializable method)
    if isinstance(output, BasePredictionResult):
        return output.to_serializable()

    # Level 2: Already a dict (assume it's serializable)
    if isinstance(output, dict):
        # If it looks like a base dict (has metadata), return as-is
        if "metadata" in output:
            return output
        # Otherwise, wrap it
        return {
            **output,
            "metadata": {"model_type": "unknown", "format": "dict"},
        }

    # Level 1: Raw numpy array
    if isinstance(output, np.ndarray):
        return {
            "predictions": output.tolist(),
            "num_predictions": len(output) if output.ndim > 0 else 1,
            "metadata": {
                "model_type": "unknown",
                "format": "array",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
            },
        }

    # Level 1: Raw list
    if isinstance(output, (list, tuple)):
        return {
            "predictions": list(output),
            "num_predictions": len(output),
            "metadata": {"model_type": "unknown", "format": "list"},
        }

    # Level 1: Scalar value
    if isinstance(output, (int, float, bool, str)):
        return {
            "prediction": output,
            "num_predictions": 1,
            "metadata": {"model_type": "unknown", "format": "scalar"},
        }

    # Unknown type - try to provide helpful error message
    type_name = type(output).__name__
    module_name = type(output).__module__
    raise TypeError(
        f"Cannot serialize output of type {module_name}.{type_name}. "
        f"Expected one of:\n"
        f"  - BasePredictionResult subclass (with .to_serializable() method)\n"
        f"  - dict (Level 2 base format)\n"
        f"  - np.ndarray (raw predictions)\n"
        f"  - list/tuple (raw predictions)\n"
        f"  - scalar (int, float, bool, str)\n"
        f"\n"
        f"If this is a custom prediction result class, ensure it inherits from "
        f"BasePredictionResult and implements to_serializable()."
    )
