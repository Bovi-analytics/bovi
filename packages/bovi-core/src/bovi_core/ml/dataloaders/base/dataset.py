"""
Abstract Dataset interface.

Datasets are "dumb" - they return raw NumPy arrays/dicts.
All transformations happen in DataLoaders via FrameworkAdapter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.models import Model
    from mlflow.models import ModelSignature

    from .data_source import DataSource

logger = logging.getLogger(__name__)


class Dataset(ABC):
    """
    Abstract dataset combining DataSource.

    Defines WHAT to return for each item.

    Note: Datasets do NOT handle transforms - they always return raw NumPy data.
    Transforms are applied in DataLoaders via FrameworkAdapter.
    """

    # Type annotations for instance attributes
    source: DataSource
    config: Config | None

    def __init__(
        self,
        source: DataSource,
        config: Config | None = None,
    ) -> None:
        self.source = source
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        """Number of items in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Get item by index.

        Returns:
            Dict with keys:
            - "data": Processed data (image, video, features)
            - "label": Label (if available)
            - "metadata": Additional info
        """
        pass

    @property
    def metadata(self) -> dict[str, object]:
        """Dataset-level metadata."""
        return {
            "length": len(self),
            "source_type": self.source.__class__.__name__,
            "config": self.config is not None,
        }

    # ========================================
    # Unity Catalog Support Methods
    # ========================================

    def get_input_example(
        self,
        n_samples: int = 1,
        batch: bool = True,
        indices: list[int] | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        """
        Generate input example for MLflow signature inference.

        This method creates sample data that represents the expected input
        format for models trained on this dataset. Used for Unity Catalog
        model registration.

        Args:
            n_samples: Number of samples to include in the example.
            batch: If True, batch samples together (recommended for most models).
            indices: Specific indices to use (default: first n_samples).

        Returns:
            Single sample dict (if n_samples=1 and batch=False) or
            batched dict with stacked arrays (if batch=True).

        Example:
            >>> # Single unbatched sample
            >>> dataset.get_input_example(n_samples=1, batch=False)
            {"image": np.ndarray(3, 224, 224), "label": 0}

            >>> # Multiple samples batched together
            >>> dataset.get_input_example(n_samples=5, batch=True)
            {"image": np.ndarray(5, 3, 224, 224), "label": np.array([0,1,0,2,1])}

            >>> # Specific samples
            >>> dataset.get_input_example(n_samples=3, indices=[10, 20, 30])
            {"image": np.ndarray(3, 3, 224, 224), ...}
        """

        # Validate n_samples
        n_samples = min(n_samples, len(self))
        if n_samples == 0:
            raise ValueError("Dataset is empty, cannot generate input example")

        # Determine indices to sample
        if indices is None:
            # Use first n_samples
            indices = list(range(n_samples))
        else:
            # Validate provided indices
            indices = indices[:n_samples]
            if any(i >= len(self) for i in indices):
                raise ValueError(f"Index out of range for dataset of length {len(self)}")

        # Get samples
        if n_samples == 1 and not batch:
            # Return single unbatched sample
            return self[indices[0]]

        # Get multiple samples
        samples = [self[i] for i in indices]

        if not batch:
            # Return list of samples
            return samples

        # Batch samples together
        return self._batch_samples(samples)

    def _batch_samples(self, samples: list[dict[str, object]]) -> dict[str, object]:
        """
        Batch multiple sample dicts into single batched dict.

        Stacks arrays along a new batch dimension (axis 0).
        Handles different data types appropriately:
        - numpy arrays: Stack with np.stack()
        - scalars (int, float): Convert to np.array()
        - strings: Keep as list
        - dicts: Keep as list (nested structure)
        - None: Keep as None

        Args:
            samples: List of sample dicts from __getitem__().

        Returns:
            Batched dict with arrays stacked along dimension 0.

        Example:
            >>> samples = [
            ...     {"image": np.zeros((3, 224, 224)), "label": 0},
            ...     {"image": np.ones((3, 224, 224)), "label": 1},
            ... ]
            >>> batched = self._batch_samples(samples)
            >>> batched["image"].shape
            (2, 3, 224, 224)
            >>> batched["label"]
            array([0, 1])
        """
        if not samples:
            return {}

        batched: dict[str, object] = {}
        keys = samples[0].keys()

        for key in keys:
            values = [s[key] for s in samples]

            # Get first non-None value to determine type
            first_value = next((v for v in values if v is not None), None)

            if first_value is None:
                # All values are None
                batched[key] = None
                continue

            # Handle different types
            if isinstance(first_value, np.ndarray):
                # Stack numpy arrays
                # Filter out None values
                valid_arrays = [v for v in values if v is not None]
                if len(valid_arrays) == len(values):
                    # All valid - stack normally
                    batched[key] = np.stack(values, axis=0)  # type: ignore[arg-type]
                else:
                    # Some None - keep as list
                    batched[key] = values

            elif isinstance(first_value, (int, float, bool, np.integer, np.floating)):
                # Convert scalars to numpy array
                batched[key] = np.array(values)

            elif isinstance(first_value, str):
                # Keep strings as list
                batched[key] = values

            elif isinstance(first_value, dict):
                # Nested dicts - keep as list
                batched[key] = values

            elif isinstance(first_value, list):
                # Lists - keep as list of lists
                batched[key] = values

            elif hasattr(first_value, "__array__"):
                # Array-like objects (torch.Tensor, tf.Tensor, etc.)
                try:
                    batched[key] = np.stack([np.array(v) for v in values], axis=0)
                except (ValueError, TypeError):
                    # Can't stack - keep as list
                    batched[key] = values

            else:
                # Unknown type - keep as list
                batched[key] = values

        return batched

    def get_mlflow_signature(
        self,
        model: Model[object] | None = None,
        n_samples: int = 5,
        predict_kwargs: dict[str, object] | None = None,
    ) -> ModelSignature:
        """
        Generate MLflow signature from dataset samples.

        Creates a ModelSignature that defines the input and output schema
        for MLflow model registration. The signature is inferred from actual
        data samples and optionally from model predictions.

        Args:
            model: Optional Model instance to infer output signature.
                   If None, only input signature is generated.
            n_samples: Number of samples to use for signature inference.
                       More samples = better type inference.
            predict_kwargs: Optional kwargs to pass to model.predict().

        Returns:
            mlflow.models.ModelSignature with input (and optionally output) schema.

        Example:
            >>> # Input-only signature (for data validation)
            >>> signature = dataset.get_mlflow_signature()
            >>> print(signature.inputs)
            # {'image': float32 (required), 'label': long (required)}

            >>> # Full signature with predictions
            >>> from bovi_core.ml import create_model
            >>> model = create_model(config, "yolo", "best")
            >>> signature = dataset.get_mlflow_signature(model=model, n_samples=10)
            >>> print(signature.inputs)
            >>> print(signature.outputs)

        Raises:
            ImportError: If mlflow is not installed.
            ValueError: If dataset is empty or model prediction fails.
        """
        try:
            from mlflow.models import infer_signature
        except ImportError:
            raise ImportError(
                "mlflow is required for signature generation. Install with: pip install mlflow"
            )

        # Get input example
        input_example = self.get_input_example(n_samples=n_samples, batch=True)

        if model is None:
            # Input-only signature
            logger.info("Generating input-only signature from dataset samples")
            return infer_signature(input_example, None)

        # Get predictions to infer output schema
        try:
            logger.info(f"Generating signature with model predictions using {n_samples} samples")
            predict_kwargs = predict_kwargs or {}

            # Request base format for MLflow signature (Level 2: portable dict)
            predict_fn = getattr(model, "predict", None)
            if predict_fn is None:
                raise ValueError("Model does not have a predict method")
            prediction_result = predict_fn(input_example, return_format="base", **predict_kwargs)

            # Convert to serializable format
            from bovi_core.ml.utils.signature_utils import output_to_serializable

            predictions = output_to_serializable(prediction_result)

            return infer_signature(input_example, predictions)

        except Exception as e:
            logger.warning(
                f"Failed to generate signature with model predictions: {e}. "
                f"Falling back to input-only signature."
            )
            return infer_signature(input_example, None)

    def get_signature_info(self) -> dict[str, object]:
        """
        Get information about the dataset schema for debugging.

        Returns:
            Dict with schema information including field names, types, and shapes.

        Example:
            >>> info = dataset.get_signature_info()
            >>> print(info)
            {
                'num_samples': 1000,
                'sample_fields': ['image', 'label'],
                'field_types': {'image': 'numpy.ndarray', 'label': 'int'},
                'field_shapes': {'image': (3, 224, 224), 'label': ()},
                'example_sample': {...}
            }
        """
        # Get a sample
        sample = self[0]

        field_types: dict[str, str] = {}
        field_shapes: dict[str, tuple[int, ...]] = {}

        for key, value in sample.items():
            field_types[key] = type(value).__name__

            if isinstance(value, np.ndarray):
                field_shapes[key] = value.shape
            elif hasattr(value, "shape"):
                field_shapes[key] = tuple(value.shape)  # type: ignore[arg-type]
            else:
                field_shapes[key] = ()

        return {
            "num_samples": len(self),
            "sample_fields": list(sample.keys()),
            "field_types": field_types,
            "field_shapes": field_shapes,
            "example_sample": sample,
        }
