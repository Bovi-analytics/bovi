"""
Abstract Dataset interface.

Datasets are "dumb" - they return raw NumPy arrays/dicts.
Record transforms may wrap the source; framework conversion belongs to loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mlflow.models import ModelSignature

    from bovi_core.config import Config
    from bovi_core.ml.predictors import PredictorProtocol

    from ..sources.base_source import DataSource


class Dataset(ABC):
    """
    Abstract dataset combining DataSource.

    Defines WHAT to return for each item.

    Datasets define framework-neutral samples. Record transforms run on the
    source before feature selection; framework conversion belongs to loaders.
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
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get item by index.

        Returns:
            Dict with keys:
            - "features" (or a domain field such as "image"): Model inputs
            - "labels" (or domain-specific target fields): Targets, if available
            - "metadata": Additional info
        """
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        """Dataset-level metadata."""
        return {
            "length": len(self),
            "source_type": self.source.__class__.__name__,
            "config": self.config is not None,
        }

    # ========================================
    # Sample and schema inspection
    # ========================================

    def get_input_example(
        self,
        n_samples: int = 1,
        batch: bool = True,
        indices: list[int] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return dataset samples, batched with the same NumPy contract as loaders.

        Includes labels and metadata. Publishing code should select serving
        fields explicitly with get_serving_input_example.

        Args:
            n_samples: Positive maximum number of samples.
            batch: Collate samples when True; otherwise return one sample or a list.
            indices: Optional sample indices, in the requested order.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if not len(self):
            raise ValueError("Dataset is empty, cannot generate input example")
        n_samples = min(n_samples, len(self))
        selected = list(range(n_samples)) if indices is None else indices[:n_samples]
        if not selected:
            raise ValueError("indices must not be empty")
        if any(i < -len(self) or i >= len(self) for i in selected):
            raise ValueError(f"Index out of range for dataset of length {len(self)}")
        samples = [self[i] for i in selected]
        if not batch:
            return samples[0] if len(samples) == 1 else samples
        return self._batch_samples(samples)

    def _batch_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Use loader collation: nested arrays, with metadata kept per sample."""
        from bovi_core.ml.dataloaders.batching import collate_numpy_samples

        if not samples:
            return {}
        batched = collate_numpy_samples(samples)
        if not isinstance(batched, dict):
            raise ValueError("Dataset samples must have matching mapping keys")
        return batched

    def get_mlflow_signature(
        self,
        predictor: PredictorProtocol | None = None,
        n_samples: int = 5,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> ModelSignature:
        """Delegate serving-schema inference to the optional publishing layer."""
        from bovi_core.ml.publishing.signatures import infer_dataset_signature

        return infer_dataset_signature(
            self, predictor=predictor, n_samples=n_samples, predict_kwargs=predict_kwargs
        )

    def get_signature_info(self) -> dict[str, Any]:
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
