"""Serving examples and optional MLflow schema inference."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.models import ModelSignature

    from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset
    from bovi_core.ml.predictors import PredictorProtocol

logger = logging.getLogger(__name__)


def get_serving_input_example(
    dataset: Dataset,
    n_samples: int = 5,
    *,
    batch: bool = True,
    input_fields: Sequence[str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Select serving inputs without treating training targets as model inputs.

    By default, omit standard labels, metadata, and sample indices. Supply
    explicit top-level input fields for a model with a different serving schema.
    Dataset samples themselves remain unchanged.
    """
    example = dataset.get_input_example(n_samples=n_samples, batch=batch)
    excluded = {"label", "labels", "metadata", "index"}

    def select(sample: dict[str, Any]) -> dict[str, Any]:
        fields = (
            input_fields
            if input_fields is not None
            else [key for key in sample if key not in excluded]
        )
        if not fields:
            raise ValueError("No serving input fields selected")
        return {key: sample[key] for key in fields}

    return select(example) if isinstance(example, dict) else [select(item) for item in example]


def infer_dataset_signature(
    dataset: Dataset,
    predictor: PredictorProtocol | None = None,
    n_samples: int = 5,
    predict_kwargs: dict[str, Any] | None = None,
    *,
    batch: bool = True,
    input_fields: Sequence[str] | None = None,
) -> ModelSignature:
    """Infer a serving signature, falling back to input-only if prediction fails."""
    try:
        from mlflow.models import infer_signature
    except ImportError as exc:
        raise ImportError("mlflow is required for signature generation") from exc

    example = get_serving_input_example(
        dataset, n_samples=n_samples, batch=batch, input_fields=input_fields
    )
    predictions = None
    if predictor is not None:
        from bovi_core.ml.utils.signature_utils import output_to_serializable

        try:
            result = predictor.predict(example, return_format="base", **(predict_kwargs or {}))
            predictions = output_to_serializable(result)
        except Exception as exc:
            logger.warning("Signature prediction failed; using input-only schema: %s", exc)
    return infer_signature(example, predictions)
