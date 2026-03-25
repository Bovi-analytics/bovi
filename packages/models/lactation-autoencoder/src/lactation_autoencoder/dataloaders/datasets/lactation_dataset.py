"""
LactationDataset: Lactation autoencoder dataset with standard feature/label structure.

Feature Structure (inputs to model):
- milk: Daily milk production [304 floats, normalized 0-1]
- events: Tokenized event indices [304 ints, event indices]
- parity: Lactation number [1 int]
- herd_stats: 10 herd statistics [10 floats, normalized]

Label Structure (target for model):
- milk: Same as feature milk (autoencoder reconstructs input)

Standard output structure:
{
    "features": {
        "milk": np.array([304], float32),
        "events": np.array([304], int32),
        "parity": np.array([1], float32),
        "herd_stats": np.array([10], float32)
    },
    "labels": np.array([304], float32)  # Same as features["milk"]
}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import torch
from bovi_core.ml.dataloaders.base import DataSource
from bovi_core.ml.dataloaders.datasets import FeatureVectorDataset
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import override

from lactation_autoencoder.types import LactationItem

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.models import Model
    from mlflow.models import ModelSignature


class LactationDataset(FeatureVectorDataset, TorchDataset[LactationItem]):
    """
    Lactation autoencoder dataset with hierarchical herd statistics.

    Extends FeatureVectorDataset to provide lactation-specific structure:
    - Features: milk, events, parity, herd_stats
    - Labels: milk (for autoencoder reconstruction)

    The hierarchical herd statistics provide context:
    - Level 1: Specific herd + parity combination
    - Level 2: Herd average (if parity missing)
    - Level 3: Parity average (if herd missing)
    - Level 4: Global average (if both missing)

    This ensures the model never fails due to missing metadata.

    Note: Following the "Datasets are Dumb" principle, transforms are NOT
    applied in the dataset. Use TransformedSource to wrap the source with
    transforms before passing to the dataset.

    Args:
        source: LactationJSONSource or TransformedSource instance
        config: Optional config
        max_days: Maximum sequence length (default: 304)

    Example:
        from lactation_autoencoder.dataloaders.sources import LactationPKLSource
        from lactation_autoencoder.dataloaders.datasets import LactationDataset
        from bovi_core.ml.dataloaders.sources import TransformedSource
        from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

        source = LactationPKLSource(
            json_root_dir="data/jsons/",
            herd_stats_dir="data/json/"
        )

        # Create transforms
        transforms = [
            TransformRegistry.create("imputation", method="forward_fill"),
            TransformRegistry.create("event_tokenization"),
            TransformRegistry.create("milk_normalization", max_milk=80.0),
        ]

        # Wrap source with transforms
        transformed_source = TransformedSource(source, transforms)
        dataset = LactationDataset(transformed_source, config=config)

        item = dataset[0]
        X = item["features"]  # Dict with milk, events, parity, herd_stats
        y = item["labels"]     # milk array (same as X["milk"])
        metadata = item["metadata"]  # Optional metadata
    """

    max_days: int

    def __init__(
        self,
        source: DataSource[dict[str, object]],
        config: Config | None = None,
        max_days: int = 304,
    ):
        """
        Initialize lactation dataset.

        Args:
            source: LactationJSONSource or TransformedSource instance
            config: Optional config instance
            max_days: Maximum sequence length (default: 304 days)
        """
        super().__init__(source, config)
        self.max_days = max_days

    @override
    def _get_features(self, raw_data: dict[str, object]) -> dict[str, object]:
        """
        Extract features from processed data.

        Features:
        - milk: Daily production (normalized)
        - events: Tokenized event indices
        - parity: Lactation number
        - herd_stats: 10 contextual statistics

        All sequences padded/truncated to max_days.

        Args:
            raw_data: Raw data dict from source (after transforms)

        Returns:
            Dict of features with keys: milk, events, parity, herd_stats
        """
        # Get sequences
        milk = raw_data.get("milk")
        # Use if/else to avoid numpy array truth value error
        events = raw_data.get("events_encoded")
        if events is None:
            events = raw_data.get("events")
        parity = raw_data.get("parity")
        herd_stats = raw_data.get("herd_stats")

        # Validate milk data
        if milk is None:
            raise ValueError("'milk' field not found in raw data")
        if events is None:
            raise ValueError("'events' or 'events_encoded' field not found in raw data")
        if parity is None:
            raise ValueError("'parity' field not found in raw data")
        if herd_stats is None:
            raise ValueError("'herd_stats' field not found in raw data")

        # Convert to numpy arrays
        milk = np.asarray(milk, dtype=np.float32)

        # Tokenize events if they're strings
        events_input = cast(list[str] | list[int] | npt.NDArray[np.integer[npt.NBitBase]], events)
        events_tokenized = self._tokenize_events(events_input, raw_data)
        events_arr = np.asarray(events_tokenized, dtype=np.int32)

        parity = float(cast(float | int | str, parity))
        herd_stats = np.asarray(herd_stats, dtype=np.float32)

        # Pad/truncate sequences
        milk = self._pad_sequence(milk, self.max_days)
        events_padded = self._pad_sequence(events_arr, self.max_days)

        return {
            "milk": milk.astype(np.float32),
            "events": events_padded.astype(np.int32),
            "parity": np.array([parity], dtype=np.float32),
            "herd_stats": herd_stats.astype(np.float32),
        }

    @override
    def _get_labels(self, raw_data: dict[str, object]) -> npt.NDArray[np.float32]:
        """
        Extract labels (target for model).

        For autoencoder: target = input (reconstruct milk)

        Args:
            raw_data: Raw data dict from source (after transforms)

        Returns:
            Milk array (target for reconstruction)
        """
        milk = raw_data.get("milk")

        if milk is None:
            raise ValueError("'milk' field not found in raw data")

        milk = np.asarray(milk, dtype=np.float32)
        milk = self._pad_sequence(milk, self.max_days)

        return milk.astype(np.float32)

    @override
    def _get_metadata(self, raw_data: dict[str, object], index: int) -> dict[str, object] | None:
        """
        Extract metadata about the lactation record.

        Args:
            raw_data: Raw data dict
            index: Dataset index

        Returns:
            Dict with animal_id, herd_id, parity, etc.
        """
        return {
            "index": index,
            "animal_id": raw_data.get("animal_id"),
            "herd_id": raw_data.get("herd_id"),
            "parity": raw_data.get("parity"),
        }

    def _tokenize_events(
        self,
        events: list[str] | list[int] | npt.NDArray[np.integer[npt.NBitBase]],
        raw_data: dict[str, object],
    ) -> list[int]:
        """
        Convert event strings to token indices.

        If events are already integers, return as-is.
        If events are strings, use event_to_idx from raw_data to tokenize.

        Args:
            events: List of event strings or indices
            raw_data: Raw data containing event_to_idx mapping

        Returns:
            List of integer event indices
        """
        # If already integers, return as list[int]
        events_arr = np.asarray(events)
        if np.issubdtype(events_arr.dtype, np.integer):
            return [int(x) for x in events_arr]

        # Get event mapping
        event_to_idx_raw = raw_data.get("event_to_idx", {})
        event_to_idx = cast(dict[str, int], event_to_idx_raw) if event_to_idx_raw else {}
        if not event_to_idx:
            # No event mapping available, return all zeros
            return [0] * len(events)

        # Tokenize event strings
        tokenized: list[int] = []
        for event in events:
            if isinstance(event, str):
                # Try to match event, fallback to "unknown" or 0
                token = event_to_idx.get(event.lower(), event_to_idx.get("unknown", 0))
            else:
                token = int(event)
            tokenized.append(int(token))

        return tokenized

    def _pad_sequence(
        self, seq: npt.NDArray[np.generic], max_len: int
    ) -> npt.NDArray[np.generic]:
        """
        Pad or truncate sequence to max_len.

        Args:
            seq: Input sequence
            max_len: Target length

        Returns:
            Padded/truncated sequence of length max_len
        """
        seq = np.asarray(seq)

        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            pad_width = (0, max_len - len(seq))
            return np.pad(seq, pad_width, constant_values=0)

    @override
    def get_input_example(
        self,
        n_samples: int = 1,
        batch: bool = True,
        indices: list[int] | None = None,
    ) -> dict[str, object] | list[dict[str, object]]:
        """
        Generate input example for MLflow signature inference.

        This overrides the base class to return features in the format expected
        by the LactationPredictor (flat dict with milk, events, parity, herd_stats).

        Args:
            n_samples: Number of samples to include
            batch: If True, batch samples together (stack arrays)
            indices: Specific indices to use (default: first n_samples)

        Returns:
            Dict with keys: milk, events, parity, herd_stats
            Each value is a batched numpy array if batch=True

        Example:
            >>> # Single sample
            >>> example = dataset.get_input_example(n_samples=1, batch=True)
            >>> example.keys()
            dict_keys(['milk', 'events', 'parity', 'herd_stats'])
            >>> example['milk'].shape
            (1, 304)
        """
        # Get samples using parent method
        parent_example = super().get_input_example(
            n_samples=n_samples, batch=batch, indices=indices
        )

        # Parent returns: {'features': {...}, 'labels': ..., 'metadata': ...}
        # We need to extract just the features dict for the predictor

        if batch:
            # Parent batched the features dict into a list
            # We need to reconstruct batched arrays
            features = parent_example.get('features') if isinstance(parent_example, dict) else None
            if isinstance(parent_example, dict) and isinstance(features, list):
                # Batch the feature dicts ourselves
                feature_dicts = cast(list[dict[str, object]], parent_example['features'])

                # Stack each feature across samples
                batched_features: dict[str, object] = {}
                for key in feature_dicts[0].keys():
                    values = [f[key] for f in feature_dicts]
                    arrays = cast(list[npt.NDArray[np.generic]], values)
                    batched_features[key] = np.stack(arrays, axis=0)

                return batched_features
            elif isinstance(parent_example, dict) and 'features' in parent_example:
                # Already a dict (single sample case)
                return cast(dict[str, object], parent_example['features'])
            else:
                return parent_example
        else:
            # Unbatched - return features dict from single sample
            if isinstance(parent_example, dict) and 'features' in parent_example:
                return cast(dict[str, object], parent_example['features'])
            elif isinstance(parent_example, list) and parent_example:
                # List of samples - extract features from first
                return cast(dict[str, object], parent_example[0]['features'])
            else:
                return parent_example

    @override
    def get_mlflow_signature(
        self,
        model: Model[object] | None = None,
        n_samples: int = 5,
        predict_kwargs: dict[str, object] | None = None,
    ) -> ModelSignature:
        """
        Generate MLflow signature for lactation model.

        Overrides base class to use unbatched samples, as the LactationPredictor
        expects single dicts (not batched arrays).

        Args:
            model: Optional LactationAutoencoderModel for output inference
            n_samples: Number of samples (only 1 is used for lactation)
            predict_kwargs: Optional kwargs for model.predict()

        Returns:
            mlflow.models.ModelSignature

        Example:
            >>> signature = dataset.get_mlflow_signature(model=model, n_samples=1)
            >>> print(signature.inputs)
        """
        try:
            from mlflow.models import infer_signature
        except ImportError:
            raise ImportError(
                "mlflow is required for signature generation. Install with: pip install mlflow"
            )

        # Get single unbatched sample (lactation predictor handles single dicts)
        input_example = self.get_input_example(n_samples=1, batch=False)

        if model is None:
            # Input-only signature
            return infer_signature(input_example, None)

        # Get predictions to infer output schema
        try:
            predict_kwargs = predict_kwargs or {}

            # Request base format for MLflow signature
            prediction_result = model.predict(
                input_example,
                return_format="base",
                **predict_kwargs
            )

            # Convert to serializable format
            from bovi_core.ml.utils.signature_utils import output_to_serializable
            predictions = output_to_serializable(prediction_result)

            return infer_signature(input_example, predictions)

        except Exception as e:
            # Fall back to input-only signature
            import logging
            logging.warning(
                f"Signature generation failed: {e}. Using input-only signature."
            )
            return infer_signature(input_example, None)


def collate_lactation_batch(batch: list[LactationItem]) -> dict[str, torch.Tensor]:
    """
    Collate function for LactationDataset batches.

    Converts list of dataset items into batched tensors suitable for training.
    Each item from LactationDataset has structure:
    - features: {milk, events, parity, herd_stats}
    - labels: milk array
    - metadata: optional dict

    Args:
        batch: List of LactationItem dicts from LactationDataset.__getitem__

    Returns:
        Dict with batched tensors:
        - milk: (B, 304) float32
        - events: (B, 304) int64
        - parity: (B, 1) float32
        - herd_stats: (B, 10) float32
        - labels: (B, 304) float32

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from lactation_autoencoder.dataloaders.datasets import LactationDataset, collate_lactation_batch
        >>>
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=collate_lactation_batch
        ... )

    """
    features = [item["features"] for item in batch]
    milk = torch.stack([torch.tensor(f["milk"]) for f in features])
    events = torch.stack([torch.tensor(f["events"]) for f in features])
    parity = torch.stack([torch.tensor(f["parity"]) for f in features])
    herd_stats = torch.stack([torch.tensor(f["herd_stats"]) for f in features])
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])

    return {
        "milk": milk,
        "events": events,
        "parity": parity,
        "herd_stats": herd_stats,
        "labels": labels,
    }
