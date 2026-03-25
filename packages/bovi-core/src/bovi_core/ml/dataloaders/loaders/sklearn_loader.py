"""
Sklearn-compatible DataLoader implementation.

Provides simple iterator for sklearn and other ML libraries.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..base import AbstractDataLoader, Dataset

# Type alias for index arrays
IndexArray = NDArray[np.intp]

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class SklearnDataLoader(AbstractDataLoader):
    """
    Sklearn-compatible DataLoader.

    Provides a simple iterator interface for sklearn and other
    traditional ML libraries that don't have specialized data loading.

    Key features:
    - Simple batch iteration
    - Memory-efficient processing
    - Compatible with numpy arrays
    - Optional shuffling

    Args:
        dataset: Dataset to load from.
        config: Config instance.
        split: Dataset split ("train", "val", "test").
        batch_size: Batch size (overrides config).
        shuffle: Whether to shuffle (overrides config default).
        seed: Random seed for shuffling (default: 42).

    Example:
        ```python
        from bovi_core.ml.dataloaders import (
            ImageDataset,
            LocalFileSource,
            SklearnDataLoader
        )

        # Create dataset (without transforms for sklearn)
        source = LocalFileSource("data/features", file_pattern="*.npy")
        dataset = ImageDataset(source)

        # Create loader
        loader = SklearnDataLoader(
            dataset,
            config=config,
            split="train",
            batch_size=32
        )

        # Iterate
        for batch in loader:
            features = batch["image"]  # numpy arrays
            labels = batch["label"]
            # ... sklearn model.fit(features, labels)
        ```
    """

    # Type annotations for instance attributes
    batch_size: int
    shuffle: bool
    seed: int
    indices: IndexArray

    def __init__(
        self,
        dataset: Dataset,
        config: Config,
        split: str = "train",
        model_name: str | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset, config, split, model_name)

        # Get config for this split (if available)
        split_config = None
        dataloader_config = None
        if model_name and hasattr(config.experiment, "models"):
            model_config = getattr(config.experiment.models, model_name, None)
            if model_config and hasattr(model_config, "dataloaders"):
                split_config = getattr(model_config.dataloaders, split, None)
                # Get nested dataloader config if it exists
                if split_config and hasattr(split_config, "dataloader"):
                    dataloader_config = split_config.dataloader

        # Determine parameters with fallback to config
        resolved_batch_size: int
        if batch_size is not None:
            resolved_batch_size = batch_size
        elif dataloader_config and hasattr(dataloader_config, "batch_size"):
            resolved_batch_size = int(dataloader_config.batch_size)
        elif split_config and hasattr(split_config, "batch_size"):
            resolved_batch_size = int(split_config.batch_size)
        else:
            resolved_batch_size = 32
        self.batch_size = resolved_batch_size

        # Default shuffle: True for train, False for val/test
        resolved_shuffle: bool
        if shuffle is not None:
            resolved_shuffle = shuffle
        elif dataloader_config and hasattr(dataloader_config, "shuffle"):
            resolved_shuffle = bool(dataloader_config.shuffle)
        elif split_config and hasattr(split_config, "shuffle"):
            resolved_shuffle = bool(split_config.shuffle)
        else:
            resolved_shuffle = split == "train"
        self.shuffle = resolved_shuffle
        self.seed = seed

        # Create index order
        self._reset_indices()

        logger.info(
            f"SklearnDataLoader ({split}): batch_size={self.batch_size}, shuffle={self.shuffle}"
        )

    def _reset_indices(self) -> None:
        """Reset and potentially shuffle indices."""
        self.indices = np.arange(len(self.dataset))

        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.indices)

    def get_pytorch_loader(self) -> None:
        """
        PyTorch not supported by SklearnDataLoader.

        Returns:
            None.
        """
        return None

    def get_tensorflow_dataset(self) -> None:
        """
        TensorFlow not supported by SklearnDataLoader.

        Returns:
            None.
        """
        return None

    def get_sklearn_iterator(self) -> Iterator[dict[str, object]]:
        """
        Return simple iterator.

        Returns:
            Iterator over batches.
        """
        return iter(self)

    def __iter__(self) -> Iterator[dict[str, object]]:
        """
        Iterate over batches.

        Yields:
            Dict with batched data.
        """
        # Reset indices for new epoch
        self._reset_indices()

        # Iterate in batches
        for start_idx in range(0, len(self.dataset), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start_idx:end_idx]

            # Load batch
            batch_items = [self.dataset[int(idx)] for idx in batch_indices]

            # Collate batch
            if not batch_items:
                continue

            # Get keys from first item
            keys = batch_items[0].keys()

            collated: dict[str, object] = {}
            for key in keys:
                items = [item[key] for item in batch_items]

                # Handle different types
                if items[0] is None:
                    # Keep None as list
                    collated[key] = items
                elif isinstance(items[0], (int, float)):
                    # Numbers -> numpy array
                    collated[key] = np.array(items)
                elif isinstance(items[0], str):
                    # Strings -> list
                    collated[key] = items
                elif hasattr(items[0], "shape"):
                    # Arrays/tensors -> stack
                    try:
                        # Try to stack
                        collated[key] = np.stack(
                            [
                                np.array(item) if not isinstance(item, np.ndarray) else item
                                for item in items
                            ]
                        )
                    except (ValueError, TypeError):
                        # Can't stack -> keep as list
                        collated[key] = items
                else:
                    # Other types -> list
                    collated[key] = items

            yield collated

    def __len__(self) -> int:
        """Number of batches."""
        return math.ceil(len(self.dataset) / self.batch_size)

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch."""
        return len(self)

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self.dataset)

    def get_all_data(self) -> dict[str, object]:
        """
        Load all data into memory at once.

        Useful for sklearn models that need full dataset.

        Returns:
            Dict with all data as numpy arrays.
        """
        # Load all items
        all_items = [self.dataset[i] for i in range(len(self.dataset))]

        if not all_items:
            return {}

        # Get keys from first item
        keys = all_items[0].keys()

        result: dict[str, object] = {}
        for key in keys:
            items = [item[key] for item in all_items]

            # Convert to numpy arrays where possible
            if items[0] is None:
                result[key] = items
            elif isinstance(items[0], (int, float)):
                result[key] = np.array(items)
            elif isinstance(items[0], str):
                result[key] = items
            elif hasattr(items[0], "shape"):
                try:
                    result[key] = np.stack(
                        [
                            np.array(item) if not isinstance(item, np.ndarray) else item
                            for item in items
                        ]
                    )
                except (ValueError, TypeError):
                    result[key] = items
            else:
                result[key] = items

        return result
