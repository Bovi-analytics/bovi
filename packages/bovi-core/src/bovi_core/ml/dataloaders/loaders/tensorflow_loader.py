"""
TensorFlow DataLoader implementation.

Wraps tf.data.Dataset with sensible defaults.
Applies transforms with explicit shape setting.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

from ..base import AbstractDataLoader, Dataset

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class TensorFlowDataLoader(AbstractDataLoader):
    """
    TensorFlow DataLoader wrapper.

    Wraps tf.data.Dataset with optimized defaults for training and inference.
    Transforms are applied with explicit shape setting to prevent graph errors.

    Key features:
    - Automatic prefetching with AUTOTUNE
    - Parallel data loading with num_parallel_calls=AUTOTUNE
    - Transform support with explicit shape inference
    - Configurable batch size and shuffle

    Args:
        dataset: Dataset to load from (returns raw NumPy)
        config: Config instance
        split: Dataset split ("train", "val", "test")
        model_name: Model name for config lookup
        transform: Optional Albumentations transform to apply per-sample
        batch_size: Batch size (overrides config)
        shuffle: Whether to shuffle (overrides config default)
        buffer_size: Shuffle buffer size (default: 1000)
        prefetch_buffer_size: Number of batches to prefetch (default: AUTOTUNE)
        cache: Whether to cache dataset in memory (default: False)

    Example:
        ```python
        import albumentations as A
        from bovi_core.ml.dataloaders import (
            ImageDataset,
            LocalFileSource,
            TensorFlowDataLoader,
        )

        # Create dataset (no transforms!)
        source = LocalFileSource("data/images", file_pattern="*.jpg")
        dataset = ImageDataset(source)  # Returns raw NumPy (H, W, C), uint8

        # Create transform
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create loader WITH transform
        loader = TensorFlowDataLoader(
            dataset,
            config=config,
            split="train",
            transform=transform,
            batch_size=32,
        )

        # Iterate - images are float32 with explicit shapes
        for batch in loader:
            images = batch["image"]  # (32, 224, 224, 3) float32
            labels = batch["label"]
            # ... training code
        ```
    """

    # Type annotations for instance attributes
    transform: Callable[..., dict[str, object]] | None
    _output_shapes: dict[str, tuple[int | None, ...]]
    batch_size: int
    shuffle: bool
    buffer_size: int
    prefetch_buffer_size: int
    cache: bool
    _tf_dataset: object | None  # tf.data.Dataset, but TF types are complex

    def __init__(
        self,
        dataset: Dataset,
        config: Config,
        split: str = "train",
        model_name: str | None = None,
        transform: Callable[..., dict[str, object]] | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        buffer_size: int = 1000,
        prefetch_buffer_size: int | None = None,
        cache: bool = False,
    ) -> None:
        super().__init__(dataset, config, split, model_name)

        try:
            import tensorflow as tf
        except ImportError as err:
            raise ImportError(
                "TensorFlow is required for TensorFlowDataLoader. "
                "Install with: pip install tensorflow"
            ) from err

        self.transform = transform
        self._output_shapes = {}

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
        self.buffer_size = buffer_size

        # Auto-tune prefetch
        if prefetch_buffer_size is None:
            self.prefetch_buffer_size = int(tf.data.AUTOTUNE)
        else:
            self.prefetch_buffer_size = prefetch_buffer_size

        self.cache = cache

        # Perform dry run to infer output shapes after transform
        self._infer_output_shapes()

        # Create TensorFlow Dataset
        self._tf_dataset = None
        self._create_tensorflow_dataset()

        logger.info(
            f"TensorFlowDataLoader ({split}): "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"transform={transform is not None}, "
            f"cache={self.cache}"
        )

    def _infer_output_shapes(self) -> None:
        """
        Dry run to infer output shapes after transforms.

        CRITICAL: This prevents the "Broken Shape" problem where TF loses
        dimension info after tf.numpy_function.
        """
        if len(self.dataset) == 0:
            return

        # Get a sample
        sample = self.dataset[0]

        # Apply transform if provided
        if self.transform is not None and "image" in sample:
            image = sample["image"]
            if isinstance(image, np.ndarray):
                transformed = self.transform(image=image)
                transformed_image = transformed["image"]
                # Normalize uint8 to float32
                if isinstance(transformed_image, np.ndarray):
                    if transformed_image.dtype == np.uint8:
                        transformed_image = transformed_image.astype(np.float32) / 255.0
                    sample = {**sample, "image": transformed_image}

        # Record shapes
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                self._output_shapes[key] = value.shape
            elif hasattr(value, "shape"):
                self._output_shapes[key] = tuple(value.shape)

        logger.debug(f"Inferred output shapes: {self._output_shapes}")

    def _generator(self) -> Iterator[dict[str, object]]:
        """Generator function for tf.data.Dataset."""
        for i in range(len(self.dataset)):
            sample: dict[str, object] = self.dataset[i]

            # Apply transform if provided
            if self.transform is not None and "image" in sample:
                image = sample["image"]
                if isinstance(image, np.ndarray):
                    transformed = self.transform(image=image)
                    transformed_image = transformed["image"]
                    # Normalize uint8 to float32
                    if isinstance(transformed_image, np.ndarray):
                        if transformed_image.dtype == np.uint8:
                            transformed_image = transformed_image.astype(np.float32) / 255.0
                        sample = {**sample, "image": transformed_image}

            yield sample

    def _create_tensorflow_dataset(self) -> None:
        """Create the underlying TensorFlow Dataset."""
        import tensorflow as tf

        # Infer output signature from shapes
        if len(self.dataset) > 0:
            sample: dict[str, object] = self.dataset[0]

            # Apply transform for signature inference
            if self.transform is not None and "image" in sample:
                image = sample["image"]
                if isinstance(image, np.ndarray):
                    transformed = self.transform(image=image)
                    transformed_image = transformed["image"]
                    if isinstance(transformed_image, np.ndarray):
                        if transformed_image.dtype == np.uint8:
                            transformed_image = transformed_image.astype(np.float32) / 255.0
                        sample = {**sample, "image": transformed_image}

            output_signature: dict[str, tf.TensorSpec] = {}

            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    # Use inferred shape if available
                    shape = self._output_shapes.get(key, value.shape)
                    dtype = (
                        tf.float32
                        if value.dtype in [np.float32, np.float64]
                        else tf.dtypes.as_dtype(value.dtype)
                    )
                    output_signature[key] = tf.TensorSpec(shape=shape, dtype=dtype)
                elif isinstance(value, (int, np.integer)):
                    output_signature[key] = tf.TensorSpec(shape=(), dtype=tf.int64)
                elif isinstance(value, (float, np.floating)):
                    output_signature[key] = tf.TensorSpec(shape=(), dtype=tf.float32)
                elif isinstance(value, str):
                    output_signature[key] = tf.TensorSpec(shape=(), dtype=tf.string)
                elif value is None:
                    # Skip None values
                    continue
                else:
                    output_signature[key] = tf.TensorSpec(shape=(), dtype=tf.string)

            # Create dataset from generator
            ds = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        else:
            # Empty dataset - create empty dataset with range(0)
            ds = tf.data.Dataset.range(0)

        # Apply transformations with AUTOTUNE for parallelization
        if self.cache:
            ds = ds.cache()

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.buffer_size)

        ds = ds.batch(self.batch_size)

        # CRITICAL: Use AUTOTUNE for parallel prefetching
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        self._tf_dataset = ds

    def get_pytorch_loader(self) -> None:
        """
        PyTorch not supported by TensorFlowDataLoader.

        Returns:
            None
        """
        return None

    def get_tensorflow_dataset(self) -> object | None:
        """
        Return TensorFlow Dataset instance.

        Returns:
            tf.data.Dataset or None
        """
        return self._tf_dataset

    def get_sklearn_iterator(self) -> None:
        """
        Simple iterator not needed (use TensorFlow dataset).

        Returns:
            None
        """
        return None

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Iterate over batches."""
        if self._tf_dataset is None:
            raise RuntimeError("TensorFlow Dataset not initialized")
        return iter(self._tf_dataset)  # type: ignore[call-overload]

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

    @property
    def element_spec(self) -> object | None:
        """Return the element spec of the dataset (for debugging shapes)."""
        if self._tf_dataset is None:
            return None
        return getattr(self._tf_dataset, "element_spec", None)
