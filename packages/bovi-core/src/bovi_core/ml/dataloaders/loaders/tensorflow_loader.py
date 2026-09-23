"""
TensorFlow DataLoader implementation.

Wraps tf.data.Dataset with sensible defaults.
Applies transforms with explicit shape setting.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Iterator, Mapping
from typing import Any

import numpy as np

from bovi_core.ml.dataloaders.datasets.base_dataset import Dataset
from bovi_core.ml.dataloaders.loaders.base_loader import AbstractDataLoader

logger = logging.getLogger(__name__)


class TensorFlowDataLoader(AbstractDataLoader):
    """Build native tf.data batches from framework-neutral dataset samples.

    Use TransformedDataset for decoded-sample preprocessing. No image
    normalization or layout conversion is performed by this loader.

    output_signature can specify nested per-sample TensorSpecs. Without it,
    one sample is read to infer shapes and dtypes; random transforms therefore
    need a suitable explicit signature when their output shapes can vary.
    drop_keys explicitly omits top-level fields such as opaque metadata.
    cache stores prepared samples, so random transforms are not rerun each
    epoch when caching is enabled.

    seed and set_epoch control shuffle order, not transform RNGs.
    """

    # Type annotations for instance attributes
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
        *,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool | None = None,
        buffer_size: int = 1000,
        prefetch_buffer_size: int | None = None,
        cache: bool = False,
        output_signature: Any | None = None,
        drop_keys: Collection[str] = (),
        seed: int | None = None,
        reshuffle_each_iteration: bool = True,
    ) -> None:
        super().__init__(dataset, split=split)

        try:
            import tensorflow as tf
        except ImportError as err:
            raise ImportError(
                "TensorFlow is required for TensorFlowDataLoader. "
                "Install with: pip install tensorflow"
            ) from err

        self.output_signature = output_signature
        self.drop_keys = tuple(drop_keys)
        self.seed = seed
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self._epoch: int | None = None
        self._tf_source: Any | None = None
        self._output_shapes = {}

        self.batch_size = batch_size

        # Default shuffle: True for train, False for val/test
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.buffer_size = buffer_size

        # Auto-tune prefetch
        if prefetch_buffer_size is None:
            self.prefetch_buffer_size = int(tf.data.AUTOTUNE)
        else:
            self.prefetch_buffer_size = prefetch_buffer_size

        self.cache = cache

        # Perform dry run to infer output shapes after transform
        if self.output_signature is None:
            self._infer_output_shapes()

        # Create TensorFlow Dataset
        self._tf_dataset = None
        self._create_tensorflow_dataset()

        logger.info(
            f"TensorFlowDataLoader ({split}): "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"cache={self.cache}"
        )

    def _infer_output_shapes(self) -> None:
        """Read one dataset sample to infer the generator output signature."""
        if len(self.dataset) == 0:
            return

        import tensorflow as tf

        sample = self._prepare_sample(self.dataset[0])
        self.output_signature = tf.nest.map_structure(
            lambda value: tf.TensorSpec(shape=value.shape, dtype=value.dtype), sample
        )

        # Record shapes
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                self._output_shapes[key] = value.shape
            elif hasattr(value, "shape"):
                self._output_shapes[key] = tuple(value.shape)

        logger.debug(f"Inferred output shapes: {self._output_shapes}")

    def _prepare_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Omit configured fields and convert nested tensor-compatible leaves."""
        import tensorflow as tf

        sample = {key: value for key, value in sample.items() if key not in self.drop_keys}

        def convert(value: Any, path: str) -> Any:
            if isinstance(value, Mapping):
                return {key: convert(item, f"{path}.{key}") for key, item in value.items()}
            try:
                array = np.asarray(value)
                if array.dtype.kind not in "biufcSU":
                    raise TypeError("expected numeric, boolean, or string data")
                return tf.convert_to_tensor(array)
            except (TypeError, ValueError) as err:
                raise TypeError(
                    f"TensorFlow sample field {path!r} is not tensor-compatible; "
                    "convert it in the dataset or omit its top-level field with drop_keys"
                ) from err

        return {key: convert(value, key) for key, value in sample.items()}

    def _generator(self) -> Iterator[dict[str, Any]]:
        """Generator function for tf.data.Dataset."""
        for i in range(len(self.dataset)):
            yield self._prepare_sample(self.dataset[i])

    def _create_tensorflow_dataset(self) -> None:
        """Create the underlying TensorFlow Dataset."""
        import tensorflow as tf

        if self._tf_source is None:
            if self.output_signature is not None:
                ds = tf.data.Dataset.from_generator(
                    self._generator, output_signature=self.output_signature
                )
            else:
                ds = tf.data.Dataset.range(0)
            if self.cache:
                ds = ds.cache()
            self._tf_source = ds
        ds = self._tf_source

        if self.shuffle:
            seed = self.seed
            if seed is not None and self._epoch is not None:
                seed = (seed + self._epoch) % (2**31 - 1)
            ds = ds.shuffle(
                buffer_size=self.buffer_size,
                seed=seed,
                reshuffle_each_iteration=(
                    self.reshuffle_each_iteration if self._epoch is None else False
                ),
            )

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(buffer_size=self.prefetch_buffer_size)

        self._tf_dataset = ds

    def set_epoch(self, epoch: int) -> None:
        """Pin sample order for training and metrics passes, retaining cached samples."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if self.shuffle and self.seed is None:
            raise ValueError("set_epoch requires seed when shuffling")
        self._epoch = epoch
        self._create_tensorflow_dataset()

    def __iter__(self) -> Iterator[dict[str, Any]]:
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
    def element_spec(self) -> Any | None:
        """Return the element spec of the dataset (for debugging shapes)."""
        if self._tf_dataset is None:
            return None
        return getattr(self._tf_dataset, "element_spec", None)
