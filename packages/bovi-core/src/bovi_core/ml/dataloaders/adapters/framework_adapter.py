"""Framework adapter utilities for converting NumPy data to framework-specific formats."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import tensorflow as tf

# Type alias for sample dictionaries (keys are strings, values can be arrays, scalars, etc.)
SampleDict = dict[str, Any]


class FrameworkAdapter:
    """Static utilities for converting NumPy data to framework-specific formats."""

    @staticmethod
    def numpy_to_pytorch_collate(
        batch: list[SampleDict],
        transform: Callable[..., dict[str, Any]] | None = None,
        auto_transpose: bool = True,
        auto_normalize: bool = True,
    ) -> SampleDict:
        """
        Custom collate_fn for PyTorch DataLoader.

        Args:
            batch: List of dicts from Dataset.__getitem__
            transform: Optional Albumentations transform to apply per-sample
            auto_transpose: If True, convert HWC→CHW for 3D image arrays
            auto_normalize: If True, convert uint8→float32/255.0

        Returns:
            Dict of batched PyTorch tensors
        """
        import torch

        if not batch:
            return {}

        # Apply transforms to each sample if provided
        if transform is not None:
            transformed_batch: list[SampleDict] = []
            for item in batch:
                # Apply albumentations transform to image
                if "image" in item and isinstance(item["image"], np.ndarray):
                    transformed = transform(image=item["image"])
                    item = {**item, "image": transformed["image"]}
                elif "frames" in item and isinstance(item["frames"], np.ndarray):
                    # Video: apply transform to each frame
                    frames: NDArray[np.uint8] = item["frames"]
                    transformed_frames: list[NDArray[np.uint8]] = []
                    for i in range(frames.shape[0]):
                        transformed = transform(image=frames[i])
                        transformed_frames.append(transformed["image"])
                    item = {**item, "frames": np.stack(transformed_frames)}
                transformed_batch.append(item)
            batch = transformed_batch

        keys = batch[0].keys()
        collated: SampleDict = {}

        for key in keys:
            items = [item[key] for item in batch]
            first = items[0]

            if first is None:
                collated[key] = items

            elif isinstance(first, (str, dict)):
                # Metadata - keep as list
                collated[key] = items

            elif isinstance(first, np.ndarray):
                arr: NDArray[Any] = np.stack(items)

                # Auto-normalize uint8 images
                if auto_normalize and arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0

                # Auto-transpose HWC → CHW for images
                # Shape: (B, H, W, C) where C in [1, 3, 4]
                if auto_transpose and arr.ndim == 4 and arr.shape[-1] in [1, 3, 4]:
                    arr = np.transpose(arr, (0, 3, 1, 2))  # BHWC → BCHW

                # Video: (B, T, H, W, C) → (B, T, C, H, W)
                if auto_transpose and arr.ndim == 5 and arr.shape[-1] in [1, 3, 4]:
                    arr = np.transpose(arr, (0, 1, 4, 2, 3))  # BTHWC → BTCHW

                collated[key] = torch.as_tensor(arr)

            elif isinstance(first, (int, float)):
                collated[key] = torch.as_tensor(np.array(items))

            else:
                collated[key] = items

        return collated

    @staticmethod
    def numpy_to_tensorflow_op(
        func: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        output_shape: tuple[int | None, ...],
        output_dtype: tf.DType | None = None,
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Wrap a NumPy function as a TensorFlow operation.

        CRITICAL: Explicitly sets shape after tf.numpy_function to prevent
        the "Broken Shape" problem where TF loses dimension info.

        Args:
            func: NumPy-based transform function
            output_shape: Expected output shape (e.g., (224, 224, 3))
            output_dtype: TF dtype for output (default: tf.float32)

        Returns:
            TensorFlow-compatible wrapper function
        """
        import tensorflow as tf

        resolved_dtype = output_dtype if output_dtype is not None else tf.float32

        def tf_wrapper(tensor: tf.Tensor) -> tf.Tensor:
            result = tf.numpy_function(func, [tensor], Tout=resolved_dtype)
            result.set_shape(output_shape)  # CRITICAL: Restore shape info
            return result

        return tf_wrapper

    @staticmethod
    def create_tensorflow_transform_fn(
        transform: Callable[..., dict[str, Any]],
        output_shapes: dict[str, tuple[int | None, ...]],
        output_dtypes: dict[str, tf.DType] | None = None,
    ) -> Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]]:
        """
        Create a TensorFlow map function that applies transforms.

        Args:
            transform: Albumentations transform to apply
            output_shapes: Dict mapping keys to expected output shapes
            output_dtypes: Dict mapping keys to TF dtypes (default: float32 for images)

        Returns:
            Function suitable for tf.data.Dataset.map()
        """
        import tensorflow as tf

        resolved_dtypes = output_dtypes if output_dtypes is not None else {}

        def apply_transform(sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
            def transform_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
                transformed = transform(image=image)
                result = transformed["image"]
                # Normalize uint8 to float32
                if isinstance(result, np.ndarray) and result.dtype == np.uint8:
                    return result.astype(np.float32) / 255.0
                return result

            result: dict[str, tf.Tensor] = {}
            for key, value in sample.items():
                if key == "image":
                    dtype = resolved_dtypes.get(key, tf.float32)
                    shape = output_shapes.get(key)
                    transformed_tensor = tf.numpy_function(transform_image, [value], Tout=dtype)
                    if shape is not None:
                        transformed_tensor.set_shape(shape)
                    result[key] = transformed_tensor
                else:
                    result[key] = value

            return result

        return apply_transform
