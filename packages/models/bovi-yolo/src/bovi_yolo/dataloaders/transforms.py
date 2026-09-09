"""YOLO-specific image transforms.

Minimal transforms for image validation and optional resizing.
Ultralytics handles model-specific preprocessing internally.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
from bovi_core.ml.dataloaders.transforms.registry import TransformRegistry

logger = logging.getLogger(__name__)


class _YOLOImageTransform(A.ImageOnlyTransform):
    """Albumentations transform with a clear error for invalid image inputs."""

    def __call__(
        self,
        *args: Any,
        force_apply: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        image = kwargs.get("image")
        if "image" in kwargs and not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array for 'image', got {type(image).__name__}")
        return super().__call__(*args, force_apply=force_apply, **kwargs)


@TransformRegistry.register("image_resize")
class ImageResizeTransform(_YOLOImageTransform):
    """Resize images to a target size.

    While ultralytics handles its own internal letterboxing/resize,
    this transform standardizes input images for consistent dataset
    behavior and visualization.

    Params (from config):
        target_size: Target (width, height), default (640, 640).
        keep_aspect_ratio: Pad to maintain aspect ratio, default True.
    """

    target_size: tuple[int, int]
    keep_aspect_ratio: bool

    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        keep_aspect_ratio: bool = True,
        p: float = 1.0,
    ) -> None:
        """Initialize resize transform.

        Args:
            target_size: Target (width, height).
            keep_aspect_ratio: Whether to pad to maintain aspect ratio.
        """
        super().__init__(p=p)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.keep_aspect_ratio = keep_aspect_ratio

    def apply(self, img: npt.NDArray[np.uint8], **params: Any) -> npt.NDArray[np.uint8]:
        """Resize one image while preserving Albumentations pipeline semantics.

        Args:
            img: Image as a NumPy array.
            **params: Parameters supplied by Albumentations.

        Returns:
            Resized image.

        Raises:
            TypeError: If the image is not a NumPy array.
        """
        del params
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array for 'image', got {type(img).__name__}")

        target_w, target_h = self.target_size

        if self.keep_aspect_ratio:
            return self._resize_with_padding(img, target_w, target_h)
        return cast(npt.NDArray[np.uint8], cv2.resize(img, (target_w, target_h)))

    def _resize_with_padding(
        self,
        image: npt.NDArray[np.uint8],
        target_w: int,
        target_h: int,
    ) -> npt.NDArray[np.uint8]:
        """Resize image maintaining aspect ratio with padding.

        Args:
            image: Input image (H, W, C).
            target_w: Target width.
            target_h: Target height.

        Returns:
            Resized and padded image.
        """
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        padded: npt.NDArray[np.uint8] = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return padded

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Expose constructor arguments for Albumentations serialization."""
        return ("target_size", "keep_aspect_ratio")


@TransformRegistry.register("image_validation")
class ImageValidationTransform(_YOLOImageTransform):
    """Validate image format and channels before processing.

    Ensures images are RGB, non-empty, and within expected size bounds.
    Raises on invalid input.

    Params (from config):
        min_size: Minimum dimension, default 32.
        max_size: Maximum dimension, default 8192.
        required_channels: Expected channels, default 3.
    """

    min_size: int
    max_size: int
    required_channels: int

    def __init__(
        self,
        min_size: int = 32,
        max_size: int = 8192,
        required_channels: int = 3,
        p: float = 1.0,
    ) -> None:
        """Initialize validation transform.

        Args:
            min_size: Minimum image dimension.
            max_size: Maximum image dimension.
            required_channels: Expected number of channels.
        """
        super().__init__(p=p)
        self.min_size = min_size
        self.max_size = max_size
        self.required_channels = required_channels

    def apply(self, img: npt.NDArray[np.uint8], **params: Any) -> npt.NDArray[np.uint8]:
        """Validate one image while preserving Albumentations pipeline semantics.

        Args:
            img: Image as a NumPy array.
            **params: Parameters supplied by Albumentations.

        Returns:
            The unchanged, validated image.

        Raises:
            TypeError: If the image is not a NumPy array.
            ValueError: If image fails validation checks.
        """
        del params
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(img).__name__}")

        if img.size == 0:
            raise ValueError("Image is empty (size=0)")

        if img.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got {img.ndim}D")

        h, w, c = img.shape

        if c != self.required_channels:
            raise ValueError(f"Expected {self.required_channels} channels, got {c}")

        if h < self.min_size or w < self.min_size:
            raise ValueError(
                f"Image too small: ({h}, {w}), minimum is ({self.min_size}, {self.min_size})"
            )

        if h > self.max_size or w > self.max_size:
            raise ValueError(
                f"Image too large: ({h}, {w}), maximum is ({self.max_size}, {self.max_size})"
            )

        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Expose constructor arguments for Albumentations serialization."""
        return ("min_size", "max_size", "required_channels")
