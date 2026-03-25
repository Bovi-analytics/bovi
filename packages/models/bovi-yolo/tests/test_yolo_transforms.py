"""Tests for YOLO image transforms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


class TestImageValidationTransform:
    def test_initialization(self) -> None:
        """Test transform initializes with defaults."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        assert transform.min_size == 32
        assert transform.max_size == 8192
        assert transform.required_channels == 3

    def test_valid_image_passes(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test valid RGB image passes validation."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        result = transform({"image": sample_image})
        assert np.array_equal(result["image"], sample_image)

    def test_missing_image_key_raises(self) -> None:
        """Test missing 'image' key raises KeyError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        with pytest.raises(KeyError, match="image"):
            transform({"data": np.zeros((100, 100, 3), dtype=np.uint8)})

    def test_non_numpy_raises(self) -> None:
        """Test non-numpy input raises TypeError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        with pytest.raises(TypeError, match="numpy array"):
            transform({"image": "not_an_array"})

    def test_empty_image_raises(self) -> None:
        """Test empty image raises ValueError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        with pytest.raises(ValueError, match="empty"):
            transform({"image": np.array([], dtype=np.uint8).reshape(0, 0, 3)})

    def test_2d_image_raises(self) -> None:
        """Test 2D (grayscale) image raises ValueError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        with pytest.raises(ValueError, match="3D"):
            transform({"image": np.zeros((100, 100), dtype=np.uint8)})

    def test_wrong_channels_raises(self) -> None:
        """Test wrong number of channels raises ValueError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        with pytest.raises(ValueError, match="channels"):
            transform({"image": np.zeros((100, 100, 4), dtype=np.uint8)})

    def test_image_too_small_raises(self) -> None:
        """Test image below min_size raises ValueError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform(min_size=64)
        with pytest.raises(ValueError, match="too small"):
            transform({"image": np.zeros((32, 32, 3), dtype=np.uint8)})

    def test_image_too_large_raises(self) -> None:
        """Test image above max_size raises ValueError."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform(max_size=100)
        with pytest.raises(ValueError, match="too large"):
            transform({"image": np.zeros((200, 200, 3), dtype=np.uint8)})

    def test_custom_params(self) -> None:
        """Test custom initialization parameters."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform(
            min_size=64, max_size=1024, required_channels=1
        )
        assert transform.min_size == 64
        assert transform.max_size == 1024
        assert transform.required_channels == 1

    def test_get_params(self) -> None:
        """Test get_params returns correct dict."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform(min_size=64, max_size=4096)
        params = transform.get_params()
        assert params["name"] == "image_validation"
        assert params["min_size"] == 64
        assert params["max_size"] == 4096

    def test_preserves_other_data_keys(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test validation preserves other keys in data dict."""
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = ImageValidationTransform()
        data = {"image": sample_image, "label": "cow", "extra": 42}
        result = transform(data)
        assert result["label"] == "cow"
        assert result["extra"] == 42

    def test_registry_registration(self) -> None:
        """Test transform is registered in TransformRegistry."""
        from bovi_core.ml.dataloaders.transforms.registry import (
            TransformRegistry,
        )
        from bovi_yolo.dataloaders.transforms import ImageValidationTransform

        transform = TransformRegistry.create("image_validation")
        assert isinstance(transform, ImageValidationTransform)


class TestImageResizeTransform:
    def test_initialization_defaults(self) -> None:
        """Test transform initializes with defaults."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform()
        assert transform.target_size == (640, 640)
        assert transform.keep_aspect_ratio is True

    def test_resize_without_aspect_ratio(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test resize without keeping aspect ratio."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform(
            target_size=(320, 320), keep_aspect_ratio=False
        )
        result = transform({"image": sample_image})
        assert result["image"].shape == (320, 320, 3)

    def test_resize_with_aspect_ratio(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test resize with aspect ratio preservation and padding."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform(
            target_size=(640, 640), keep_aspect_ratio=True
        )
        result = transform({"image": sample_image})
        assert result["image"].shape == (640, 640, 3)

    def test_padding_uses_gray_value(self) -> None:
        """Test padding uses 114 gray value (YOLO convention)."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        # Create a small image that will need padding
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        transform = ImageResizeTransform(
            target_size=(640, 640), keep_aspect_ratio=True
        )
        result = transform({"image": image})

        # Check corner pixels (should be padded with 114)
        assert result["image"][0, 0, 0] == 114

    def test_missing_image_raises(self) -> None:
        """Test missing 'image' key raises KeyError."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform()
        with pytest.raises(KeyError, match="image"):
            transform({"data": np.zeros((100, 100, 3))})

    def test_non_numpy_raises(self) -> None:
        """Test non-numpy input raises TypeError."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform()
        with pytest.raises(TypeError, match="numpy array"):
            transform({"image": [1, 2, 3]})

    def test_get_params(self) -> None:
        """Test get_params returns correct dict."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform(
            target_size=(320, 320), keep_aspect_ratio=False
        )
        params = transform.get_params()
        assert params["name"] == "image_resize"
        assert params["target_size"] == (320, 320)
        assert params["keep_aspect_ratio"] is False

    def test_preserves_dtype(
        self, sample_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test resize preserves uint8 dtype."""
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = ImageResizeTransform(target_size=(320, 320))
        result = transform({"image": sample_image})
        assert result["image"].dtype == np.uint8

    def test_registry_registration(self) -> None:
        """Test transform is registered in TransformRegistry."""
        from bovi_core.ml.dataloaders.transforms.registry import (
            TransformRegistry,
        )
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        transform = TransformRegistry.create("image_resize")
        assert isinstance(transform, ImageResizeTransform)
