"""Tests for vision transforms using Albumentations.

These tests verify that the Albumentations transforms work correctly
as replacements for the old vision.py wrapper classes.
"""

import albumentations as A
import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """64x64 RGB uint8 image for testing."""
    np.random.seed(42)  # Deterministic for reproducibility
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# --- Individual Transform Tests ---


class TestResize:
    """Test A.Resize (replaces ResizeTransform)."""

    def test_resize_square(self, sample_image):
        transform = A.Resize(32, 32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)

    def test_resize_rectangular(self, sample_image):
        transform = A.Resize(48, 32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (48, 32, 3)

    def test_resize_preserves_dtype(self, sample_image):
        transform = A.Resize(32, 32)
        result = transform(image=sample_image)["image"]
        assert result.dtype == np.uint8

    def test_resize_upscale(self, sample_image):
        transform = A.Resize(128, 128)
        result = transform(image=sample_image)["image"]
        assert result.shape == (128, 128, 3)


class TestNormalize:
    """Test A.Normalize (replaces NormalizeTransform)."""

    def test_normalize_imagenet(self, sample_image):
        transform = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        result = transform(image=sample_image)["image"]
        assert result.dtype == np.float32

    def test_normalize_range(self, sample_image):
        transform = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        result = transform(image=sample_image)["image"]
        # Normalized values should be roughly in [-1, 1] range
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_normalize_preserves_shape(self, sample_image):
        transform = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape


class TestHorizontalFlip:
    """Test A.HorizontalFlip (replaces RandomHorizontalFlipTransform)."""

    def test_flip_deterministic(self, sample_image):
        transform = A.HorizontalFlip(p=1.0)
        result = transform(image=sample_image)["image"]
        # Flipped image should be mirror of original
        np.testing.assert_array_equal(result, np.fliplr(sample_image))

    def test_flip_preserves_shape(self, sample_image):
        transform = A.HorizontalFlip(p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape

    def test_flip_preserves_dtype(self, sample_image):
        transform = A.HorizontalFlip(p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.dtype == sample_image.dtype


class TestVerticalFlip:
    """Test A.VerticalFlip (replaces RandomVerticalFlipTransform)."""

    def test_flip_deterministic(self, sample_image):
        transform = A.VerticalFlip(p=1.0)
        result = transform(image=sample_image)["image"]
        np.testing.assert_array_equal(result, np.flipud(sample_image))

    def test_flip_preserves_shape(self, sample_image):
        transform = A.VerticalFlip(p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape


class TestRotate:
    """Test A.Rotate (replaces RandomRotationTransform)."""

    def test_rotate_preserves_shape(self, sample_image):
        transform = A.Rotate(limit=45, p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape

    def test_rotate_90_degrees(self, sample_image):
        transform = A.Rotate(limit=(90, 90), p=1.0, border_mode=0)
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape

    def test_rotate_preserves_dtype(self, sample_image):
        transform = A.Rotate(limit=30, p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.dtype == sample_image.dtype


class TestRandomBrightnessContrast:
    """Test A.RandomBrightnessContrast (replaces RandomBrightnessContrastTransform)."""

    def test_modifies_values(self, sample_image):
        transform = A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0)
        result = transform(image=sample_image)["image"]
        # Values should be modified (not equal to original)
        assert not np.array_equal(result, sample_image)

    def test_preserves_shape(self, sample_image):
        transform = A.RandomBrightnessContrast(p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape

    def test_preserves_dtype(self, sample_image):
        transform = A.RandomBrightnessContrast(p=1.0)
        result = transform(image=sample_image)["image"]
        assert result.dtype == sample_image.dtype


class TestRandomCrop:
    """Test A.RandomCrop (replaces RandomCropTransform)."""

    def test_crop_square(self, sample_image):
        transform = A.RandomCrop(32, 32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)

    def test_crop_rectangular(self, sample_image):
        transform = A.RandomCrop(48, 32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (48, 32, 3)

    def test_crop_preserves_dtype(self, sample_image):
        transform = A.RandomCrop(32, 32)
        result = transform(image=sample_image)["image"]
        assert result.dtype == sample_image.dtype


class TestCenterCrop:
    """Test A.CenterCrop (replaces CenterCropTransform)."""

    def test_center_crop_output_shape(self, sample_image):
        transform = A.CenterCrop(32, 32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)

    def test_center_crop_extracts_center(self, sample_image):
        transform = A.CenterCrop(32, 32)
        result = transform(image=sample_image)["image"]
        # Verify it's actually the center
        expected = sample_image[16:48, 16:48, :]
        np.testing.assert_array_equal(result, expected)

    def test_center_crop_rectangular(self, sample_image):
        transform = A.CenterCrop(32, 48)
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 48, 3)


class TestToFloat:
    """Test A.ToFloat (replaces ToTensorTransform for normalization)."""

    def test_converts_to_float32(self, sample_image):
        transform = A.ToFloat()
        result = transform(image=sample_image)["image"]
        assert result.dtype == np.float32

    def test_normalizes_to_0_1(self, sample_image):
        transform = A.ToFloat()
        result = transform(image=sample_image)["image"]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preserves_shape(self, sample_image):
        transform = A.ToFloat()
        result = transform(image=sample_image)["image"]
        assert result.shape == sample_image.shape


# --- Pipeline Tests ---


class TestComposePipeline:
    """Test A.Compose pipelines."""

    def test_resize_then_normalize(self, sample_image):
        transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32

    def test_augmentation_pipeline(self, sample_image):
        transform = A.Compose(
            [
                A.Resize(48, 48),
                A.RandomCrop(32, 32),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32

    def test_geometric_pipeline(self, sample_image):
        transform = A.Compose(
            [
                A.Resize(64, 64),
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ]
        )
        result = transform(image=sample_image)["image"]
        # Both flips applied
        expected = np.flipud(np.fliplr(sample_image))
        np.testing.assert_array_equal(result, expected)

    def test_color_pipeline(self, sample_image):
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=1.0),
                A.ToFloat(),
            ]
        )
        result = transform(image=sample_image)["image"]
        assert result.dtype == np.float32
        assert result.shape == sample_image.shape


# --- TransformRegistry Tests ---


class TestTransformRegistry:
    """Test TransformRegistry with Albumentations."""

    def test_albumentations_registered(self):
        from bovi_core.ml.dataloaders.transforms import TransformRegistry

        # Verify key transforms are registered
        assert TransformRegistry.is_registered("Resize")
        assert TransformRegistry.is_registered("Normalize")
        assert TransformRegistry.is_registered("HorizontalFlip")
        assert TransformRegistry.is_registered("VerticalFlip")
        assert TransformRegistry.is_registered("RandomCrop")
        assert TransformRegistry.is_registered("CenterCrop")

    def test_create_resize(self, sample_image):
        from bovi_core.ml.dataloaders.transforms import TransformRegistry

        transform = TransformRegistry.create("Resize", height=32, width=32)
        result = transform(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)

    def test_create_normalize(self, sample_image):
        from bovi_core.ml.dataloaders.transforms import TransformRegistry

        transform = TransformRegistry.create("Normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        result = transform(image=sample_image)["image"]
        assert result.dtype == np.float32

    def test_create_horizontal_flip(self, sample_image):
        from bovi_core.ml.dataloaders.transforms import TransformRegistry

        transform = TransformRegistry.create("HorizontalFlip", p=1.0)
        result = transform(image=sample_image)["image"]
        np.testing.assert_array_equal(result, np.fliplr(sample_image))

    def test_build_vision_pipeline(self, sample_image):
        from bovi_core.ml.dataloaders.transforms import build_vision_pipeline

        config = [
            {"name": "Resize", "params": {"height": 32, "width": 32}},
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
        ]
        pipeline = build_vision_pipeline(config)
        result = pipeline(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)

    def test_build_vision_pipeline_with_normalize(self, sample_image):
        from bovi_core.ml.dataloaders.transforms import build_vision_pipeline

        config = [
            {"name": "Resize", "params": {"height": 32, "width": 32}},
            {
                "name": "Normalize",
                "params": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            },
        ]
        pipeline = build_vision_pipeline(config)
        result = pipeline(image=sample_image)["image"]
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32

    def test_list_transforms(self):
        from bovi_core.ml.dataloaders.transforms import TransformRegistry

        transforms = TransformRegistry.list_transforms()
        assert "Resize" in transforms
        assert "Normalize" in transforms
        assert len(transforms) >= 10  # At least 10 transforms registered


# --- Cross-Framework Consistency Tests ---


class TestCrossFrameworkConsistency:
    """Test same transform works identically in PyTorch and TensorFlow loaders."""

    def test_resize_same_output_pytorch_tensorflow(self, image_source, mock_dataloader_config):
        """Same resize transform produces same results across frameworks."""
        import torch
        from bovi_core.ml.dataloaders.datasets import ImageDataset
        from bovi_core.ml.dataloaders.loaders import (
            PyTorchDataLoader,
            TensorFlowDataLoader,
        )

        dataset = ImageDataset(image_source)
        transform = A.Compose([A.Resize(32, 32)])

        # Get first sample from PyTorch loader
        pt_loader = PyTorchDataLoader(
            dataset,
            config=mock_dataloader_config,
            model_name="test_model",
            split="train",
            batch_size=1,
            transform=transform,
            shuffle=False,
        )
        pt_batch = next(iter(pt_loader.get_pytorch_loader()))
        pt_image = pt_batch["image"]
        if isinstance(pt_image, torch.Tensor):
            pt_image = pt_image.numpy()

        # Get first sample from TensorFlow loader
        tf_loader = TensorFlowDataLoader(
            dataset,
            config=mock_dataloader_config,
            model_name="test_model",
            split="train",
            batch_size=1,
            transform=transform,
            shuffle=False,
        )
        tf_batch = next(iter(tf_loader.get_tensorflow_dataset()))
        tf_image = tf_batch["image"].numpy()

        # Compare (transpose PyTorch CHW -> HWC for comparison)
        # PyTorch: (B, C, H, W), TensorFlow: (B, H, W, C)
        pt_image_hwc = np.transpose(pt_image[0], (1, 2, 0))
        tf_image_hwc = tf_image[0]

        # Both should have same shape after transform
        assert pt_image_hwc.shape == (32, 32, 3)
        assert tf_image_hwc.shape == (32, 32, 3)

        # Values should be very close (may differ slightly due to float conversion)
        np.testing.assert_allclose(pt_image_hwc, tf_image_hwc, rtol=1e-4, atol=1e-4)

    def test_normalize_same_output_pytorch_tensorflow(self, image_source, mock_dataloader_config):
        """Normalize transform produces consistent results across frameworks."""
        import torch
        from bovi_core.ml.dataloaders.datasets import ImageDataset
        from bovi_core.ml.dataloaders.loaders import (
            PyTorchDataLoader,
            TensorFlowDataLoader,
        )

        dataset = ImageDataset(image_source)
        transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # PyTorch loader
        pt_loader = PyTorchDataLoader(
            dataset,
            config=mock_dataloader_config,
            model_name="test_model",
            split="train",
            batch_size=1,
            transform=transform,
            shuffle=False,
            auto_normalize=False,  # Don't double-normalize
        )
        pt_batch = next(iter(pt_loader.get_pytorch_loader()))
        pt_image = pt_batch["image"]
        if isinstance(pt_image, torch.Tensor):
            pt_image = pt_image.numpy()

        # TensorFlow loader
        tf_loader = TensorFlowDataLoader(
            dataset,
            config=mock_dataloader_config,
            model_name="test_model",
            split="train",
            batch_size=1,
            transform=transform,
            shuffle=False,
        )
        tf_batch = next(iter(tf_loader.get_tensorflow_dataset()))
        tf_image = tf_batch["image"].numpy()

        # Compare
        pt_image_hwc = np.transpose(pt_image[0], (1, 2, 0))
        tf_image_hwc = tf_image[0]

        # Both should be float32 after normalization
        assert pt_image_hwc.dtype == np.float32
        assert tf_image_hwc.dtype == np.float32

        # Values should be close
        np.testing.assert_allclose(pt_image_hwc, tf_image_hwc, rtol=1e-4, atol=1e-4)
