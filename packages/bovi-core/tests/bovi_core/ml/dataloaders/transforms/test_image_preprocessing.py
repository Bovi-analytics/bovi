"""Vision preprocessing is explicit and independent of native frameworks."""

import pickle

import numpy as np
import pytest
from bovi_core.ml.dataloaders.datasets import TransformedDataset
from bovi_core.ml.dataloaders.transforms import (
    AlbumentationsTransform,
    ImagePreprocessing,
    TransformRegistry,
)


@pytest.mark.parametrize("shape", [(4, 5, 3), (2, 4, 5, 3)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("channels_first", [False, True])
def test_image_and_video_processing_is_explicit(shape, normalize, channels_first):
    pixels = np.full(shape, 255, dtype=np.uint8)
    sample = {"pixels": pixels, "metadata": {"id": 1}, "labels": 2}
    transform = ImagePreprocessing(
        fields=("pixels",), normalize=normalize, channels_first=channels_first
    )
    result = transform(sample)
    expected = pixels.astype(np.float32) / 255 if normalize else pixels
    if channels_first:
        expected = np.moveaxis(expected, -1, -3)
    np.testing.assert_array_equal(result["pixels"], expected)
    assert result["pixels"].dtype == (np.float32 if normalize else np.uint8)
    assert result["metadata"] is sample["metadata"]
    assert result["labels"] == 2
    assert sample["pixels"] is pixels
    assert np.all(pixels == 255)


def test_float_images_are_not_normalized_again():
    pixels = np.full((2, 3, 4), -0.5, dtype=np.float32)
    result = ImagePreprocessing(normalize=True)({"image": pixels})
    np.testing.assert_array_equal(result["image"], pixels)


def test_missing_or_invalid_image_field_fails():
    with pytest.raises(KeyError, match="image"):
        ImagePreprocessing()({"features": [1]})
    with pytest.raises(ValueError, match="HWC or THWC"):
        ImagePreprocessing()({"image": np.ones(3)})


def test_preprocessing_can_be_reconstructed_from_params():
    transform = ImagePreprocessing(normalize=True, channels_first=True)
    restored = TransformRegistry.get("image_preprocessing")(**transform.get_params())
    assert isinstance(restored, ImagePreprocessing)
    assert restored.get_params() == transform.get_params()


def test_augmentation_receives_all_explicit_targets_and_preserves_metadata():
    def flip(**targets):
        assert set(targets) == {"image", "mask"}
        return {key: np.flip(value, axis=1) for key, value in targets.items()}

    sample = {
        "image": np.arange(12).reshape(2, 2, 3),
        "mask": np.arange(4).reshape(2, 2),
        "metadata": {"id": "original"},
    }
    result = AlbumentationsTransform(flip, fields=("image", "mask"))(sample)
    np.testing.assert_array_equal(result["image"], sample["image"][:, ::-1])
    np.testing.assert_array_equal(result["mask"], sample["mask"][:, ::-1])
    assert result["metadata"] is sample["metadata"]


def test_transformed_dataset_is_lazy_repeatable_and_picklable(image_source):
    from bovi_core.ml.dataloaders.datasets import ImageDataset

    raw = ImageDataset(image_source)
    dataset = TransformedDataset(raw, [ImagePreprocessing(normalize=True, channels_first=True)])
    original = raw[0]["image"]
    expected = np.moveaxis(original.astype(np.float32) / 255, -1, -3)
    assert len(dataset) == len(raw)
    assert dataset.metadata == raw.metadata
    np.testing.assert_array_equal(dataset[0]["image"], expected)
    np.testing.assert_array_equal(dataset[0]["image"], expected)
    np.testing.assert_array_equal(raw[0]["image"], original)
    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored[0]["image"], expected)
