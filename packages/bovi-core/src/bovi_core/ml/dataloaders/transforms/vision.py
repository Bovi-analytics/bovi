"""Explicit sample-level vision transforms; no framework dependencies.

Apply these with TransformedDataset, after image decoding. Neither loaders
nor collators choose normalization, augmentation or channel order for you.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from bovi_core.ml.dataloaders.transforms.base_transform import UniversalTransform

from .registry import TransformRegistry


class AlbumentationsTransform:
    """Call an Albumentations pipeline with explicitly selected sample fields.

    Include masks, bboxes, keypoints and label fields when augmenting those
    targets, and configure the pipeline accordingly. For video, configure
    native multi-image targets; this wrapper never augments frames separately.
    Returned fields replace the corresponding fields; other metadata is kept.
    """

    def __init__(
        self, pipeline: Callable[..., dict[str, Any]], fields: Sequence[str] = ("image",)
    ) -> None:
        self.pipeline = pipeline
        self.fields = tuple(fields)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        targets = {field: sample[field] for field in self.fields}
        return {**sample, **self.pipeline(**targets)}


@TransformRegistry.register("image_preprocessing")
class ImagePreprocessing(UniversalTransform):
    """Explicit uint8 scaling and/or channel-last to channel-first conversion.

    Each configured field must be an HWC image or THWC video. Normalization
    converts uint8 to float32 / 255; floating arrays are left unchanged. Layout
    conversion is explicit, not guessed from the size of a dimension. Apply
    once, after spatial augmentations. Missing fields are configuration errors.
    """

    def __init__(
        self,
        fields: Sequence[str] = ("image",),
        normalize: bool = False,
        channels_first: bool = False,
    ) -> None:
        self.fields = tuple(fields)
        self.normalize = normalize
        self.channels_first = channels_first

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        for field in self.fields:
            image = data[field]
            if not isinstance(image, np.ndarray) or image.ndim not in (3, 4):
                raise ValueError(f"Image field {field!r} must be an HWC or THWC NumPy array")
            if self.normalize and image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if self.channels_first:
                image = np.moveaxis(image, -1, -3)
            result[field] = image
        return result

    def get_params(self) -> dict[str, object]:
        return {
            "fields": list(self.fields),
            "normalize": self.normalize,
            "channels_first": self.channels_first,
        }
