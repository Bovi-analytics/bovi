"""Type definitions for YOLO cow detection pipeline.

Defines TypedDicts used across the YOLO model, dataset, source, and predictor.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt


class YOLOMetadata(TypedDict):
    """Metadata for a YOLO dataset item.

    Attributes:
        path: Full file path to the image.
        filename: Image filename (e.g., "cow_001.jpg").
        label: Parent directory name (used as label for classification).
        source_type: Data source type ("local" or "blob").
    """

    path: str
    filename: str
    label: str
    source_type: str


class YOLOItem(TypedDict):
    """Output from YOLODataset.__getitem__().

    Attributes:
        image: RGB image as NumPy array (H, W, 3), uint8.
        label: Class label from parent directory, or None.
        metadata: Image metadata dict.
    """

    image: npt.NDArray[np.uint8]
    label: str | None
    metadata: YOLOMetadata


class YOLOInput(TypedDict, total=False):
    """Input to YOLOPredictor.predict().

    All fields are optional (total=False). Provide one of:
        image: Single image as NumPy array (H, W, 3), uint8.
        images: Batch of images as list of NumPy arrays.
        paths: List of image file paths.
    """

    image: npt.NDArray[np.uint8]
    images: list[npt.NDArray[np.uint8]]
    paths: list[str]
