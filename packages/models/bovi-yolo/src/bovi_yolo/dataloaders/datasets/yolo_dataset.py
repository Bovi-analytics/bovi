"""YOLO dataset for cow detection inference.

Thin wrapper around ImageDataset with metadata always enabled
and YOLO-specific helper methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from bovi_core.ml.dataloaders.datasets import ImageDataset
from typing_extensions import override

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import DataSource

logger = logging.getLogger(__name__)


class YOLODataset(ImageDataset):
    """Dataset for YOLO cow detection inference.

    Extends ImageDataset with:
    - Metadata always enabled (return_metadata=True)
    - Split-aware label extraction (uses grandparent dir for split name)
    - YOLO-specific helper methods for image paths and sizes

    YOLO directory layout is ``input/{split}/images/{file}``, so the
    parent directory is always "images".  This dataset extracts the
    split name (grandparent) as the label instead.

    Example:
        >>> source = LocalFileSource("data/images", "*.jpg")
        >>> dataset = YOLODataset(source, config)
        >>> item = dataset[0]
        >>> item["image"].shape  # (H, W, 3)
        >>> item["metadata"]["path"]  # "/path/to/image.jpg"
        >>> item["label"]  # "train", "val", or "test"
    """

    def __init__(
        self,
        source: DataSource[bytes],
        config: Config | None = None,
    ) -> None:
        """Initialize YOLO dataset.

        Args:
            source: DataSource to load images from.
            config: Optional config instance.
        """
        super().__init__(source, config, return_metadata=True)
        logger.info("YOLODataset: %d images", len(self.source))

    @override
    def __getitem__(self, index: int) -> dict[str, object]:
        """Get image by index with metadata always included.

        For YOLO's ``{split}/images/{file}`` layout, the label is set
        to the split directory name instead of the immediate parent.

        Args:
            index: Image index.

        Returns:
            Dict with image (NumPy HWC uint8), label (split name), and metadata.
        """
        item = super().__getitem__(index)

        metadata = self.source.get_metadata(index)
        path = Path(str(metadata.get("path", "")))
        # YOLO layout: .../split/images/file.jpg → use grandparent as label
        if path.parent.name == "images":
            item["label"] = path.parent.parent.name

        # Add image dimensions to metadata from the already-loaded array
        image = item["image"]
        if isinstance(image, np.ndarray) and isinstance(item.get("metadata"), dict):
            item["metadata"]["height"] = image.shape[0]
            item["metadata"]["width"] = image.shape[1]

        return item

    def get_image_paths(self) -> list[str]:
        """Get all image file paths in dataset.

        Returns:
            List of absolute file paths for all images.
        """
        paths: list[str] = []
        for idx in range(len(self)):
            metadata = self.source.get_metadata(idx)
            paths.append(str(metadata.get("path", "")))
        return paths

    def get_image_sizes(self) -> list[tuple[int, int]]:
        """Get (height, width) for all images.

        Note: This loads each image to determine size. For large datasets,
        consider caching results.

        Returns:
            List of (height, width) tuples.
        """
        sizes: list[tuple[int, int]] = []
        for idx in range(len(self)):
            item = self[idx]
            image = item["image"]
            if isinstance(image, np.ndarray):
                sizes.append((image.shape[0], image.shape[1]))
        return sizes
