"""
Image dataset implementation.

Returns raw NumPy arrays - transforms are applied in DataLoaders.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..base import Dataset, DataSource

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Dataset for image classification/detection tasks.

    Loads images from a DataSource and returns raw NumPy arrays.
    Transforms are NOT applied here - they happen in DataLoaders.

    Args:
        source: DataSource to load images from.
        config: Optional config instance.
        return_metadata: Whether to include metadata in output (default: False).

    Returns:
        Dict with keys:
        - "image": NumPy array (H, W, C), uint8
        - "label": Label (if available in metadata)
        - "metadata": Additional metadata (if return_metadata=True)
    """

    # Type annotations for instance attributes
    return_metadata: bool

    def __init__(
        self,
        source: DataSource,
        config: Config | None = None,
        return_metadata: bool = False,
    ) -> None:
        super().__init__(source, config)
        self.return_metadata = return_metadata

        logger.info(f"ImageDataset: {len(self.source)} images, return_metadata={return_metadata}")

    def __len__(self) -> int:
        """Number of images in dataset."""
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Get image by index.

        Args:
            index: Image index.

        Returns:
            Dict with image (NumPy HWC uint8), label, and optionally metadata.
        """
        # Load raw image bytes
        image_bytes = self.source.load_item(index)

        # Decode image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get metadata (for label)
        metadata = self.source.get_metadata(index)

        # Convert to NumPy array (HWC, uint8)
        image_np = np.array(image)

        # Build output
        output: dict[str, object] = {
            "image": image_np,
            "label": metadata.get("label", None),
        }

        if self.return_metadata:
            output["metadata"] = metadata

        return output

    def get_label_counts(self) -> dict[str, int]:
        """
        Get counts of each label in dataset.

        Useful for analyzing class balance.

        Returns:
            Dict mapping label -> count.
        """
        label_counts: dict[str, int] = {}

        for idx in range(len(self)):
            metadata = self.source.get_metadata(idx)
            label = metadata.get("label", "unknown")
            # Ensure label is a string for dict key
            label_str = str(label) if label is not None else "unknown"

            if label_str in label_counts:
                label_counts[label_str] += 1
            else:
                label_counts[label_str] = 1

        return label_counts

    def get_sample(self, index: int = 0) -> dict[str, object]:
        """
        Get a sample item for inspection.

        Useful for debugging and verifying data format.

        Args:
            index: Index to sample (default: 0).

        Returns:
            Sample item dict.
        """
        return self[index]
