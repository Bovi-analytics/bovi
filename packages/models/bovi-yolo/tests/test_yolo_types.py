"""Tests for YOLO type definitions."""

from __future__ import annotations

import numpy as np


class TestYOLOMetadata:
    def test_metadata_structure(self) -> None:
        """Test YOLOMetadata TypedDict can be created."""
        from bovi_yolo.types import YOLOMetadata

        metadata: YOLOMetadata = {
            "path": "/path/to/image.jpg",
            "filename": "image.jpg",
            "label": "cow",
            "source_type": "local",
        }
        assert metadata["path"] == "/path/to/image.jpg"
        assert metadata["filename"] == "image.jpg"
        assert metadata["source_type"] == "local"


class TestYOLOItem:
    def test_item_structure(self) -> None:
        """Test YOLOItem TypedDict can be created."""
        from bovi_yolo.types import YOLOItem, YOLOMetadata

        metadata: YOLOMetadata = {
            "path": "/path/to/image.jpg",
            "filename": "image.jpg",
            "label": "cow",
            "source_type": "local",
        }
        item: YOLOItem = {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
            "label": "cow",
            "metadata": metadata,
        }
        assert item["image"].shape == (480, 640, 3)
        assert item["label"] == "cow"
        assert item["metadata"]["filename"] == "image.jpg"

    def test_item_with_none_label(self) -> None:
        """Test YOLOItem accepts None label."""
        from bovi_yolo.types import YOLOItem, YOLOMetadata

        metadata: YOLOMetadata = {
            "path": "/path/to/image.jpg",
            "filename": "image.jpg",
            "label": "",
            "source_type": "local",
        }
        item: YOLOItem = {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
            "label": None,
            "metadata": metadata,
        }
        assert item["label"] is None


class TestYOLOInput:
    def test_input_with_image(self) -> None:
        """Test YOLOInput with single image."""
        from bovi_yolo.types import YOLOInput

        input_data: YOLOInput = {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        assert input_data["image"].shape == (480, 640, 3)

    def test_input_with_images(self) -> None:
        """Test YOLOInput with batch of images."""
        from bovi_yolo.types import YOLOInput

        input_data: YOLOInput = {
            "images": [
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros((320, 480, 3), dtype=np.uint8),
            ],
        }
        assert len(input_data["images"]) == 2

    def test_input_with_paths(self) -> None:
        """Test YOLOInput with file paths."""
        from bovi_yolo.types import YOLOInput

        input_data: YOLOInput = {
            "paths": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
        }
        assert len(input_data["paths"]) == 2
