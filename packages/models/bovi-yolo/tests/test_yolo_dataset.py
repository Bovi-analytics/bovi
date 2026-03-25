"""Tests for YOLO dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bovi_core.ml.dataloaders.sources import LocalFileSource


class TestYOLODataset:
    def test_dataset_length(self, temp_image_dir: Path) -> None:
        """Test dataset length matches source."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        assert len(dataset) == 1

    def test_getitem_returns_dict(self, temp_image_dir: Path) -> None:
        """Test __getitem__ returns dict with expected keys."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]

        assert isinstance(item, dict)
        assert "image" in item
        assert "metadata" in item

    def test_image_is_numpy_uint8(self, temp_image_dir: Path) -> None:
        """Test image is numpy array with correct dtype."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]

        image = item["image"]
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3

    def test_image_shape_hwc(self, temp_image_dir: Path) -> None:
        """Test image shape is (H, W, C)."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]
        h, w, c = item["image"].shape
        assert h > 0
        assert w > 0
        assert c == 3

    def test_metadata_always_present(self, temp_image_dir: Path) -> None:
        """Test metadata is always returned (return_metadata=True)."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]

        assert "metadata" in item
        metadata = item["metadata"]
        assert "path" in metadata
        assert "filename" in metadata

    def test_get_image_paths(self, temp_image_dir: Path) -> None:
        """Test get_image_paths returns list of paths."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        paths = dataset.get_image_paths()

        assert len(paths) == 1
        assert isinstance(paths[0], str)
        assert paths[0].endswith(".jpeg")

    def test_get_image_sizes(self, temp_image_dir: Path) -> None:
        """Test get_image_sizes returns (height, width) tuples."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        sizes = dataset.get_image_sizes()

        assert len(sizes) == 1
        h, w = sizes[0]
        assert h > 0
        assert w > 0

    def test_label_is_split_name(self, temp_image_dir: Path) -> None:
        """Test label uses split directory name, not 'images'."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        for split in ("train", "val", "test"):
            pattern = "*.jpeg" if split == "train" else "*.jpg"
            source = LocalFileSource(
                root_dir=temp_image_dir / split / "images",
                file_pattern=pattern,
            )
            dataset = YOLODataset(source=source)
            item = dataset[0]
            assert item["label"] == split, f"Expected label '{split}', got '{item['label']}'"

    def test_modified_is_iso_timestamp(self, temp_image_dir: Path) -> None:
        """Test modified field is an ISO timestamp string, not a raw float."""
        from datetime import datetime

        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]
        metadata = item["metadata"]

        modified = metadata["modified"]
        assert isinstance(modified, str), f"Expected str, got {type(modified)}"
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(modified)
        assert parsed.year >= 2024

    def test_metadata_contains_image_dimensions(self, temp_image_dir: Path) -> None:
        """Test metadata includes height and width from the loaded image."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]
        metadata = item["metadata"]

        assert "height" in metadata
        assert "width" in metadata
        assert metadata["height"] == 480
        assert metadata["width"] == 640

    def test_iteration(self, temp_image_dir: Path) -> None:
        """Test dataset can be iterated."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "test" / "images",
            file_pattern="*.jpg",
        )
        dataset = YOLODataset(source=source)
        items = [dataset[i] for i in range(len(dataset))]
        assert len(items) == 1
