"""Tests for YOLO image source factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from bovi_core.ml.dataloaders.sources import LocalFileSource


class TestYOLOImageSourceFromLocal:
    def test_creates_local_file_source(self, temp_image_dir: Path) -> None:
        """Test from_local creates a LocalFileSource."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images"
        )
        assert isinstance(source, LocalFileSource)

    def test_source_length_matches_images(self, temp_image_dir: Path) -> None:
        """Test source length matches number of images."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images"
        )
        assert len(source) == 1

    def test_load_item_returns_bytes(self, temp_image_dir: Path) -> None:
        """Test load_item returns image bytes."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images"
        )
        item = source.load_item(0)
        assert item is not None

    def test_get_metadata_returns_expected_keys(
        self, temp_image_dir: Path
    ) -> None:
        """Test get_metadata returns path and filename."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images"
        )
        metadata = source.get_metadata(0)
        assert "path" in metadata
        assert "filename" in metadata

    def test_file_pattern_filtering(self, temp_image_dir: Path) -> None:
        """Test file pattern correctly filters files."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        # Only .jpeg files
        source = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images", file_pattern="*.jpeg"
        )
        assert len(source) == 1

        # Non-matching pattern
        source_empty = YOLOImageSource.from_local(
            temp_image_dir / "train" / "images", file_pattern="*.png"
        )
        assert len(source_empty) == 0

    def test_custom_file_pattern(self, temp_image_dir: Path) -> None:
        """Test custom file pattern works."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_local(
            temp_image_dir / "test" / "images", file_pattern="*.jpg"
        )
        assert len(source) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test handling of empty directory."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        source = YOLOImageSource.from_local(empty_dir)
        assert len(source) == 0


class TestYOLOImageSourceFromConfig:
    def test_unsupported_source_type_raises(self) -> None:
        """Test that unsupported source type raises ValueError."""
        from unittest.mock import MagicMock

        from bovi_yolo.dataloaders.sources import YOLOImageSource

        config = MagicMock()
        config.experiment.dataloaders.train.source.type = "s3"

        with pytest.raises(ValueError, match="Unsupported source type"):
            YOLOImageSource.from_config(config, split="train")
