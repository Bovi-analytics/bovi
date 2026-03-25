"""Tests for LocalFileSource."""

import pytest
from pathlib import Path
from bovi_core.ml.dataloaders.sources.local_source import LocalFileSource


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create temporary directory with test images"""
    # Create structure:
    # tmp/
    #   ├── class_a/
    #   │   ├── img1.jpg
    #   │   └── img2.jpg
    #   └── class_b/
    #       └── img3.jpg

    class_a = tmp_path / "class_a"
    class_b = tmp_path / "class_b"
    class_a.mkdir()
    class_b.mkdir()

    (class_a / "img1.jpg").write_bytes(b"fake_image_1")
    (class_a / "img2.jpg").write_bytes(b"fake_image_2")
    (class_b / "img3.jpg").write_bytes(b"fake_image_3")

    return tmp_path


def test_local_source_finds_files(temp_image_dir):
    """Test that LocalFileSource finds all matching files"""
    source = LocalFileSource(temp_image_dir, file_pattern="*.jpg")
    assert len(source) == 3


def test_local_source_loads_file(temp_image_dir):
    """Test loading file content"""
    source = LocalFileSource(temp_image_dir, file_pattern="*.jpg")
    data = source.load_item(0)
    assert data.startswith(b"fake_image")


def test_local_source_metadata(temp_image_dir):
    """Test metadata extraction"""
    source = LocalFileSource(temp_image_dir, file_pattern="*.jpg")
    metadata = source.get_metadata(0)

    assert "path" in metadata
    assert "label" in metadata
    assert metadata["label"] in ["class_a", "class_b"]
    assert metadata["size_bytes"] > 0


def test_local_source_get_keys(temp_image_dir):
    """Test getting all keys"""
    source = LocalFileSource(temp_image_dir, file_pattern="*.jpg")
    keys = source.get_keys()
    assert keys == [0, 1, 2]


def test_local_source_non_recursive(tmp_path):
    """Test non-recursive search"""
    # Create nested structure
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "root.txt").write_bytes(b"root")
    (subdir / "nested.txt").write_bytes(b"nested")

    source = LocalFileSource(tmp_path, file_pattern="*.txt", recursive=False)
    assert len(source) == 1  # Should only find root.txt
