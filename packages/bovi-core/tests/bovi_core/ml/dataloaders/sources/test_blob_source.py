"""Tests for BlobImageSource."""

from unittest.mock import Mock, patch

import pytest
from bovi_core.ml.dataloaders.sources.blob_source import BlobImageSource


@pytest.fixture
def mock_config():
    """Mock Config with blob utilities"""
    config = Mock()
    return config


@patch("bovi_core.ml.dataloaders.sources.blob_source.blob_utils.list_blobs_by_pattern")
def test_blob_source_lists_blobs(mock_list, mock_config):
    """Test that BlobImageSource lists blobs using blob_utils"""
    mock_list.return_value = ["images/train/cat/cat1.jpg", "images/train/dog/dog1.jpg"]

    source = BlobImageSource(config=mock_config, prefix="images/train/", substring=".jpg")

    assert len(source) == 2
    mock_list.assert_called_once_with(
        dir_path="images/train/", substring=".jpg", config=mock_config
    )


@patch("bovi_core.ml.dataloaders.sources.blob_source.blob_utils.list_blobs_by_pattern")
@patch("bovi_core.ml.dataloaders.sources.blob_source.blob_utils.get_file_blob")
def test_blob_source_loads_image(mock_get_file, mock_list, mock_config):
    """Test loading image using blob_utils"""
    mock_list.return_value = ["images/cat1.jpg"]
    mock_get_file.return_value = b"fake_image_data"

    source = BlobImageSource(config=mock_config, prefix="images/")
    data = source.load_item(0)

    assert data == b"fake_image_data"
    mock_get_file.assert_called_once_with("images/cat1.jpg", config=mock_config)


@patch("bovi_core.ml.dataloaders.sources.blob_source.blob_utils.list_blobs_by_pattern")
def test_blob_source_metadata(mock_list, mock_config):
    """Test metadata extraction"""
    mock_list.return_value = ["images/train/cat/cat1.jpg", "images/train/dog/dog1.jpg"]

    source = BlobImageSource(config=mock_config, prefix="images/train/")
    metadata = source.get_metadata(0)

    assert metadata["label"] in ["cat", "dog"]
    assert "path" in metadata
    assert "filename" in metadata


@patch("bovi_core.ml.dataloaders.sources.blob_source.blob_utils.list_blobs_by_pattern")
def test_blob_source_get_keys(mock_list, mock_config):
    """Test getting all keys"""
    mock_list.return_value = ["img1.jpg", "img2.jpg"]

    source = BlobImageSource(config=mock_config, prefix="images/")
    keys = source.get_keys()

    assert keys == [0, 1]
