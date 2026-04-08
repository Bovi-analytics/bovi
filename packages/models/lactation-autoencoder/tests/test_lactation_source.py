"""Tests for LactationPKLSource."""

import json

import pytest

from lactation_autoencoder.dataloaders.sources.lactation_pkl_source import LactationPKLSource


@pytest.fixture
def json_data_dir(tmp_path):
    """Create mock JSON data directory with lactation records."""
    json_dir = tmp_path / "jsons"
    json_dir.mkdir()

    # Create mock lactation data
    lactation_1 = {
        "animal_id": "cow_001",
        "herd_id": 1001,
        "parity": 1,
        "milk": [20.0, 21.0, 22.0, 23.0, 24.0] + [20.0] * 299,  # 304 days
        "events": ["milking"] * 304,
    }

    lactation_2 = {
        "animal_id": "cow_002",
        "herd_id": 1001,
        "parity": 2,
        "milk": [25.0, 26.0, 27.0, 28.0, 29.0] + [25.0] * 299,
        "events": ["milking"] * 304,
    }

    lactation_3 = {
        "animal_id": "cow_003",
        "herd_id": 1002,
        "parity": 1,
        "milk": [30.0, 31.0, 32.0, 33.0, 34.0] + [30.0] * 299,
        "events": ["milking"] * 304,
    }

    # Write JSON files
    with open(json_dir / "animal_001.json", "w") as f:
        json.dump(lactation_1, f)

    with open(json_dir / "animal_002.json", "w") as f:
        json.dump(lactation_2, f)

    with open(json_dir / "animal_003.json", "w") as f:
        json.dump(lactation_3, f)

    return json_dir


class TestLactationPKLSourceInitialization:
    """Test LactationPKLSource initialization."""

    def test_source_initialization(self, json_data_dir):
        """Test source initializes correctly."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        assert len(source) == 3
        assert source.json_root_dir == json_data_dir

    def test_missing_json_directory(self):
        """Test error when JSON directory doesn't exist."""
        with pytest.raises(ValueError, match="JSON directory not found"):
            LactationPKLSource(json_root_dir="/nonexistent/path")


class TestLactationPKLSourceBasic:
    """Test basic LactationPKLSource functionality."""

    def test_source_length(self, json_data_dir):
        """Test source returns correct length."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        assert len(source) == 3

    def test_load_item_basic(self, json_data_dir):
        """Test load_item returns data."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert isinstance(item, dict)
        assert "animal_id" in item
        assert "herd_id" in item
        assert "parity" in item

    def test_load_item_has_milk_data(self, json_data_dir):
        """Test load_item includes milk data."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert "milk" in item
        assert len(item["milk"]) == 304

    def test_load_item_has_events(self, json_data_dir):
        """Test load_item includes events."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert "events" in item
        assert len(item["events"]) == 304

    def test_load_item_no_herd_stats(self, json_data_dir):
        """Test load_item does NOT include herd_stats (enrichment is a transform now)."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert "herd_stats" not in item

    def test_load_item_no_event_to_idx(self, json_data_dir):
        """Test load_item does NOT include event_to_idx (tokenization is a transform now)."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert "event_to_idx" not in item

    def test_iteration(self, json_data_dir):
        """Test iterating over source."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        items = list(source)
        assert len(items) == 3

        for item in items:
            assert "animal_id" in item
            assert "herd_stats" not in item


class TestLactationPKLSourceMemoryManagement:
    """Test memory management options."""

    def test_keep_in_memory_true(self, json_data_dir):
        """Test with keep_in_memory=True."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            keep_in_memory=True,
        )

        assert source.data_cache is not None

    def test_keep_in_memory_false(self, json_data_dir):
        """Test with keep_in_memory=False."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            keep_in_memory=False,
        )

        assert source.data_cache is None

    def test_consistent_access_with_and_without_cache(self, json_data_dir):
        """Test that cached and non-cached access return same data."""
        source_cached = LactationPKLSource(
            json_root_dir=json_data_dir,
            keep_in_memory=True,
        )

        source_uncached = LactationPKLSource(
            json_root_dir=json_data_dir,
            keep_in_memory=False,
        )

        item_cached = source_cached.load_item(0)
        item_uncached = source_uncached.load_item(0)

        assert item_cached["animal_id"] == item_uncached["animal_id"]
        assert item_cached["milk"] == item_uncached["milk"]


class TestLactationPKLSourceLoadItem:
    """Test load_item behavior."""

    def test_first_item(self, json_data_dir):
        """Test accessing first item."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(0)
        assert item["animal_id"] == "cow_001"

    def test_last_item(self, json_data_dir):
        """Test accessing last item via index calculation."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        item = source.load_item(len(source) - 1)
        assert item["animal_id"] == "cow_003"

    def test_out_of_bounds(self, json_data_dir):
        """Test out of bounds indexing."""
        source = LactationPKLSource(json_root_dir=json_data_dir)

        with pytest.raises((IndexError, KeyError)):
            source.load_item(100)
