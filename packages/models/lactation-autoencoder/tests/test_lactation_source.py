"""Tests for LactationPKLSource."""

import json
import pickle

import numpy as np
import pytest

from lactation_autoencoder.dataloaders.sources.lactation_pkl_source import LactationPKLSource


@pytest.fixture
def herd_stats_dir(tmp_path):
    """Create mock herd statistics directory with pickle files."""
    stats_dir = tmp_path / "herd_stats"
    stats_dir.mkdir()

    # Create mock event mapping
    event_to_idx = {
        "milking": 0,
        "vaccination": 1,
        "treatment": 2,
        "heat": 3,
    }
    with open(stats_dir / "event_to_idx_dict.pkl", "wb") as f:
        pickle.dump(event_to_idx, f)

    # Create mock herd parameter index
    idx_to_herd_par = {i: f"param_{i}" for i in range(10)}
    with open(stats_dir / "idx_to_herd_par_dict.pkl", "wb") as f:
        pickle.dump(idx_to_herd_par, f)

    # Create herd stats per parity (Level 1: herd + parity)
    # CRITICAL: Parity MUST be strings ('1', '2', etc.) - not integers!
    # Structure: {herd_id: {parity_str: {stat_name: float}}}
    herd_stats_per_parity = {
        1001: {
            "1": {
                "param_0": 1.0,
                "param_1": 2.0,
                "param_2": 3.0,
                "param_3": 4.0,
                "param_4": 5.0,
                "param_5": 6.0,
                "param_6": 7.0,
                "param_7": 8.0,
                "param_8": 9.0,
                "param_9": 10.0,
            },
            "2": {
                "param_0": 1.5,
                "param_1": 2.5,
                "param_2": 3.5,
                "param_3": 4.5,
                "param_4": 5.5,
                "param_5": 6.5,
                "param_6": 7.5,
                "param_7": 8.5,
                "param_8": 9.5,
                "param_9": 10.5,
            },
        },
        1002: {
            "1": {
                "param_0": 2.0,
                "param_1": 3.0,
                "param_2": 4.0,
                "param_3": 5.0,
                "param_4": 6.0,
                "param_5": 7.0,
                "param_6": 8.0,
                "param_7": 9.0,
                "param_8": 10.0,
                "param_9": 11.0,
            },
        },
    }
    with open(stats_dir / "herd_stats_per_parity_dict.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity, f)

    # Create herd stats means per herd (Level 2: herd only)
    # Structure: {herd_id: {stat_name: float}}
    herd_stats_per_herd = {
        1001: {
            "param_0": 1.25,
            "param_1": 2.25,
            "param_2": 3.25,
            "param_3": 4.25,
            "param_4": 5.25,
            "param_5": 6.25,
            "param_6": 7.25,
            "param_7": 8.25,
            "param_8": 9.25,
            "param_9": 10.25,
        },
        1002: {
            "param_0": 2.0,
            "param_1": 3.0,
            "param_2": 4.0,
            "param_3": 5.0,
            "param_4": 6.0,
            "param_5": 7.0,
            "param_6": 8.0,
            "param_7": 9.0,
            "param_8": 10.0,
            "param_9": 11.0,
        },
    }
    with open(stats_dir / "herd_stats_means_per_herd.pkl", "wb") as f:
        pickle.dump(herd_stats_per_herd, f)

    # Create herd stats means per parity (Level 3: parity only)
    # CRITICAL: Parity MUST be strings ('1', '2', etc.)
    # Structure: {parity_str: {stat_name: float}}
    herd_stats_per_parity_global = {
        "1": {
            "param_0": 1.5,
            "param_1": 2.5,
            "param_2": 3.5,
            "param_3": 4.5,
            "param_4": 5.5,
            "param_5": 6.5,
            "param_6": 7.5,
            "param_7": 8.5,
            "param_8": 9.5,
            "param_9": 10.5,
        },
        "2": {
            "param_0": 2.0,
            "param_1": 3.0,
            "param_2": 4.0,
            "param_3": 5.0,
            "param_4": 6.0,
            "param_5": 7.0,
            "param_6": 8.0,
            "param_7": 9.0,
            "param_8": 10.0,
            "param_9": 11.0,
        },
    }
    with open(stats_dir / "herd_stat_means_per_parity.pkl", "wb") as f:
        pickle.dump(herd_stats_per_parity_global, f)

    # Create global herd stats means (Level 4: global fallback)
    # Structure: {stat_name: float}
    herd_stat_means_global = {
        "param_0": 1.75,
        "param_1": 2.75,
        "param_2": 3.75,
        "param_3": 4.75,
        "param_4": 5.75,
        "param_5": 6.75,
        "param_6": 7.75,
        "param_7": 8.75,
        "param_8": 9.75,
        "param_9": 10.75,
    }
    with open(stats_dir / "herd_stat_means_global.pkl", "wb") as f:
        pickle.dump(herd_stat_means_global, f)

    return stats_dir


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

    def test_source_initialization(self, json_data_dir, herd_stats_dir):
        """Test source initializes correctly."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        assert len(source) == 3
        assert source.json_root_dir == json_data_dir
        assert source.herd_stats_dir == herd_stats_dir

    def test_missing_json_directory(self, herd_stats_dir):
        """Test error when JSON directory doesn't exist."""
        with pytest.raises(ValueError, match="JSON directory not found"):
            LactationPKLSource(
                json_root_dir="/nonexistent/path",
                herd_stats_dir=herd_stats_dir,
            )

    def test_missing_herd_stats_directory(self, json_data_dir):
        """Test error when herd stats directory doesn't exist."""
        with pytest.raises(ValueError, match="Herd stats directory not found"):
            LactationPKLSource(
                json_root_dir=json_data_dir,
                herd_stats_dir="/nonexistent/path",
            )

    def test_herd_stats_loaded(self, json_data_dir, herd_stats_dir):
        """Test herd statistics are loaded correctly."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        assert source.event_to_idx is not None
        assert "milking" in source.event_to_idx
        assert source.herd_stats_per_parity is not None
        assert source.herd_stats_per_herd is not None


class TestLactationPKLSourceBasic:
    """Test basic LactationPKLSource functionality."""

    def test_source_length(self, json_data_dir, herd_stats_dir):
        """Test source returns correct length."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        assert len(source) == 3

    def test_load_item_basic(self, json_data_dir, herd_stats_dir):
        """Test load_item returns data."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(0)
        assert isinstance(item, dict)
        assert "animal_id" in item
        assert "herd_id" in item
        assert "parity" in item

    def test_load_item_has_milk_data(self, json_data_dir, herd_stats_dir):
        """Test load_item includes milk data."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(0)
        assert "milk" in item
        assert len(item["milk"]) == 304

    def test_load_item_has_events(self, json_data_dir, herd_stats_dir):
        """Test load_item includes events."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(0)
        assert "events" in item
        assert len(item["events"]) == 304

    def test_load_item_has_herd_stats(self, json_data_dir, herd_stats_dir):
        """Test load_item includes herd statistics."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(0)
        assert "herd_stats" in item
        assert len(item["herd_stats"]) == 10


class TestLactationPKLSourceHierarchicalFallback:
    """Test hierarchical fallback for herd statistics."""

    def test_level_1_herd_and_parity(self, json_data_dir, herd_stats_dir):
        """Test Level 1 fallback: herd_id + parity (most specific)."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        # Get item with herd_id=1001, parity=1
        # This should match Level 1: {1001: {'1': {...}}}
        item = source.load_item(0)
        assert item["herd_id"] == 1001
        assert item["parity"] == 1

        # Expected stats in order of idx_to_herd_par (param_0 to param_9)
        expected_stats = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32
        )
        assert np.allclose(item["herd_stats"], expected_stats)

    def test_level_1_second_parity(self, json_data_dir, herd_stats_dir):
        """Test Level 1 with parity=2."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        # Get item with herd_id=1001, parity=2
        item = source.load_item(1)  # cow_002 has parity=2
        assert item["herd_id"] == 1001
        assert item["parity"] == 2

        # Expected stats for (1001, '2')
        expected_stats = np.array(
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], dtype=np.float32
        )
        assert np.allclose(item["herd_stats"], expected_stats)

    def test_level_1_different_herd(self, json_data_dir, herd_stats_dir):
        """Test Level 1 with different herd."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        # Get item with herd_id=1002, parity=1
        item = source.load_item(2)  # cow_003 has herd_id=1002
        assert item["herd_id"] == 1002
        assert item["parity"] == 1

        # Expected stats for (1002, '1')
        expected_stats = np.array(
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32
        )
        assert np.allclose(item["herd_stats"], expected_stats)

    def test_parity_conversion_to_string(self, json_data_dir, herd_stats_dir):
        """Test that parity is correctly converted to string for lookup."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        # All items should find their stats even though parity is int in JSON
        for i in range(len(source)):
            item = source.load_item(i)
            # Should not raise any exception
            assert "herd_stats" in item
            assert len(item["herd_stats"]) == 10
            assert item["herd_stats"].dtype == np.float32

    def test_iteration(self, json_data_dir, herd_stats_dir):
        """Test iterating over source."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        items = list(source)
        assert len(items) == 3

        for item in items:
            assert "animal_id" in item
            assert "herd_stats" in item
            assert len(item["herd_stats"]) == 10


class TestLactationPKLSourceFallbackLevels:
    """Test all 4 hierarchical fallback levels in detail."""

    @pytest.fixture
    def fallback_test_dir(self, tmp_path):
        """Create directory with test data for all fallback levels."""
        stats_dir = tmp_path / "fallback_stats"
        stats_dir.mkdir()

        # Create event mapping
        event_to_idx = {"milking": 0, "unknown": 1}
        with open(stats_dir / "event_to_idx_dict.pkl", "wb") as f:
            pickle.dump(event_to_idx, f)

        # Create idx to herd parameters
        idx_to_herd_par = {i: f"param_{i}" for i in range(10)}
        with open(stats_dir / "idx_to_herd_par_dict.pkl", "wb") as f:
            pickle.dump(idx_to_herd_par, f)

        # Level 1: Most specific (herd + parity)
        herd_stats_per_parity = {
            1001: {"1": {f"param_{i}": float(i + 1) for i in range(10)}},  # herd 1001, parity 1
        }
        with open(stats_dir / "herd_stats_per_parity_dict.pkl", "wb") as f:
            pickle.dump(herd_stats_per_parity, f)

        # Level 2: Herd only (for cases where parity not in Level 1)
        herd_stats_per_herd = {
            1001: {f"param_{i}": float(i + 10) for i in range(10)},  # herd 1001 average
            1002: {f"param_{i}": float(i + 20) for i in range(10)},  # herd 1002 average
        }
        with open(stats_dir / "herd_stats_means_per_herd.pkl", "wb") as f:
            pickle.dump(herd_stats_per_herd, f)

        # Level 3: Parity only (for cases where herd not in Level 2)
        herd_stats_per_parity_global = {
            "1": {f"param_{i}": float(i + 100) for i in range(10)},  # parity 1 average
            "2": {f"param_{i}": float(i + 110) for i in range(10)},  # parity 2 average
        }
        with open(stats_dir / "herd_stat_means_per_parity.pkl", "wb") as f:
            pickle.dump(herd_stats_per_parity_global, f)

        # Level 4: Global fallback
        herd_stat_means_global = {f"param_{i}": float(i + 200) for i in range(10)}
        with open(stats_dir / "herd_stat_means_global.pkl", "wb") as f:
            pickle.dump(herd_stat_means_global, f)

        return stats_dir

    def test_fallback_level_1_exact_match(self, tmp_path, fallback_test_dir):
        """Test Level 1: herd_id + parity exact match."""
        json_dir = tmp_path / "jsons"
        json_dir.mkdir()

        # Create lactation with herd_id=1001, parity=1 (exact match in Level 1)
        lactation = {
            "animal_id": "test_cow_1",
            "herd_id": 1001,
            "parity": 1,
            "milk": [20.0] * 304,
            "events": ["milking"] * 304,
        }
        with open(json_dir / "animal_001.json", "w") as f:
            json.dump(lactation, f)

        source = LactationPKLSource(
            json_root_dir=json_dir,
            herd_stats_dir=fallback_test_dir,
        )
        item = source.load_item(0)

        # Should use Level 1: herd 1001, parity '1'
        expected = np.array([float(i + 1) for i in range(10)], dtype=np.float32)
        assert np.allclose(item["herd_stats"], expected)

    def test_fallback_level_2_herd_only(self, tmp_path, fallback_test_dir):
        """Test Level 2 fallback: herd_id only (parity not in Level 1)."""
        json_dir = tmp_path / "jsons"
        json_dir.mkdir()

        # Create lactation with herd_id=1002, parity=1 (not in Level 1, should use Level 2)
        lactation = {
            "animal_id": "test_cow_2",
            "herd_id": 1002,
            "parity": 1,
            "milk": [20.0] * 304,
            "events": ["milking"] * 304,
        }
        with open(json_dir / "animal_002.json", "w") as f:
            json.dump(lactation, f)

        source = LactationPKLSource(
            json_root_dir=json_dir,
            herd_stats_dir=fallback_test_dir,
        )
        item = source.load_item(0)

        # Should use Level 2: herd 1002 (no parity-specific data)
        expected = np.array([float(i + 20) for i in range(10)], dtype=np.float32)
        assert np.allclose(item["herd_stats"], expected)

    def test_fallback_level_3_parity_only(self, tmp_path, fallback_test_dir):
        """Test Level 3 fallback: parity only (herd not in Level 2)."""
        json_dir = tmp_path / "jsons"
        json_dir.mkdir()

        # Create lactation with herd_id=9999 (not in Level 2), parity=2 (in Level 3)
        lactation = {
            "animal_id": "test_cow_3",
            "herd_id": 9999,
            "parity": 2,
            "milk": [20.0] * 304,
            "events": ["milking"] * 304,
        }
        with open(json_dir / "animal_003.json", "w") as f:
            json.dump(lactation, f)

        source = LactationPKLSource(
            json_root_dir=json_dir,
            herd_stats_dir=fallback_test_dir,
        )
        item = source.load_item(0)

        # Should use Level 3: parity '2'
        expected = np.array([float(i + 110) for i in range(10)], dtype=np.float32)
        assert np.allclose(item["herd_stats"], expected)

    def test_fallback_level_4_global(self, tmp_path, fallback_test_dir):
        """Test Level 4 fallback: global average (unknown herd and parity)."""
        json_dir = tmp_path / "jsons"
        json_dir.mkdir()

        # Create lactation with unknown herd_id (9999) and unknown parity (99)
        lactation = {
            "animal_id": "test_cow_4",
            "herd_id": 9999,
            "parity": 99,
            "milk": [20.0] * 304,
            "events": ["milking"] * 304,
        }
        with open(json_dir / "animal_004.json", "w") as f:
            json.dump(lactation, f)

        source = LactationPKLSource(
            json_root_dir=json_dir,
            herd_stats_dir=fallback_test_dir,
        )
        item = source.load_item(0)

        # Should use Level 4: global average
        expected = np.array([float(i + 200) for i in range(10)], dtype=np.float32)
        assert np.allclose(item["herd_stats"], expected)


class TestLactationPKLSourceMemoryManagement:
    """Test memory management options."""

    def test_keep_in_memory_true(self, json_data_dir, herd_stats_dir):
        """Test with keep_in_memory=True."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
            keep_in_memory=True,
        )

        assert source.data_cache is not None

    def test_keep_in_memory_false(self, json_data_dir, herd_stats_dir):
        """Test with keep_in_memory=False."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
            keep_in_memory=False,
        )

        assert source.data_cache is None

    def test_consistent_access_with_and_without_cache(self, json_data_dir, herd_stats_dir):
        """Test that cached and non-cached access return same data."""
        source_cached = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
            keep_in_memory=True,
        )

        source_uncached = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
            keep_in_memory=False,
        )

        item_cached = source_cached.load_item(0)
        item_uncached = source_uncached.load_item(0)

        assert item_cached["animal_id"] == item_uncached["animal_id"]
        assert np.allclose(item_cached["herd_stats"], item_uncached["herd_stats"])


class TestLactationPKLSourceLoadItem:
    """Test load_item behavior."""

    def test_first_item(self, json_data_dir, herd_stats_dir):
        """Test accessing first item."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(0)
        assert item["animal_id"] == "cow_001"

    def test_last_item(self, json_data_dir, herd_stats_dir):
        """Test accessing last item via index calculation."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        item = source.load_item(len(source) - 1)
        assert item["animal_id"] == "cow_003"

    def test_out_of_bounds(self, json_data_dir, herd_stats_dir):
        """Test out of bounds indexing."""
        source = LactationPKLSource(
            json_root_dir=json_data_dir,
            herd_stats_dir=herd_stats_dir,
        )

        with pytest.raises((IndexError, KeyError)):
            source.load_item(100)
