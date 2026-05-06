"""Tests for config_utils module."""

from bovi_core.utils.config_utils import extract_experiment_name_from_path


class TestExtractExperimentNameFromPath:
    """Tests for extract_experiment_name_from_path function."""

    def test_versioned_path_extracts_experiment_name(self):
        """Test extraction from versioned path structure."""
        path = "/project/data/experiments/lactation_autoencoder/versions/v1/config/config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "lactation_autoencoder"

    def test_legacy_path_extracts_experiment_name(self):
        """Test extraction from legacy path structure."""
        path = "/project/data/experiments/yolo_experiment/config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "yolo_experiment"

    def test_nested_versioned_path(self):
        """Test extraction from deeply nested versioned path."""
        path = "/some/root/data/experiments/my_exp/versions/v2/config/dev.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "my_exp"

    def test_windows_style_path(self):
        """Test extraction works with different path separators."""
        path = "C:/Users/user/project/data/experiments/test_exp/versions/v1/config/config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "test_exp"

    def test_path_without_experiments_returns_none(self):
        """Test returns None when 'experiments' not in path."""
        path = "/project/data/configs/my_config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result is None

    def test_path_with_experiments_at_end_returns_none(self):
        """Test returns None when 'experiments' is last element without following name."""
        path = "/project/data/experiments"
        result = extract_experiment_name_from_path(path)
        assert result is None

    def test_relative_path_with_experiments(self):
        """Test extraction works with relative paths."""
        path = "data/experiments/relative_exp/versions/v1/config/config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "relative_exp"

    def test_experiment_name_with_special_characters(self):
        """Test extraction with underscores and numbers in experiment name."""
        path = "/data/experiments/yolo_v8_2024_exp/versions/v1/config/config.yaml"
        result = extract_experiment_name_from_path(path)
        assert result == "yolo_v8_2024_exp"
