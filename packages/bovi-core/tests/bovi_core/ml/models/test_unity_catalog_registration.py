"""
Tests for Unity Catalog registration utilities.

Tests cover:
- _generate_uc_model_name() for name generation
- _resolve_alias_version() for version handling
- _generate_model_tags() for tag generation
- _get_framework() for framework detection

Note: Integration tests for register_to_unity_catalog() should be run
in a Databricks environment with proper credentials configured.
"""

from unittest.mock import Mock, MagicMock, patch
import pytest

from bovi_core.ml.models import Model
from bovi_core.ml.models.unity_catalog import (
    _generate_uc_model_name,
    _resolve_alias_version,
    _generate_model_tags,
    _get_framework,
)


class MockModel(Model):
    """Mock model for testing."""

    def __init__(self, config, model_type="pytorch"):
        self.config = config
        self.model_type = model_type
        self.model_name = "test_model"
        self.weights_name = "best"
        self.weights_location = "local"
        self.verbose = 0
        self.weights_path = "/tmp/model.pt"
        self.model = Mock()
        self.predictor = Mock()
        self.blob_container_client = None

    def _set_model_types(self):
        pass

    def load_model(self):
        pass

    def __call__(self, *args, **kwargs):
        """Mock callable for testing."""
        return self.model(*args, **kwargs)


@pytest.fixture
def mock_config():
    """Create mock config."""
    config = Mock()
    config.project = Mock()
    config.project.name = "test_project"
    config.experiment = Mock()
    config.experiment.experiment_name = "test_experiment"
    config.experiment.experiment_version = "1.0"
    config.container_client = Mock()
    return config


@pytest.fixture
def mock_model(mock_config):
    """Create mock model."""
    return MockModel(mock_config, model_type="pytorch")


class TestGenerateUCModelName:
    """Tests for _generate_uc_model_name()."""

    def test_auto_generate_name(self, mock_model):
        """Test auto-generating model name from config."""
        name = _generate_uc_model_name(
            model=mock_model,
            catalog="projects",
            schema="bovi_core",
            model_name=None
        )

        assert name == "projects.bovi_core.test_project_test_model"

    def test_custom_name(self, mock_model):
        """Test with custom model name."""
        name = _generate_uc_model_name(
            model=mock_model,
            catalog="prod",
            schema="models",
            model_name="custom_model"
        )

        assert name == "prod.models.custom_model"

    def test_different_catalog_schema(self, mock_model):
        """Test with different catalog and schema."""
        name = _generate_uc_model_name(
            model=mock_model,
            catalog="production",
            schema="ml_models",
            model_name="detector"
        )

        assert name == "production.ml_models.detector"


class TestResolveAliasVersion:
    """Tests for _resolve_alias_version()."""

    @patch('mlflow.MlflowClient')
    def test_alias_not_exists(self, mock_mlflow_client):
        """Test when alias doesn't exist."""
        client = MagicMock()
        mock_mlflow_client.return_value = client
        client.search_model_versions.return_value = []

        result = _resolve_alias_version(
            "projects.bovi_core.model",
            "v1.0"
        )

        assert result == "v1.0"

    @patch('mlflow.MlflowClient')
    def test_alias_exists_increments_version(self, mock_mlflow_client):
        """Test auto-incrementing when alias exists."""
        client = MagicMock()
        mock_mlflow_client.return_value = client

        # Mock existing version with alias
        version = Mock()
        version.aliases = ["v1.0"]
        client.search_model_versions.return_value = [version]

        result = _resolve_alias_version(
            "projects.bovi_core.model",
            "v1.0"
        )

        # Should increment to v1.1
        assert result == "v1.1"

    @patch('mlflow.MlflowClient')
    def test_alias_without_version_number(self, mock_mlflow_client):
        """Test alias without version number."""
        client = MagicMock()
        mock_mlflow_client.return_value = client

        version = Mock()
        version.aliases = ["Champion"]
        client.search_model_versions.return_value = [version]

        result = _resolve_alias_version(
            "projects.bovi_core.model",
            "Champion"
        )

        # Should append _v2
        assert result == "Champion_v2"

    @patch('mlflow.MlflowClient')
    def test_model_not_exists(self, mock_mlflow_client):
        """Test when model doesn't exist in UC."""
        client = MagicMock()
        mock_mlflow_client.return_value = client
        client.search_model_versions.side_effect = Exception("Model not found")

        result = _resolve_alias_version(
            "projects.bovi_core.new_model",
            "v1.0"
        )

        # Should return original alias
        assert result == "v1.0"


class TestGenerateModelTags:
    """Tests for _generate_model_tags()."""

    def test_auto_generated_tags(self, mock_model):
        """Test auto-generated tags."""
        tags = _generate_model_tags(mock_model)

        assert tags["project"] == "test_project"
        assert tags["model_type"] == "pytorch"
        assert tags["framework"] == "pytorch"
        assert tags["experiment"] == "test_experiment"
        assert tags["experiment_version"] == "1.0"
        assert tags["weights_name"] == "best"

    def test_merge_custom_tags(self, mock_model):
        """Test merging custom tags."""
        custom = {"custom_key": "custom_value", "task": "detection"}
        tags = _generate_model_tags(mock_model, custom)

        assert tags["custom_key"] == "custom_value"
        assert tags["task"] == "detection"
        assert tags["project"] == "test_project"

    def test_custom_overrides_auto(self, mock_model):
        """Test custom tags override auto-generated."""
        custom = {"project": "override_project"}
        tags = _generate_model_tags(mock_model, custom)

        assert tags["project"] == "override_project"


class TestGetFramework:
    """Tests for _get_framework()."""

    def test_pytorch_framework(self, mock_model):
        """Test PyTorch framework detection."""
        mock_model.model_type = "pytorch"
        assert _get_framework(mock_model) == "pytorch"

    def test_tensorflow_framework(self, mock_model):
        """Test TensorFlow framework detection."""
        mock_model.model_type = "tensorflow"
        assert _get_framework(mock_model) == "tensorflow"

    def test_keras_framework(self, mock_model):
        """Test Keras framework detection."""
        mock_model.model_type = "keras"
        assert _get_framework(mock_model) == "keras"

    def test_unknown_framework(self, mock_model):
        """Test unknown framework defaults to python."""
        mock_model.model_type = "custom"
        assert _get_framework(mock_model) == "python"
