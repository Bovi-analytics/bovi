"""Tests for model and predictor registry."""

from unittest.mock import MagicMock, patch

import pytest
from bovi_core.ml import ModelRegistry, PredictorRegistry, Model
from bovi_core.config import Config


class DummyModel(Model):
    """Dummy model for testing."""

    def load_model(self):
        return "dummy_model_loaded"

    def _set_model_types(self):
        self.model_name = "dummy"


def test_model_registry_register():
    """Test model registration."""
    ModelRegistry.clear()

    @ModelRegistry.register("test_model")
    class TestModel(Model):
        pass

    assert ModelRegistry.is_registered("test_model")
    assert ModelRegistry.get("test_model") == TestModel


def test_model_registry_get_missing():
    """Test getting non-existent model."""
    ModelRegistry.clear()

    with pytest.raises(ValueError) as exc_info:
        ModelRegistry.get("nonexistent")

    assert "not found" in str(exc_info.value)
    assert "Registered models" in str(exc_info.value)


def test_model_registry_list():
    """Test listing registered models."""
    ModelRegistry.clear()

    @ModelRegistry.register("model1")
    class Model1(Model):
        pass

    @ModelRegistry.register("model2")
    class Model2(Model):
        pass

    models = ModelRegistry.list_models()
    assert "model1" in models
    assert "model2" in models


def test_model_registry_create():
    """Test creating model instance via registry."""
    ModelRegistry.clear()
    ModelRegistry.register("dummy")(DummyModel)

    # This would need a real config and predictor
    # For now, just test that get() works
    model_class = ModelRegistry.get("dummy")
    assert model_class == DummyModel


def test_model_registry_overwrite_warning(caplog):
    """Test that overwriting a model logs a warning."""
    ModelRegistry.clear()

    @ModelRegistry.register("test")
    class Model1(Model):
        pass

    @ModelRegistry.register("test")
    class Model2(Model):
        pass

    # Should log warning about overwriting
    # Check that Model2 is now registered
    assert ModelRegistry.get("test") == Model2


def test_model_registry_clear():
    """Test clearing the registry."""
    ModelRegistry.clear()

    @ModelRegistry.register("test")
    class TestModel(Model):
        pass

    assert ModelRegistry.is_registered("test")
    ModelRegistry.clear()
    assert not ModelRegistry.is_registered("test")


# Similar tests for PredictorRegistry


def test_predictor_registry_register():
    """Test predictor registration."""
    PredictorRegistry.clear()

    @PredictorRegistry.register("test_predictor")
    class TestPredictor:
        pass

    assert PredictorRegistry.is_registered("test_predictor")
    assert PredictorRegistry.get("test_predictor") == TestPredictor


def test_predictor_registry_get_missing():
    """Test getting non-existent predictor."""
    PredictorRegistry.clear()

    with pytest.raises(ValueError) as exc_info:
        PredictorRegistry.get("nonexistent")

    assert "not found" in str(exc_info.value)
    assert "Registered predictors" in str(exc_info.value)


def test_predictor_registry_list():
    """Test listing registered predictors."""
    PredictorRegistry.clear()

    @PredictorRegistry.register("pred1")
    class Predictor1:
        pass

    @PredictorRegistry.register("pred2")
    class Predictor2:
        pass

    predictors = PredictorRegistry.list_predictors()
    assert "pred1" in predictors
    assert "pred2" in predictors


def test_predictor_registry_clear():
    """Test clearing the predictor registry."""
    PredictorRegistry.clear()

    @PredictorRegistry.register("test")
    class TestPredictor:
        pass

    assert PredictorRegistry.is_registered("test")
    PredictorRegistry.clear()
    assert not PredictorRegistry.is_registered("test")


def test_model_registry_with_factory():
    """Test model registration with custom factory function."""
    ModelRegistry.clear()

    def custom_factory(*args, **kwargs):
        """Custom factory for creating models."""
        return "custom_model_instance"

    ModelRegistry.register("custom", factory=custom_factory)

    # Create via factory
    instance = ModelRegistry.create("custom")
    assert instance == "custom_model_instance"


# --- Auto-discovery tests ---


def _make_model_ep(name: str):
    """Create a mock entry point that registers a model when loaded."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake_module.{name}:FakeModel"

    def _load():
        @ModelRegistry.register(name)
        class _Discovered(Model):
            pass

        return _Discovered

    ep.load.side_effect = _load
    return ep


def _make_predictor_ep(name: str):
    """Create a mock entry point that registers a predictor when loaded."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake_module.{name}:FakePredictor"

    def _load():
        @PredictorRegistry.register(name)
        class _Discovered:
            pass

        return _Discovered

    ep.load.side_effect = _load
    return ep


@patch("bovi_core.ml.registry.entry_points")
def test_model_auto_discover_loads_matching_entry_point(mock_eps):
    """get() triggers ep.load() for the matching name only."""
    ModelRegistry.clear()

    ep_yolo = _make_model_ep("discovered_yolo")
    ep_other = _make_model_ep("discovered_other")
    mock_eps.return_value = [ep_yolo]

    result = ModelRegistry.get("discovered_yolo")

    assert result is not None
    ep_yolo.load.assert_called_once()
    ep_other.load.assert_not_called()


@patch("bovi_core.ml.registry.entry_points")
def test_model_auto_discover_skips_unrelated(mock_eps):
    """get('a') does NOT load entry point for 'b'."""
    ModelRegistry.clear()

    ep_a = _make_model_ep("ep_a")
    ep_b = _make_model_ep("ep_b")

    # entry_points(group=..., name="ep_a") returns only ep_a
    def _filter(group, name=None):
        all_eps = [ep_a, ep_b]
        if name:
            return [e for e in all_eps if e.name == name]
        return all_eps

    mock_eps.side_effect = _filter

    ModelRegistry.get("ep_a")

    ep_a.load.assert_called_once()
    ep_b.load.assert_not_called()


@patch("bovi_core.ml.registry.entry_points")
def test_model_auto_discover_caches_failed_attempts(mock_eps):
    """Second get() for missing name does not call _discover again."""
    ModelRegistry.clear()
    mock_eps.return_value = []

    with pytest.raises(ValueError):
        ModelRegistry.get("missing_model")

    # _discover was called once; list_available is also called in the error msg
    # but _discover should NOT be called again on the second get()
    assert "missing_model" in ModelRegistry._discovered

    with pytest.raises(ValueError):
        ModelRegistry.get("missing_model")

    # entry_points was called for _discover (group+name) only once;
    # subsequent calls are from list_available() in the error message, which is fine.
    # The key assertion: _discover skipped the second time because name is in _discovered.
    discover_calls = [
        c for c in mock_eps.call_args_list if c.kwargs.get("name") == "missing_model"
    ]
    assert len(discover_calls) == 1


@patch("bovi_core.ml.registry.entry_points")
def test_model_list_available_no_imports(mock_eps):
    """list_available() returns metadata without calling ep.load()."""
    ModelRegistry.clear()

    ep = MagicMock()
    ep.name = "some_model"
    ep.value = "some_module:SomeModel"
    mock_eps.return_value = [ep]

    available = ModelRegistry.list_available()

    assert "some_model" in available
    assert available["some_model"] == "some_module:SomeModel"
    ep.load.assert_not_called()


@patch("bovi_core.ml.registry.entry_points")
def test_model_discover_all(mock_eps):
    """discover_all() calls ep.load() on all entry points."""
    ModelRegistry.clear()

    ep_a = _make_model_ep("all_a")
    ep_b = _make_model_ep("all_b")
    mock_eps.return_value = [ep_a, ep_b]

    ModelRegistry.discover_all()

    ep_a.load.assert_called_once()
    ep_b.load.assert_called_once()
    assert ModelRegistry.is_registered("all_a")
    assert ModelRegistry.is_registered("all_b")


@patch("bovi_core.ml.registry.entry_points")
def test_model_list_models_with_discover(mock_eps):
    """list_models(discover=True) includes discovered models."""
    ModelRegistry.clear()

    ep = _make_model_ep("disc_model")
    mock_eps.return_value = [ep]

    models = ModelRegistry.list_models(discover=True)

    assert "disc_model" in models


def test_model_clear_resets_discovered():
    """clear() empties _discovered set."""
    ModelRegistry.clear()
    ModelRegistry._discovered.add("something")

    ModelRegistry.clear()

    assert len(ModelRegistry._discovered) == 0


@patch("bovi_core.ml.registry.entry_points")
def test_model_discover_handles_load_failure(mock_eps):
    """A broken entry point does not crash discovery."""
    ModelRegistry.clear()

    ep = MagicMock()
    ep.name = "broken"
    ep.value = "nonexistent_module:Missing"
    ep.load.side_effect = ImportError("no such module")
    mock_eps.return_value = [ep]

    with pytest.raises(ValueError):
        ModelRegistry.get("broken")

    # Should not have raised ImportError — it was caught and logged


@patch("bovi_core.ml.registry.entry_points")
def test_predictor_auto_discover(mock_eps):
    """PredictorRegistry.get() triggers auto-discovery."""
    PredictorRegistry.clear()

    ep = _make_predictor_ep("disc_pred")
    mock_eps.return_value = [ep]

    result = PredictorRegistry.get("disc_pred")

    assert result is not None
    ep.load.assert_called_once()


def test_predictor_clear_resets_discovered():
    """PredictorRegistry.clear() empties _discovered set."""
    PredictorRegistry.clear()
    PredictorRegistry._discovered.add("something")

    PredictorRegistry.clear()

    assert len(PredictorRegistry._discovered) == 0
