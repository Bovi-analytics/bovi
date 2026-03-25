"""
Plugin registry system for models and predictors.

Allows models to self-register from any package, enabling
plugin-style architecture without editing core code.

Auto-discovery via ``importlib.metadata`` entry points means
consumer code never has to import model packages manually.
"""

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

if TYPE_CHECKING:
    from bovi_core.config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Global registry for model classes.

    Models register themselves via the ``@ModelRegistry.register`` decorator.
    When ``get()`` is called for an unregistered name, the registry
    automatically scans the ``bovi.models`` entry-point group and loads
    only the matching entry point (lazy, on-demand discovery).
    """

    _models: Dict[str, Type] = {}
    _factories: Dict[str, Callable] = {}
    _discovered: set[str] = set()
    _EP_GROUP = "bovi.models"

    @classmethod
    def register(cls, name: str, factory: Optional[Callable] = None):
        """Register a model class.

        Usage:
            @ModelRegistry.register("yolo")
            class YOLOModel(Model):
                pass

        Args:
            name: Unique identifier for the model.
            factory: Optional factory function for complex initialization.
        """

        def decorator(model_class: Type):
            if name in cls._models:
                logger.warning(
                    f"Model '{name}' is already registered. "
                    f"Overwriting {cls._models[name]} with {model_class}"
                )

            cls._models[name] = model_class
            if factory:
                cls._factories[name] = factory

            logger.debug(f"Registered model: {name} -> {model_class}")
            return model_class

        if factory is not None:
            cls._factories[name] = factory
            return lambda x: x

        return decorator

    @classmethod
    def _discover(cls, name: str) -> bool:
        """Attempt to discover and load a model by name via entry points.

        Only loads the entry point that matches *name*, keeping discovery
        lazy.  Results are cached in ``_discovered`` so a missing name is
        not re-scanned on every call.

        Args:
            name: Model name to discover.

        Returns:
            True if the model is now registered.
        """
        if name in cls._discovered:
            return name in cls._models
        cls._discovered.add(name)

        for ep in entry_points(group=cls._EP_GROUP, name=name):
            logger.debug("Auto-discovering model '%s' from entry point: %s", name, ep)
            try:
                ep.load()
            except Exception:
                logger.warning(
                    "Failed to load entry point '%s' for model '%s'",
                    ep,
                    name,
                    exc_info=True,
                )

        return name in cls._models

    @classmethod
    def get(cls, name: str) -> Type:
        """Get a registered model class by name.

        If *name* is not yet registered, auto-discovery is attempted
        via entry points before raising.
        """
        if name not in cls._models:
            cls._discover(name)
        if name not in cls._models:
            available = ", ".join(sorted(cls._models.keys()))
            available_eps = ", ".join(sorted(cls.list_available().keys()))
            raise ValueError(
                f"Model '{name}' not found in registry.\n"
                f"Registered models: {available}\n"
                f"Available entry points: {available_eps}\n"
                f"Hint: Install a package that provides a 'bovi.models' entry point for '{name}'."
            )
        return cls._models[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Create a model instance.

        Args:
            name: Model identifier.
            *args: Positional arguments for model constructor.
            **kwargs: Keyword arguments for model constructor.

        Returns:
            Instantiated model.
        """
        if name in cls._factories:
            return cls._factories[name](*args, **kwargs)

        model_class = cls.get(name)
        return model_class(*args, **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """List all models available via entry points (without importing them).

        Returns:
            Dict mapping model name to entry point value string.
        """
        return {ep.name: str(ep.value) for ep in entry_points(group=cls._EP_GROUP)}

    @classmethod
    def discover_all(cls) -> None:
        """Import all model entry points.

        Use sparingly — this loads every model package and its dependencies.
        """
        for ep in entry_points(group=cls._EP_GROUP):
            if ep.name not in cls._models and ep.name not in cls._discovered:
                cls._discovered.add(ep.name)
                try:
                    ep.load()
                except Exception:
                    logger.warning("Failed to load entry point: %s", ep, exc_info=True)

    @classmethod
    def list_models(cls, discover: bool = False) -> Dict[str, str]:
        """List all registered models.

        Args:
            discover: If True, discover all entry points first.
        """
        if discover:
            cls.discover_all()
        return {
            name: f"{model_class.__module__}.{model_class.__name__}"
            for name, model_class in cls._models.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name in cls._models

    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._models.clear()
        cls._factories.clear()
        cls._discovered.clear()


class PredictorRegistry:
    """Global registry for predictor classes.

    Mirrors ``ModelRegistry`` with auto-discovery via the
    ``bovi.predictors`` entry-point group.
    """

    _predictors: Dict[str, Type] = {}
    _discovered: set[str] = set()
    _EP_GROUP = "bovi.predictors"

    @classmethod
    def register(cls, name: str):
        """Register a predictor class."""

        def decorator(predictor_class: Type):
            if name in cls._predictors:
                logger.warning(
                    f"Predictor '{name}' is already registered. "
                    f"Overwriting {cls._predictors[name]} with {predictor_class}"
                )

            cls._predictors[name] = predictor_class
            logger.debug(f"Registered predictor: {name} -> {predictor_class}")
            return predictor_class

        return decorator

    @classmethod
    def _discover(cls, name: str) -> bool:
        """Attempt to discover and load a predictor by name via entry points."""
        if name in cls._discovered:
            return name in cls._predictors
        cls._discovered.add(name)

        for ep in entry_points(group=cls._EP_GROUP, name=name):
            logger.debug("Auto-discovering predictor '%s' from entry point: %s", name, ep)
            try:
                ep.load()
            except Exception:
                logger.warning(
                    "Failed to load entry point '%s' for predictor '%s'",
                    ep,
                    name,
                    exc_info=True,
                )

        return name in cls._predictors

    @classmethod
    def get(cls, name: str) -> Type:
        """Get a registered predictor class by name.

        If *name* is not yet registered, auto-discovery is attempted
        via entry points before raising.
        """
        if name not in cls._predictors:
            cls._discover(name)
        if name not in cls._predictors:
            available = ", ".join(sorted(cls._predictors.keys()))
            available_eps = ", ".join(sorted(cls.list_available().keys()))
            raise ValueError(
                f"Predictor '{name}' not found in registry.\n"
                f"Registered predictors: {available}\n"
                f"Available entry points: {available_eps}\n"
                "Hint: Install a package that provides a "
                f"'bovi.predictors' entry point for '{name}'."
            )
        return cls._predictors[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Create a predictor instance."""
        predictor_class = cls.get(name)
        return predictor_class(*args, **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, str]:
        """List all predictors available via entry points (without importing them)."""
        return {ep.name: str(ep.value) for ep in entry_points(group=cls._EP_GROUP)}

    @classmethod
    def discover_all(cls) -> None:
        """Import all predictor entry points."""
        for ep in entry_points(group=cls._EP_GROUP):
            if ep.name not in cls._predictors and ep.name not in cls._discovered:
                cls._discovered.add(ep.name)
                try:
                    ep.load()
                except Exception:
                    logger.warning("Failed to load entry point: %s", ep, exc_info=True)

    @classmethod
    def list_predictors(cls, discover: bool = False) -> Dict[str, str]:
        """List all registered predictors.

        Args:
            discover: If True, discover all entry points first.
        """
        if discover:
            cls.discover_all()
        return {
            name: f"{pred_class.__module__}.{pred_class.__name__}"
            for name, pred_class in cls._predictors.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a predictor is registered."""
        return name in cls._predictors

    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)."""
        cls._predictors.clear()
        cls._discovered.clear()


def create_model(
    config: "Config",
    model_name: str,
    weights_path: Optional[str] = None,
) -> Any:
    """Create a model with auto-discovery — one-line model loading.

    Discovers the model and predictor via entry points, then delegates
    to the model's ``from_config()`` for path resolution.

    Args:
        config: Config instance with experiment settings.
        model_name: Registry name (e.g., ``"yolo"``, ``"autoencoder"``).
        weights_path: Optional override. If ``None``, ``from_config()``
            resolves it from the experiment config.

    Returns:
        Instantiated model ready for use.

    Example:
        >>> from bovi_core.config import Config
        >>> from bovi_core.ml import create_model
        >>> config = Config(experiment_name="yolo", project_name="bovi-yolo")
        >>> model = create_model(config, "yolo")
    """
    predictor = PredictorRegistry.create(model_name, config=config)
    model_class = ModelRegistry.get(model_name)
    return model_class.from_config(config=config, predictor=predictor, weights_path=weights_path)
