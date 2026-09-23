"""Lazy plugin registries for model providers, predictors, and data factories."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast

logger = logging.getLogger(__name__)

RegisteredT = TypeVar("RegisteredT")


class _PluginRegistry(Generic[RegisteredT]):
    _entries: ClassVar[dict[str, type[Any]]]
    _discovered: ClassVar[set[str]]
    _entry_point_group: ClassVar[str]

    @classmethod
    def register(cls, name: str) -> Callable[[type[RegisteredT]], type[RegisteredT]]:
        def decorator(registered_class: type[RegisteredT]) -> type[RegisteredT]:
            if name in cls._entries:
                logger.warning("Plugin '%s' is already registered; overwriting it", name)
            cls._entries[name] = registered_class
            return registered_class

        return decorator

    @classmethod
    def _discover(cls, name: str) -> None:
        if name in cls._discovered:
            return
        cls._discovered.add(name)

        for entry_point in entry_points(group=cls._entry_point_group, name=name):
            try:
                loaded = entry_point.load()
            except Exception:
                logger.warning("Failed to load plugin entry point '%s'", entry_point, exc_info=True)
                continue
            if name not in cls._entries and isinstance(loaded, type):
                cls._entries[name] = loaded

    @classmethod
    def get(cls, name: str) -> type[RegisteredT]:
        if name not in cls._entries:
            cls._discover(name)
        if name not in cls._entries:
            available = ", ".join(sorted(cls.list_available()))
            raise ValueError(
                f"Plugin '{name}' not found in {cls._entry_point_group}. "
                f"Available entry points: {available or '(none)'}"
            )
        return cls._entries[name]

    @classmethod
    def create(cls, name: str, *args: object, **kwargs: object) -> RegisteredT:
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> dict[str, str]:
        return {
            entry_point.name: str(entry_point.value)
            for entry_point in entry_points(group=cls._entry_point_group)
        }

    @classmethod
    def discover_all(cls) -> None:
        for name in cls.list_available():
            cls._discover(name)

    @classmethod
    def list_registered(cls, discover: bool = False) -> dict[str, str]:
        if discover:
            cls.discover_all()
        return {
            name: f"{registered.__module__}.{registered.__name__}"
            for name, registered in cls._entries.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._entries

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()
        cls._discovered.clear()


class ModelProviderRegistry(_PluginRegistry[object]):
    """Registry for model-family construction and loading providers."""

    _entries: ClassVar[dict[str, type[Any]]] = {}
    _discovered: ClassVar[set[str]] = set()
    _entry_point_group = "bovi.model_providers"

    @classmethod
    def list_providers(cls, discover: bool = False) -> dict[str, str]:
        return cls.list_registered(discover)


class PredictorRegistry(_PluginRegistry[object]):
    """Registry for predictors receiving runtime models through their constructor."""

    _entries: ClassVar[dict[str, type[Any]]] = {}
    _discovered: ClassVar[set[str]] = set()
    _entry_point_group = "bovi.predictors"

    @classmethod
    def list_predictors(cls, discover: bool = False) -> dict[str, str]:
        return cls.list_registered(discover)


DataLoaderFactoryCallable = Callable[[Any, Any], Any]
DataLoaderFactoryT = TypeVar("DataLoaderFactoryT", bound=DataLoaderFactoryCallable)


class DataLoaderFactoryRegistry:
    """Registry for model-owned dataloader factory functions."""

    _entries: ClassVar[dict[str, DataLoaderFactoryCallable]] = {}
    _discovered: ClassVar[set[str]] = set()
    _entry_point_group = "bovi.dataloader_factories"

    @classmethod
    def register(cls, name: str) -> Callable[[DataLoaderFactoryT], DataLoaderFactoryT]:
        def decorator(factory: DataLoaderFactoryT) -> DataLoaderFactoryT:
            if name in cls._entries:
                logger.warning("Plugin '%s' is already registered; overwriting it", name)
            cls._entries[name] = factory
            return factory

        return decorator

    @classmethod
    def _discover(cls, name: str) -> None:
        if name in cls._discovered:
            return
        cls._discovered.add(name)

        for entry_point in entry_points(group=cls._entry_point_group, name=name):
            try:
                loaded = entry_point.load()
            except Exception:
                logger.warning("Failed to load plugin entry point '%s'", entry_point, exc_info=True)
                continue
            if name not in cls._entries and callable(loaded):
                cls._entries[name] = cast(DataLoaderFactoryCallable, loaded)

    @classmethod
    def get(cls, name: str) -> DataLoaderFactoryCallable:
        if name not in cls._entries:
            cls._discover(name)
        if name not in cls._entries:
            available = ", ".join(sorted(cls.list_available()))
            raise ValueError(
                f"Plugin '{name}' not found in {cls._entry_point_group}. "
                f"Available entry points: {available or '(none)'}"
            )
        return cls._entries[name]

    @classmethod
    def create(cls, name: str, data_config: Any, model_config: Any) -> Any:
        return cls.get(name)(data_config, model_config)

    @classmethod
    def list_available(cls) -> dict[str, str]:
        return {
            entry_point.name: str(entry_point.value)
            for entry_point in entry_points(group=cls._entry_point_group)
        }

    @classmethod
    def list_factories(cls, discover: bool = False) -> dict[str, str]:
        if discover:
            for name in cls.list_available():
                cls._discover(name)
        return {
            name: f"{factory.__module__}.{factory.__name__}"
            for name, factory in cls._entries.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._entries

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()
        cls._discovered.clear()
