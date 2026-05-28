"""Startup behavior tests for the autoencoder Function App."""

import importlib

import main
from bovi_core.ml import ModelRegistry, PredictorRegistry


def test_model_runtime_loads_lazily():
    main._model_runtime = None
    assert main._model_runtime is None
    assert main.health() == {"status": "ok"}
    assert main.health_check() == {"status": "ok"}


def test_function_app_import_registers_autoencoder_runtime():
    ModelRegistry.clear()
    PredictorRegistry.clear()

    main = importlib.reload(globals()["main"])
    main._ensure_autoencoder_registered()

    assert ModelRegistry.is_registered("autoencoder")
    assert PredictorRegistry.is_registered("autoencoder")
