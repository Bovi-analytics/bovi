"""Startup behavior tests for the autoencoder Function App."""

import main


def test_model_runtime_loads_lazily():
    assert main._model_runtime is None
    assert main.health() == {"status": "ok"}
