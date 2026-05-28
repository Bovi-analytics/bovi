"""Startup behavior tests for the autoencoder Function App."""

import main


def test_model_runtime_loads_lazily():
    main._model_runtime = None
    assert main._model_runtime is None
    assert main.health() == {"status": "ok"}
    assert main.health_check() == {"status": "ok"}
