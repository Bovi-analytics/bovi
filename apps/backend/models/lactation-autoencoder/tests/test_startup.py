"""Startup behavior tests for the autoencoder Function App."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import main
from bovi_core.ml import ModelProviderRegistry, PredictorRegistry


def test_model_runtime_loads_lazily():
    main._model_runtime = None
    assert main._model_runtime is None
    assert main.health() == {"status": "ok"}
    assert main.health_check() == {"status": "ok"}


def test_function_app_import_registers_autoencoder_runtime():
    ModelProviderRegistry.clear()
    PredictorRegistry.clear()

    main = importlib.reload(globals()["main"])
    main._ensure_autoencoder_registered()

    assert ModelProviderRegistry.is_registered("autoencoder")
    assert PredictorRegistry.is_registered("autoencoder")


def test_model_runtime_is_assembled_through_provider_injection(monkeypatch, tmp_path):
    weights_path = tmp_path / "weights" / "autoencoder"
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            models=SimpleNamespace(
                autoencoder=SimpleNamespace(
                    default_weights_location="local",
                    local_weights=SimpleNamespace(default=str(weights_path)),
                    dataloaders=SimpleNamespace(
                        inference=SimpleNamespace(transforms=[{"name": "example"}])
                    ),
                )
            )
        )
    )
    asset_paths = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
    )
    model_config = MagicMock(name="model_config")
    model = MagicMock(name="model")
    predictor = MagicMock(name="predictor")
    transforms = {"example": MagicMock()}
    provider = MagicMock()
    provider.load_artifact.return_value = model
    provider_factory = MagicMock(return_value=provider)
    predictor_factory = MagicMock(return_value=predictor)
    transform_factory = MagicMock(return_value=transforms)

    monkeypatch.setattr(main, "_model_runtime", None)
    monkeypatch.setattr(main, "ensure_model_assets", MagicMock(return_value=asset_paths))
    monkeypatch.setattr(main, "Config", MagicMock(return_value=config))
    monkeypatch.setattr(main, "_ensure_autoencoder_registered", MagicMock())
    monkeypatch.setattr(
        main.LactationAutoencoderModelConfig,
        "from_config",
        MagicMock(return_value=model_config),
    )
    monkeypatch.setattr(main.ModelProviderRegistry, "create", provider_factory)
    monkeypatch.setattr(main.PredictorRegistry, "create", predictor_factory)
    monkeypatch.setattr(main.TransformRegistry, "from_config", transform_factory)

    runtime = main._get_model_runtime()

    assert runtime.model is model
    assert runtime.predictor is predictor
    assert runtime.transforms is transforms
    provider_factory.assert_called_once_with("autoencoder")
    provider.load_artifact.assert_called_once()
    loaded_config, artifact = provider.load_artifact.call_args.args
    assert loaded_config is model_config
    assert artifact.local_path == weights_path
    assert artifact.source_uri == weights_path.resolve().as_uri()
    predictor_factory.assert_called_once_with(
        "autoencoder",
        model=model,
        config=config,
    )
    transform_factory.assert_called_once_with(
        config.experiment.models.autoencoder.dataloaders.inference.transforms
    )
