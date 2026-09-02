"""Tests for the lactation model lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf
from bovi_core.config import Config
from bovi_core.ml import ModelProviderRegistry, ResolvedModelArtifact
from lactation_autoencoder.models import (
    LactationAutoencoderModel,
    LactationAutoencoderModelConfig,
    LactationAutoencoderModelProvider,
)
from lactation_autoencoder.predictors import LactationPredictor


def _model_config(**overrides: object) -> LactationAutoencoderModelConfig:
    values: dict[str, object] = {
        "framework": "tensorflow",
        "input_dim": 304,
        "latent_dim": 64,
        "num_events": 15,
        "num_herd_stats": 10,
    }
    values.update(overrides)
    return LactationAutoencoderModelConfig.model_validate(values)


def _native_model(signature: MagicMock | None = None) -> MagicMock:
    native_model = MagicMock(spec=tf.Module)
    native_model.signatures = {"serving_default": signature or MagicMock()}
    return native_model


def test_model_wraps_native_model_and_serving_signature() -> None:
    signature = MagicMock(return_value={"output": "prediction"})
    native_model = _native_model(signature)
    config = _model_config()

    model = LactationAutoencoderModel(native_model, config, signature)

    assert model.native_model is native_model
    assert model.config is config
    assert model(input_11="milk") == {"output": "prediction"}
    signature.assert_called_once_with(input_11="milk")


def test_model_config_reads_existing_architecture_shape() -> None:
    config = MagicMock()
    config.experiment.models.autoencoder.framework = "tensorflow"
    config.experiment.models.autoencoder.architecture.__dict__ = {
        "input_dim": 304,
        "latent_dim": 64,
        "num_events": 15,
        "num_herd_stats": 10,
    }

    parsed = LactationAutoencoderModelConfig.from_config(cast(Config, config))

    assert parsed.input_dim == 304
    assert parsed.latent_dim == 64
    assert parsed.signature_name == "serving_default"


def test_provider_loads_resolved_native_payload() -> None:
    native_model = _native_model()
    artifact = ResolvedModelArtifact[object](
        format="tensorflow_saved_model",
        source_uri="memory://lactation-autoencoder",
        payload=native_model,
    )

    model = LactationAutoencoderModelProvider().load_artifact(_model_config(), artifact)

    assert model.native_model is native_model


@patch("lactation_autoencoder.models.provider.tf.saved_model.load")
def test_provider_loads_local_saved_model(
    load_saved_model: MagicMock,
    tmp_path: Path,
) -> None:
    saved_model_dir = tmp_path / "saved-model"
    saved_model_dir.mkdir()
    (saved_model_dir / "saved_model.pb").touch()
    load_saved_model.return_value = _native_model()
    artifact = ResolvedModelArtifact[object](
        format="tensorflow_saved_model",
        source_uri=saved_model_dir.as_uri(),
        local_path=saved_model_dir,
    )

    model = LactationAutoencoderModelProvider().load_artifact(_model_config(), artifact)

    load_saved_model.assert_called_once_with(str(saved_model_dir))
    assert model.native_model is load_saved_model.return_value


def test_provider_rejects_unsupported_format() -> None:
    artifact = ResolvedModelArtifact[object](
        format="keras_h5",
        source_uri="memory://lactation-autoencoder",
        payload=_native_model(),
    )

    with pytest.raises(ValueError, match="Unsupported lactation model artifact format"):
        LactationAutoencoderModelProvider().load_artifact(_model_config(), artifact)


def test_provider_requires_configured_signature() -> None:
    native_model = _native_model()
    native_model.signatures = {"other": MagicMock()}
    artifact = ResolvedModelArtifact[object](
        format="tensorflow_saved_model",
        source_uri="memory://lactation-autoencoder",
        payload=native_model,
    )

    with pytest.raises(ValueError, match="serving_default"):
        LactationAutoencoderModelProvider().load_artifact(_model_config(), artifact)


def test_provider_is_registered() -> None:
    assert ModelProviderRegistry.get("autoencoder") is LactationAutoencoderModelProvider


def test_predictor_receives_model_through_constructor() -> None:
    signature = MagicMock(return_value=tf.constant([[0.25]], dtype=tf.float32))
    model = LactationAutoencoderModel(_native_model(signature), _model_config(), signature)
    predictor = LactationPredictor(model=model, config=cast(Config, MagicMock()))
    sample = {
        "milk": np.zeros(304, dtype=np.float32),
        "events": np.zeros(304, dtype=np.float32),
        "parity": 1,
        "herd_stats": np.zeros(10, dtype=np.float32),
    }

    result = predictor.predict(sample, return_format="raw")

    assert predictor.model is model
    assert isinstance(result, tf.Tensor)
    signature.assert_called_once()
