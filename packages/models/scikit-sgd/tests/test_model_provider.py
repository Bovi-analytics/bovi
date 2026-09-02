"""Tests for the scikit SGD runtime and lifecycle provider."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from bovi_core.ml import ModelProviderRegistry, ResolvedCheckpoint, ResolvedModelArtifact
from scikit_sgd import ScikitSGDModelConfig, ScikitSGDModelProvider
from sklearn.linear_model import SGDRegressor


def test_provider_is_registered() -> None:
    assert ModelProviderRegistry.get("scikit_sgd") is ScikitSGDModelProvider


def test_provider_creates_configured_fresh_model(model_config: ScikitSGDModelConfig) -> None:
    model = ScikitSGDModelProvider().create(model_config)

    assert isinstance(model.native_model, SGDRegressor)
    assert model.native_model.fit_intercept is True
    assert model.native_model.random_state == 42


def test_provider_restores_checkpoint_payload(model_config: ScikitSGDModelConfig) -> None:
    estimator = SGDRegressor().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    checkpoint = ResolvedCheckpoint[object](
        format="scikit-joblib",
        source_uri="memory://checkpoint",
        payload=estimator,
    )

    model = ScikitSGDModelProvider().restore_checkpoint(model_config, checkpoint)

    assert model.native_model is estimator


def test_provider_loads_local_joblib_artifact(
    model_config: ScikitSGDModelConfig,
    tmp_path: Path,
) -> None:
    estimator = SGDRegressor().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(estimator, artifact_path)
    artifact = ResolvedModelArtifact[object](
        format="scikit-joblib",
        source_uri=artifact_path.as_uri(),
        local_path=artifact_path,
    )

    model = ScikitSGDModelProvider().load_artifact(model_config, artifact)

    assert model(np.array([[0.5]])).shape == (1,)


def test_provider_rejects_wrong_resource_type(model_config: ScikitSGDModelConfig) -> None:
    artifact = ResolvedModelArtifact[object](
        format="scikit-joblib",
        source_uri="memory://wrong-model",
        payload={"not": "an estimator"},
    )

    with pytest.raises(TypeError, match="SGDRegressor"):
        ScikitSGDModelProvider().load_artifact(model_config, artifact)
