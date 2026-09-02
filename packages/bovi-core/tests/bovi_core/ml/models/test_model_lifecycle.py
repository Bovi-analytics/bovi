from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from bovi_core.config import Config
from bovi_core.ml.models import (
    Model,
    ModelArtifactReference,
    ModelConfig,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
)


class ExampleConfig(ModelConfig):
    model_key = "example"
    framework: str = "example"
    width: int = 1


class ExampleModel(Model[object, ExampleConfig]):
    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.native_model


def test_model_only_stores_runtime_dependencies():
    native_model = object()
    config = ExampleConfig()

    model = ExampleModel(native_model=native_model, config=config)

    assert model.native_model is native_model
    assert model.config is config
    assert not hasattr(model, "predictor")
    assert not hasattr(model, "weights_path")


def test_model_config_reads_framework_and_architecture_from_model_node():
    config = cast(
        Config,
        SimpleNamespace(
            experiment=SimpleNamespace(
                models=SimpleNamespace(
                    example=SimpleNamespace(
                        framework="example",
                        architecture=SimpleNamespace(width=3),
                    )
                )
            )
        ),
    )

    model_config = ExampleConfig.from_config(config)

    assert model_config.framework == "example"
    assert model_config.width == 3


def test_resolved_resources_require_materialized_content():
    with pytest.raises(ValueError, match="local_path or payload"):
        ResolvedCheckpoint(format="example", source_uri="checkpoint://one")

    with pytest.raises(ValueError, match="local_path or payload"):
        ResolvedModelArtifact(format="example", source_uri="artifact://one")


def test_resolved_artifact_can_hold_local_path():
    artifact = ResolvedModelArtifact(
        format="example",
        source_uri="artifact://one",
        local_path=Path("model.bin"),
    )

    assert artifact.local_path == Path("model.bin")


def test_artifact_reference_is_immutable():
    reference = ModelArtifactReference(uri="artifact://one", format="example")

    with pytest.raises(Exception):
        reference.uri = "artifact://two"  # type: ignore[misc]
