"""Tests for explicit Unity Catalog artifact publishing."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from bovi_core.config import Config
from bovi_core.ml.models import Model, ModelConfig, ResolvedModelArtifact
from bovi_core.ml.publishing.unity_catalog import (
    UnityCatalogPublisher,
    generate_model_tags,
    generate_uc_model_name,
    resolve_alias_version,
)


class DummyModelConfig(ModelConfig):
    model_key = "test_model"


class DummyModel(Model[object, DummyModelConfig]):
    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.native_model


@pytest.fixture
def runtime_config() -> Config:
    return cast(
        Config,
        SimpleNamespace(
            author_email="owner@example.com",
            project=SimpleNamespace(name="test_project"),
            experiment=SimpleNamespace(
                experiment_name="test_experiment",
                experiment_version="1.0",
            ),
        ),
    )


@pytest.fixture
def model() -> DummyModel:
    return DummyModel(
        native_model=object(),
        config=DummyModelConfig(framework="pytorch"),
    )


@pytest.fixture
def artifact(tmp_path: Path) -> ResolvedModelArtifact[object]:
    artifact_path = tmp_path / "model.pt"
    artifact_path.write_bytes(b"exported model")
    return ResolvedModelArtifact(
        format="pytorch-module",
        source_uri="file:///exports/model.pt",
        local_path=artifact_path,
    )


def test_generate_uc_model_name_uses_explicit_runtime_dependencies(
    runtime_config: Config, model: DummyModel
) -> None:
    assert (
        generate_uc_model_name(
            config=runtime_config,
            model=model,
            catalog="projects",
            schema="bovi_core",
            model_name=None,
        )
        == "projects.bovi_core.test_project_test_model"
    )
    assert (
        generate_uc_model_name(
            config=runtime_config,
            model=model,
            catalog="prod",
            schema="models",
            model_name="custom_model",
        )
        == "prod.models.custom_model"
    )


def test_generate_model_tags_uses_config_model_and_artifact(
    runtime_config: Config,
    model: DummyModel,
    artifact: ResolvedModelArtifact[object],
) -> None:
    tags = generate_model_tags(
        config=runtime_config,
        model=model,
        artifact=artifact,
        custom_tags={"project": "override", "task": "detection"},
    )

    assert tags == {
        "project": "override",
        "model_key": "test_model",
        "framework": "pytorch",
        "artifact_format": "pytorch-module",
        "artifact_uri": "file:///exports/model.pt",
        "experiment": "test_experiment",
        "experiment_version": "1.0",
        "task": "detection",
    }


@pytest.mark.parametrize(
    ("existing_aliases", "requested", "expected"),
    [
        ([], "v1.0", "v1.0"),
        (["v1.0"], "v1.0", "v1.1"),
        (["v1.0", "v1.1"], "v1.0", "v1.2"),
        (["Champion"], "Champion", "Champion_v2"),
        (["Champion", "Champion_v2"], "Champion", "Champion_v3"),
    ],
)
def test_resolve_alias_version(existing_aliases: list[str], requested: str, expected: str) -> None:
    client = MagicMock()
    client.search_model_versions.return_value = [SimpleNamespace(aliases=existing_aliases)]

    assert resolve_alias_version(client, "catalog.schema.model", requested) == expected


def test_resolve_alias_returns_requested_alias_when_model_does_not_exist() -> None:
    client = MagicMock()
    client.search_model_versions.side_effect = RuntimeError("not found")

    assert resolve_alias_version(client, "catalog.schema.model", "Champion") == "Champion"


def test_publish_logs_preexported_artifact_and_sets_catalog_metadata(
    runtime_config: Config,
    model: DummyModel,
    artifact: ResolvedModelArtifact[object],
    tmp_path: Path,
) -> None:
    mlflow = MagicMock()
    mlflow.start_run.return_value = nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    )
    model_version = SimpleNamespace(version="7")
    mlflow.register_model.return_value = model_version
    registry_client = MagicMock()
    registry_client.search_model_versions.return_value = []
    workspace_client = MagicMock()
    workspace_client.workspace.get_status.return_value = object()
    predictor = MagicMock()
    python_model = object()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models: {}", encoding="utf-8")

    publisher = UnityCatalogPublisher(
        mlflow=mlflow,
        registry_client=registry_client,
        workspace_client=workspace_client,
    )
    result = publisher.publish(
        config=runtime_config,
        model=model,
        predictor=predictor,
        artifact=artifact,
        python_model=python_model,
        pip_requirements=["torch==2.7.1", "numpy"],
        alias="v1.0",
        description="A test model",
        input_example={"feature": [1.0]},
        signature="explicit-signature",
        mlflow_experiment_name="/Users/owner@example.com/test/run_0",
        config_artifact_path=config_path,
    )

    assert result is model_version
    assert artifact.local_path is not None
    mlflow.pyfunc.log_model.assert_called_once_with(
        python_model=python_model,
        name="model",
        artifacts={
            "model_path": str(artifact.local_path.resolve()),
            "config_yaml": str(config_path.resolve()),
        },
        signature="explicit-signature",
        input_example={"feature": [1.0]},
        pip_requirements=["torch==2.7.1", "numpy"],
    )
    mlflow.register_model.assert_called_once_with(
        model_uri="runs:/run-123/model",
        name="projects.bovi_core.test_project_test_model",
        tags={
            "project": "test_project",
            "model_key": "test_model",
            "framework": "pytorch",
            "artifact_format": "pytorch-module",
            "artifact_uri": "file:///exports/model.pt",
            "experiment": "test_experiment",
            "experiment_version": "1.0",
        },
    )
    registry_client.set_registered_model_alias.assert_called_once_with(
        name="projects.bovi_core.test_project_test_model",
        alias="v1.0",
        version="7",
    )
    registry_client.update_model_version.assert_called_once_with(
        name="projects.bovi_core.test_project_test_model",
        version="7",
        description="A test model",
    )
    predictor.predict.assert_not_called()


def test_publish_infers_signature_through_explicit_predictor(
    runtime_config: Config,
    model: DummyModel,
    artifact: ResolvedModelArtifact[object],
) -> None:
    mlflow = MagicMock()
    mlflow.start_run.return_value = nullcontext(
        SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    )
    mlflow.register_model.return_value = SimpleNamespace(version="1")
    mlflow.models.infer_signature.return_value = "inferred-signature"
    registry_client = MagicMock()
    workspace_client = MagicMock()
    predictor = MagicMock()
    predictor.predict.return_value = {"prediction": 2.0, "metadata": {}}
    input_example = {"feature": [1.0]}

    publisher = UnityCatalogPublisher(
        mlflow=mlflow,
        registry_client=registry_client,
        workspace_client=workspace_client,
    )
    publisher.publish(
        config=runtime_config,
        model=model,
        predictor=predictor,
        artifact=artifact,
        python_model=object(),
        pip_requirements=["numpy"],
        input_example=input_example,
        mlflow_experiment_name="/Users/owner@example.com/test/run_0",
    )

    predictor.predict.assert_called_once_with(input_example, return_format="base")
    mlflow.models.infer_signature.assert_called_once_with(
        input_example, {"prediction": 2.0, "metadata": {}}
    )


def test_publish_rejects_artifact_without_local_path(
    runtime_config: Config, model: DummyModel
) -> None:
    publisher = UnityCatalogPublisher(
        mlflow=MagicMock(),
        registry_client=MagicMock(),
        workspace_client=MagicMock(),
    )
    artifact = ResolvedModelArtifact(
        format="pytorch-module",
        source_uri="memory://model",
        payload=object(),
    )

    with pytest.raises(ValueError, match="local_path"):
        publisher.publish(
            config=runtime_config,
            model=model,
            predictor=MagicMock(),
            artifact=artifact,
            python_model=object(),
            pip_requirements=[],
            input_example={},
            signature="signature",
        )
