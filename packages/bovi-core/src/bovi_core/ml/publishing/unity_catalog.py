"""Publish exported model artifacts to Databricks Unity Catalog."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from bovi_core.ml.models.resources import ResolvedModelArtifact
from bovi_core.ml.utils.signature_utils import output_to_serializable

if TYPE_CHECKING:
    from bovi_core.config import Config
    from bovi_core.ml.dataloaders.base import Dataset
    from bovi_core.ml.models import Model
    from bovi_core.ml.predictors import PredictorProtocol

logger = logging.getLogger(__name__)


class _RegistryClient(Protocol):
    def search_model_versions(self, filter_string: str) -> Sequence[Any]: ...

    def set_registered_model_alias(self, *, name: str, alias: str, version: str) -> None: ...

    def update_model_version(self, *, name: str, version: str, description: str) -> None: ...


class _WorkspaceAPI(Protocol):
    def get_status(self, path: str) -> object: ...

    def mkdirs(self, path: str) -> None: ...


class _WorkspaceClient(Protocol):
    workspace: _WorkspaceAPI


class UnityCatalogPublisher:
    """Publish an already exported artifact through MLflow to Unity Catalog.

    The publisher does not export models or resolve remote artifacts. Callers
    explicitly supply the runtime context used for metadata, the predictor used
    for signature inference, and a locally resolved deployment artifact.
    """

    def __init__(
        self,
        mlflow: Any | None = None,
        registry_client: _RegistryClient | None = None,
        workspace_client: _WorkspaceClient | None = None,
    ) -> None:
        self._mlflow = mlflow
        self._registry_client = registry_client
        self._workspace_client = workspace_client

    def publish(
        self,
        *,
        config: Config,
        model: Model[Any, Any],
        predictor: PredictorProtocol,
        artifact: ResolvedModelArtifact[Any],
        python_model: object,
        pip_requirements: Sequence[str],
        catalog: str = "projects",
        schema: str = "bovi_core",
        model_name: str | None = None,
        alias: str | None = None,
        dataset: Dataset | None = None,
        input_example: Any | None = None,
        signature: Any | None = None,
        description: str | None = None,
        tags: Mapping[str, object] | None = None,
        n_samples: int = 5,
        auto_increment_alias: bool = True,
        mlflow_experiment_name: str | None = None,
        config_artifact_path: Path | None = None,
        pyproject_artifact_path: Path | None = None,
    ) -> Any:
        """Log and register one local deployment artifact.

        ``artifact`` must have a local path. Resolving cloud references and
        producing framework-specific artifacts are separate responsibilities.
        """
        artifact_path = self._require_local_artifact(artifact)
        mlflow = self._get_mlflow()
        registry_client = self._get_registry_client(mlflow)

        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_tracking_uri("databricks")

        experiment_name = mlflow_experiment_name or self._generate_experiment_name(config, mlflow)
        self._ensure_workspace_path(experiment_name.rsplit("/", 1)[0])
        mlflow.set_experiment(experiment_name)

        full_model_name = generate_uc_model_name(
            config=config,
            model=model,
            catalog=catalog,
            schema=schema,
            model_name=model_name,
        )
        if alias and auto_increment_alias:
            alias = resolve_alias_version(registry_client, full_model_name, alias)

        input_example, signature = self._resolve_signature(
            mlflow=mlflow,
            predictor=predictor,
            dataset=dataset,
            input_example=input_example,
            signature=signature,
            n_samples=n_samples,
        )
        model_tags = generate_model_tags(
            config=config,
            model=model,
            artifact=artifact,
            custom_tags=tags,
        )
        artifacts = {"model_path": str(artifact_path)}
        if config_artifact_path is not None:
            artifacts["config_yaml"] = str(config_artifact_path.resolve())
        if pyproject_artifact_path is not None:
            artifacts["pyproject_toml"] = str(pyproject_artifact_path.resolve())

        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                python_model=python_model,
                name="model",
                artifacts=artifacts,
                signature=signature,
                input_example=input_example,
                pip_requirements=list(pip_requirements),
            )
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=full_model_name,
                tags=model_tags,
            )

        if alias:
            registry_client.set_registered_model_alias(
                name=full_model_name,
                alias=alias,
                version=model_version.version,
            )
        if description:
            registry_client.update_model_version(
                name=full_model_name,
                version=model_version.version,
                description=description,
            )
        return model_version

    @staticmethod
    def _require_local_artifact(artifact: ResolvedModelArtifact[Any]) -> Path:
        if artifact.local_path is None:
            raise ValueError("Unity Catalog publishing requires an artifact with a local_path")
        path = artifact.local_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {path}")
        return path

    def _get_mlflow(self) -> Any:
        if self._mlflow is None:
            try:
                import mlflow
            except ImportError as exc:
                raise ImportError("mlflow is required for Unity Catalog publishing") from exc
            self._mlflow = mlflow
        return self._mlflow

    def _get_registry_client(self, mlflow: Any) -> _RegistryClient:
        if self._registry_client is None:
            self._registry_client = mlflow.MlflowClient()
        return self._registry_client

    def _ensure_workspace_path(self, path: str) -> None:
        workspace_client = self._workspace_client
        if workspace_client is None:
            try:
                from databricks.sdk import WorkspaceClient
            except ImportError as exc:
                raise ImportError(
                    "databricks-sdk is required for Unity Catalog publishing"
                ) from exc
            workspace_client = WorkspaceClient()
            self._workspace_client = workspace_client

        current_path = ""
        for part in path.strip("/").split("/"):
            current_path = f"{current_path}/{part}"
            try:
                workspace_client.workspace.get_status(current_path)
            except Exception:
                workspace_client.workspace.mkdirs(current_path)

    @staticmethod
    def _generate_experiment_name(config: Config, mlflow: Any) -> str:
        experiment_name = getattr(config.experiment, "experiment_name", "default_experiment")
        experiment_version = getattr(config.experiment, "experiment_version", "v1")
        base_path = (
            f"/Users/{config.author_email}/projects/{config.project.name}"
            f"/data/experiments/{experiment_name}/versions/{experiment_version}"
        )
        return f"{base_path}/run_{_get_next_experiment_run(mlflow, base_path)}"

    @staticmethod
    def _resolve_signature(
        *,
        mlflow: Any,
        predictor: PredictorProtocol,
        dataset: Dataset | None,
        input_example: Any | None,
        signature: Any | None,
        n_samples: int,
    ) -> tuple[Any, Any]:
        if input_example is None:
            if dataset is None:
                raise ValueError("dataset or input_example is required for publishing")
            input_example = dataset.get_input_example(n_samples=n_samples, batch=True)

        if signature is None:
            predictions = predictor.predict(input_example, return_format="base")
            signature = mlflow.models.infer_signature(
                input_example, output_to_serializable(predictions)
            )
        return input_example, signature


def generate_uc_model_name(
    *,
    config: Config,
    model: Model[Any, Any],
    catalog: str,
    schema: str,
    model_name: str | None,
) -> str:
    """Build a three-level Unity Catalog model name."""
    if model_name is None:
        model_key = getattr(type(model.config), "model_key", "model")
        model_name = f"{config.project.name}_{model_key}"
    return f"{catalog}.{schema}.{model_name}"


def generate_model_tags(
    *,
    config: Config,
    model: Model[Any, Any],
    artifact: ResolvedModelArtifact[Any],
    custom_tags: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build searchable metadata without reading storage state from the model."""
    tags: dict[str, object] = {
        "project": config.project.name,
        "model_key": getattr(type(model.config), "model_key", "model"),
        "framework": model.config.framework,
        "artifact_format": artifact.format,
        "artifact_uri": artifact.source_uri,
    }
    experiment_name = getattr(config.experiment, "experiment_name", None)
    experiment_version = getattr(config.experiment, "experiment_version", None)
    if experiment_name is not None:
        tags["experiment"] = experiment_name
    if experiment_version is not None:
        tags["experiment_version"] = experiment_version
    if custom_tags:
        tags.update(custom_tags)
    return tags


def resolve_alias_version(client: _RegistryClient, full_model_name: str, alias: str) -> str:
    """Return the next free alias while preserving the existing alias scheme."""
    try:
        versions = client.search_model_versions(filter_string=f"name='{full_model_name}'")
    except Exception:
        return alias

    existing_aliases = {
        existing_alias
        for version in versions
        for existing_alias in (getattr(version, "aliases", None) or ())
    }
    candidate = alias
    while candidate in existing_aliases:
        version_match = re.fullmatch(r"v?(\d+)\.(\d+)", candidate)
        if version_match:
            candidate = f"v{int(version_match.group(1))}.{int(version_match.group(2)) + 1}"
            continue

        suffix_match = re.fullmatch(r"(.+)_v(\d+)", candidate)
        if suffix_match:
            candidate = f"{suffix_match.group(1)}_v{int(suffix_match.group(2)) + 1}"
        else:
            candidate = f"{candidate}_v2"
    return candidate


def _get_next_experiment_run(mlflow: Any, experiment_base_path: str) -> int:
    try:
        experiments = mlflow.search_experiments(
            filter_string=f"name LIKE '{experiment_base_path}/run_%'"
        )
    except Exception:
        return 0

    run_numbers: list[int] = []
    for experiment in experiments:
        match = re.search(r"/run_(\d+)$", experiment.name)
        if match:
            run_numbers.append(int(match.group(1)))
    return max(run_numbers, default=-1) + 1
