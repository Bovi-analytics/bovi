"""Generic Azure Container App factory for multi-container workloads."""

from dataclasses import dataclass

import pulumi
import pulumi_azure_native.app as app

from ..types import ResourceTags


@dataclass
class ContainerSpec:
    name: str
    image: pulumi.Input[str]
    cpu: float
    memory: str
    env: dict[str, pulumi.Input[str]] | None = None
    secret_env: dict[str, str] | None = None


@dataclass
class MultiContainerAppArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    app_name: str
    environment_id: pulumi.Input[str]
    containers: list[ContainerSpec]
    registry_server: str | None = None
    registry_username: pulumi.Input[str] | None = None
    registry_password: pulumi.Input[str] | None = None
    registry_password_secret_name: str = "registry-password"
    secrets: dict[str, pulumi.Input[str]] | None = None
    workload_profile_name: str = "Consumption"
    min_replicas: int = 1
    max_replicas: int = 1
    tags: ResourceTags | None = None


@dataclass
class MultiContainerAppResult:
    container_app: app.ContainerApp
    id: pulumi.Output[str]
    name: pulumi.Output[str]


def _registry_args(
    args: MultiContainerAppArgs,
) -> tuple[list[app.SecretArgs], list[app.RegistryCredentialsArgs]]:
    secrets = [
        app.SecretArgs(name=name, value=value) for name, value in (args.secrets or {}).items()
    ]
    registries: list[app.RegistryCredentialsArgs] = []
    if args.registry_server and args.registry_username and args.registry_password:
        secrets.append(
            app.SecretArgs(
                name=args.registry_password_secret_name,
                value=args.registry_password,
            )
        )
        registries.append(
            app.RegistryCredentialsArgs(
                server=args.registry_server,
                username=args.registry_username,
                password_secret_ref=args.registry_password_secret_name,
            )
        )
    return secrets, registries


def create_multi_container_app(
    name: str,
    args: MultiContainerAppArgs,
) -> MultiContainerAppResult:
    secrets, registries = _registry_args(args)
    container_app = app.ContainerApp(
        name,
        resource_group_name=args.resource_group_name,
        container_app_name=args.app_name,
        environment_id=args.environment_id,
        location=args.location,
        configuration=app.ConfigurationArgs(
            active_revisions_mode="Single",
            registries=registries or None,
            secrets=secrets or None,
        ),
        template=app.TemplateArgs(
            containers=[
                app.ContainerArgs(
                    name=container.name,
                    image=container.image,
                    resources=app.ContainerResourcesArgs(
                        cpu=container.cpu,
                        memory=container.memory,
                    ),
                    env=[
                        *[
                            app.EnvironmentVarArgs(name=key, value=value)
                            for key, value in (container.env or {}).items()
                        ],
                        *[
                            app.EnvironmentVarArgs(name=key, secret_ref=secret_name)
                            for key, secret_name in (container.secret_env or {}).items()
                        ],
                    ],
                )
                for container in args.containers
            ],
            scale=app.ScaleArgs(
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
            ),
        ),
        workload_profile_name=(
            None if args.workload_profile_name == "Consumption" else args.workload_profile_name
        ),
        tags=args.tags,
    )
    return MultiContainerAppResult(
        container_app=container_app,
        id=container_app.id,
        name=container_app.name,
    )
