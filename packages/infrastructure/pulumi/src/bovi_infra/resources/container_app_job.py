"""Reusable Container Apps Job resources."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pulumi
import pulumi_azure_native.app as app

from ..types import ResourceTags


@dataclass
class ContainerAppJobVolumeArgs:
    name: str
    storage_name: str
    mount_path: str
    storage_type: str = "AzureFile"


@dataclass
class ContainerAppJobArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    job_name: str
    environment_id: pulumi.Input[str]
    image: pulumi.Input[str]
    command: list[str]
    args: list[str] = field(default_factory=list)
    env: dict[str, pulumi.Input[str]] = field(default_factory=dict)
    volumes: list[ContainerAppJobVolumeArgs] = field(default_factory=list)
    container_name: str | None = None
    cpu: float = 0.25
    memory: str = "0.5Gi"
    replica_timeout_seconds: int = 300
    replica_retry_limit: int = 0
    parallelism: int = 1
    replica_completion_count: int = 1
    registry_server: str | None = None
    registry_username: pulumi.Input[str] | None = None
    registry_password: pulumi.Input[str] | None = None
    registry_password_secret_name: str = "ghcr-token"
    depends_on: Sequence[pulumi.Resource] | None = None
    tags: ResourceTags | None = None


@dataclass
class ContainerAppJobResult:
    job: app.Job
    id: pulumi.Output[str]
    name: str


def create_container_app_job(name: str, args: ContainerAppJobArgs) -> ContainerAppJobResult:
    env_vars = [app.EnvironmentVarArgs(name=k, value=v) for k, v in args.env.items()]
    secrets: list[app.SecretArgs] = []
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

    volumes = [
        app.VolumeArgs(
            name=volume.name,
            storage_type=volume.storage_type,
            storage_name=volume.storage_name,
        )
        for volume in args.volumes
    ]
    volume_mounts = [
        app.VolumeMountArgs(
            volume_name=volume.name,
            mount_path=volume.mount_path,
        )
        for volume in args.volumes
    ]

    job = app.Job(
        name,
        resource_group_name=args.resource_group_name,
        job_name=args.job_name,
        environment_id=args.environment_id,
        location=args.location,
        configuration=app.JobConfigurationArgs(
            trigger_type="Manual",
            replica_timeout=args.replica_timeout_seconds,
            replica_retry_limit=args.replica_retry_limit,
            manual_trigger_config=app.JobConfigurationManualTriggerConfigArgs(
                parallelism=args.parallelism,
                replica_completion_count=args.replica_completion_count,
            ),
            registries=registries or None,
            secrets=secrets or None,
        ),
        template=app.JobTemplateArgs(
            containers=[
                app.ContainerArgs(
                    name=args.container_name or args.job_name,
                    image=args.image,
                    command=args.command,
                    args=args.args or None,
                    resources=app.ContainerResourcesArgs(
                        cpu=args.cpu,
                        memory=args.memory,
                    ),
                    env=env_vars,
                    volume_mounts=volume_mounts or None,
                )
            ],
            volumes=volumes or None,
        ),
        tags=args.tags,
        opts=pulumi.ResourceOptions(depends_on=args.depends_on) if args.depends_on else None,
    )

    return ContainerAppJobResult(job=job, id=job.id, name=args.job_name)
