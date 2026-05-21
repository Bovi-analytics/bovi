"""Container App resource with Azure Files volume mount for SQLite persistence."""

from dataclasses import dataclass, field

import pulumi
import pulumi_azure_native.app as app

from ..types import ResourceTags

AZURE_FILES_STORAGE_NAME = "bovifiles"
DATA_VOLUME_NAME = "data"
DATA_MOUNT_PATH = "/mnt/data"
SQLITE_DATABASE_URL = f"sqlite+aiosqlite:////{DATA_MOUNT_PATH}/bovi.db"


@dataclass
class ContainerAppArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    app_name: str
    environment_id: pulumi.Input[str]
    environment_name: pulumi.Input[str]
    image: pulumi.Input[str]
    port: int
    # Azure Files backing for SQLite
    storage_account_name: pulumi.Input[str]
    storage_account_key: pulumi.Input[str]
    files_share_name: str
    # Optional
    env: dict[str, pulumi.Input[str]] = field(default_factory=dict)
    cpu: float = 0.25
    memory: str = "0.5Gi"
    min_replicas: int = 0
    max_replicas: int = 1
    registry_server: str | None = None
    registry_username: pulumi.Input[str] | None = None
    registry_password: pulumi.Input[str] | None = None
    registry_password_secret_name: str = "ghcr-token"
    tags: ResourceTags | None = None


@dataclass
class ContainerAppResult:
    container_app: app.ContainerApp
    environment_storage: app.ManagedEnvironmentsStorage
    id: pulumi.Output[str]
    fqdn: pulumi.Output[str]
    url: pulumi.Output[str]


@dataclass
class StatelessContainerAppArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    app_name: str
    environment_id: pulumi.Input[str]
    image: pulumi.Input[str]
    port: int
    env: dict[str, pulumi.Input[str]] = field(default_factory=dict)
    cpu: float = 0.25
    memory: str = "0.5Gi"
    min_replicas: int = 0
    max_replicas: int = 1
    registry_server: str | None = None
    registry_username: pulumi.Input[str] | None = None
    registry_password: pulumi.Input[str] | None = None
    registry_password_secret_name: str = "ghcr-token"
    tags: ResourceTags | None = None


@dataclass
class StatelessContainerAppResult:
    container_app: app.ContainerApp
    id: pulumi.Output[str]
    fqdn: pulumi.Output[str]
    url: pulumi.Output[str]


def _registry_args(
    *,
    registry_server: str | None,
    registry_username: pulumi.Input[str] | None,
    registry_password: pulumi.Input[str] | None,
    registry_password_secret_name: str,
) -> tuple[list[app.SecretArgs], list[app.RegistryCredentialsArgs]]:
    secrets: list[app.SecretArgs] = []
    registries: list[app.RegistryCredentialsArgs] = []
    if registry_server and registry_username and registry_password:
        secrets.append(
            app.SecretArgs(
                name=registry_password_secret_name,
                value=registry_password,
            )
        )
        registries.append(
            app.RegistryCredentialsArgs(
                server=registry_server,
                username=registry_username,
                password_secret_ref=registry_password_secret_name,
            )
        )
    return secrets, registries


def create_container_app(name: str, args: ContainerAppArgs) -> ContainerAppResult:
    # Link the Azure Files share to the Container App Environment
    env_storage = app.ManagedEnvironmentsStorage(
        f"{name}-storage",
        resource_group_name=args.resource_group_name,
        environment_name=args.environment_name,
        storage_name=AZURE_FILES_STORAGE_NAME,
        properties=app.ManagedEnvironmentStoragePropertiesArgs(
            azure_file=app.AzureFilePropertiesArgs(
                account_name=args.storage_account_name,
                account_key=args.storage_account_key,
                share_name=args.files_share_name,
                access_mode="ReadWrite",
            )
        ),
    )

    env_vars = [
        app.EnvironmentVarArgs(name="DATABASE_URL", value=SQLITE_DATABASE_URL),
        *[app.EnvironmentVarArgs(name=k, value=v) for k, v in args.env.items()],
    ]
    secrets, registries = _registry_args(
        registry_server=args.registry_server,
        registry_username=args.registry_username,
        registry_password=args.registry_password,
        registry_password_secret_name=args.registry_password_secret_name,
    )

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
            ingress=app.IngressArgs(
                external=True,
                target_port=args.port,
                transport="http",
                allow_insecure=False,
            ),
        ),
        template=app.TemplateArgs(
            volumes=[
                app.VolumeArgs(
                    name=DATA_VOLUME_NAME,
                    storage_type="AzureFile",
                    storage_name=AZURE_FILES_STORAGE_NAME,
                )
            ],
            containers=[
                app.ContainerArgs(
                    name=args.app_name,
                    image=args.image,
                    resources=app.ContainerResourcesArgs(
                        cpu=args.cpu,
                        memory=args.memory,
                    ),
                    env=env_vars,
                    volume_mounts=[
                        app.VolumeMountArgs(
                            volume_name=DATA_VOLUME_NAME,
                            mount_path=DATA_MOUNT_PATH,
                        )
                    ],
                )
            ],
            scale=app.ScaleArgs(
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
            ),
        ),
        tags=args.tags,
        opts=pulumi.ResourceOptions(depends_on=[env_storage]),
    )

    fqdn = container_app.configuration.apply(
        lambda c: c.ingress.fqdn if c and c.ingress and c.ingress.fqdn else ""
    )
    return ContainerAppResult(
        container_app=container_app,
        environment_storage=env_storage,
        id=container_app.id,
        fqdn=fqdn,
        url=fqdn.apply(lambda h: f"https://{h}" if h else ""),
    )


def create_stateless_container_app(
    name: str,
    args: StatelessContainerAppArgs,
) -> StatelessContainerAppResult:
    env_vars = [app.EnvironmentVarArgs(name=k, value=v) for k, v in args.env.items()]
    secrets, registries = _registry_args(
        registry_server=args.registry_server,
        registry_username=args.registry_username,
        registry_password=args.registry_password,
        registry_password_secret_name=args.registry_password_secret_name,
    )

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
            ingress=app.IngressArgs(
                external=True,
                target_port=args.port,
                transport="http",
                allow_insecure=False,
            ),
        ),
        template=app.TemplateArgs(
            containers=[
                app.ContainerArgs(
                    name=args.app_name,
                    image=args.image,
                    resources=app.ContainerResourcesArgs(
                        cpu=args.cpu,
                        memory=args.memory,
                    ),
                    env=env_vars,
                )
            ],
            scale=app.ScaleArgs(
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
            ),
        ),
        tags=args.tags,
    )

    fqdn = container_app.configuration.apply(
        lambda c: c.ingress.fqdn if c and c.ingress and c.ingress.fqdn else ""
    )
    return StatelessContainerAppResult(
        container_app=container_app,
        id=container_app.id,
        fqdn=fqdn,
        url=fqdn.apply(lambda h: f"https://{h}" if h else ""),
    )
