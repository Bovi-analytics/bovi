"""Container Apps Environment resource."""

from dataclasses import dataclass

import pulumi
import pulumi_azure_native.app as app

from ..types import ResourceTags


@dataclass
class ContainerAppEnvironmentArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    env_name: str
    log_analytics_customer_id: pulumi.Input[str] | None = None
    log_analytics_shared_key: pulumi.Input[str] | None = None
    tags: ResourceTags | None = None


@dataclass
class ContainerAppEnvironmentResult:
    environment: app.ManagedEnvironment
    id: pulumi.Output[str]
    name: pulumi.Output[str]
    default_domain: pulumi.Output[str]


def create_container_app_environment(
    name: str,
    args: ContainerAppEnvironmentArgs,
) -> ContainerAppEnvironmentResult:
    logs_config = None
    if args.log_analytics_customer_id and args.log_analytics_shared_key:
        logs_config = app.AppLogsConfigurationArgs(
            destination="log-analytics",
            log_analytics_configuration=app.LogAnalyticsConfigurationArgs(
                customer_id=args.log_analytics_customer_id,
                shared_key=args.log_analytics_shared_key,
            ),
        )

    environment = app.ManagedEnvironment(
        name,
        resource_group_name=args.resource_group_name,
        environment_name=args.env_name,
        location=args.location,
        app_logs_configuration=logs_config,
        tags=args.tags,
    )
    return ContainerAppEnvironmentResult(
        environment=environment,
        id=environment.id,
        name=environment.name,
        default_domain=environment.default_domain,
    )
