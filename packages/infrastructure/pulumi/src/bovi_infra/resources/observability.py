"""Observability resources: Log Analytics Workspace and Application Insights."""

from dataclasses import dataclass

import pulumi
import pulumi_azure_native.applicationinsights as applicationinsights
import pulumi_azure_native.operationalinsights as operationalinsights

from ..types import ResourceTags


@dataclass
class LogAnalyticsArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    workspace_name: str
    retention_in_days: int = 31
    tags: ResourceTags | None = None


@dataclass
class LogAnalyticsResult:
    workspace: operationalinsights.Workspace
    customer_id: pulumi.Output[str]
    shared_key: pulumi.Output[str]


def create_log_analytics(name: str, args: LogAnalyticsArgs) -> LogAnalyticsResult:
    workspace = operationalinsights.Workspace(
        name,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name,
        location=args.location,
        sku=operationalinsights.WorkspaceSkuArgs(name="PerGB2018"),
        retention_in_days=args.retention_in_days,
        tags=args.tags,
    )
    shared_keys = operationalinsights.get_shared_keys_output(
        resource_group_name=args.resource_group_name,
        workspace_name=workspace.name,
    )
    return LogAnalyticsResult(
        workspace=workspace,
        customer_id=workspace.customer_id,
        shared_key=shared_keys.primary_shared_key,
    )


@dataclass
class AppInsightsArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    resource_name: str
    workspace_resource_id: pulumi.Input[str]
    retention_in_days: int = 30
    tags: ResourceTags | None = None


@dataclass
class AppInsightsResult:
    component: applicationinsights.Component
    connection_string: pulumi.Output[str]
    instrumentation_key: pulumi.Output[str]


def create_app_insights(name: str, args: AppInsightsArgs) -> AppInsightsResult:
    component = applicationinsights.Component(
        name,
        resource_group_name=args.resource_group_name,
        resource_name_=args.resource_name,
        location=args.location,
        kind="web",
        application_type=applicationinsights.ApplicationType.WEB,
        workspace_resource_id=args.workspace_resource_id,
        retention_in_days=args.retention_in_days,
        tags=args.tags,
    )
    return AppInsightsResult(
        component=component,
        connection_string=component.connection_string,
        instrumentation_key=component.instrumentation_key,
    )
