"""Bovi platform Azure deployment.

Wires together the reusable bovi_infra resource modules into a complete
deployment for the Bovi dairy analytics platform.

Resources deployed:
  - Storage Account + Azure Files share (SQLite persistence for central API)
  - Log Analytics Workspace + Application Insights (monitoring)
  - App Service Plan + 2x Function Apps (lactation-curves, lactation-autoencoder)
  - Container Apps Environment + Container App (central FastAPI, scale-to-zero)
"""

import hashlib

import pulumi
import pulumi_azure_native.resources as resources
import pulumi_azure_native.web as web
from bovi_infra.resources.container_app import ContainerAppArgs, create_container_app
from bovi_infra.resources.container_app_environment import (
    ContainerAppEnvironmentArgs,
    create_container_app_environment,
)
from bovi_infra.resources.function_app import (
    AppServicePlanArgs,
    FunctionAppArgs,
    create_app_service_plan,
    create_function_app,
)
from bovi_infra.resources.observability import (
    AppInsightsArgs,
    LogAnalyticsArgs,
    create_app_insights,
    create_log_analytics,
)
from bovi_infra.resources.storage import (
    FilesShareArgs,
    StorageAccountArgs,
    create_files_share,
    create_storage_account,
)
from bovi_infra.utils.naming import build_resource_name, build_resource_tags

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = pulumi.Config()
stack = pulumi.get_stack()

location = config.require("location")
subscription_id = config.require("subscriptionId")
resource_group_name = config.require("resourceGroup")
dashboard_origin = config.get("dashboardOrigin") or "http://localhost:3000"
milkbot_key = config.get_secret("milkbotKey") or ""
api_image = config.get("apiImage") or "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"

# 6-char hash suffix for globally unique resource names
suffix = hashlib.md5(subscription_id.encode()).hexdigest()[:6]
prefix = f"bovi-{stack}"
tags = build_resource_tags("bovi", stack)

# ---------------------------------------------------------------------------
# Resource Group (existing)
# ---------------------------------------------------------------------------
resource_group = resources.ResourceGroup.get(
    "resource-group",
    id=pulumi.Output.concat(
        "/subscriptions/",
        subscription_id,
        "/resourceGroups/",
        resource_group_name,
    ),
)

# ---------------------------------------------------------------------------
# Storage Account + Azure Files share
# ---------------------------------------------------------------------------
storage_account_name = f"{prefix.replace('-', '')}{suffix}"[:24]

storage_result = create_storage_account(
    "storage-account",
    StorageAccountArgs(
        resource_group_name=resource_group.name,
        location=location,
        account_name=storage_account_name,
        tags=tags,
    ),
)

files_share_result = create_files_share(
    "files-share",
    FilesShareArgs(
        resource_group_name=resource_group.name,
        account_name=storage_result.account.name,
        share_name="bovidata",
        quota_gb=1,
    ),
)

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
log_analytics_result = create_log_analytics(
    "log-analytics",
    LogAnalyticsArgs(
        resource_group_name=resource_group.name,
        location=location,
        workspace_name=build_resource_name("bovi", stack, "logs"),
        retention_in_days=31,
        tags=tags,
    ),
)

app_insights_result = create_app_insights(
    "app-insights",
    AppInsightsArgs(
        resource_group_name=resource_group.name,
        location=location,
        resource_name=build_resource_name("bovi", stack, "insights"),
        workspace_resource_id=log_analytics_result.workspace.id,
        retention_in_days=30,
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Function Apps (model services)
# ---------------------------------------------------------------------------
plan_result = create_app_service_plan(
    "consumption-plan",
    AppServicePlanArgs(
        resource_group_name=resource_group.name,
        location=location,
        plan_name=build_resource_name("bovi", stack, "plan"),
        tags=tags,
    ),
)

curves_result = create_function_app(
    "lactation-curves-func",
    FunctionAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=f"{prefix}-curves-{suffix}-func",
        server_farm_id=plan_result.plan.id,
        storage_connection_string=storage_result.connection_string,
        app_insights_connection_string=app_insights_result.connection_string,
        cors_origins=[dashboard_origin],
        extra_app_settings=[
            web.NameValuePairArgs(name="MILKBOT_KEY", value=milkbot_key),
        ],
        tags=tags,
    ),
)

autoencoder_result = create_function_app(
    "lactation-autoencoder-func",
    FunctionAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=f"{prefix}-autoenc-{suffix}-func",
        server_farm_id=plan_result.plan.id,
        storage_connection_string=storage_result.connection_string,
        app_insights_connection_string=app_insights_result.connection_string,
        cors_origins=[dashboard_origin],
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Container App Environment + central FastAPI
# ---------------------------------------------------------------------------
cae_result = create_container_app_environment(
    "container-app-env",
    ContainerAppEnvironmentArgs(
        resource_group_name=resource_group.name,
        location=location,
        env_name=build_resource_name("bovi", stack, "env"),
        log_analytics_customer_id=log_analytics_result.customer_id,
        log_analytics_shared_key=log_analytics_result.shared_key,
        tags=tags,
    ),
)

api_result = create_container_app(
    "bovi-api",
    ContainerAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=build_resource_name("bovi", stack, "api"),
        environment_id=cae_result.id,
        environment_name=cae_result.name,
        image=api_image,
        port=8000,
        storage_account_name=storage_result.account.name,
        storage_account_key=storage_result.primary_key,
        files_share_name="bovidata",
        env={"APPLICATIONINSIGHTS_CONNECTION_STRING": app_insights_result.connection_string},
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
pulumi.export("resource_group_name", resource_group.name)
pulumi.export("storage_account_name", storage_result.account.name)
pulumi.export("app_insights_name", app_insights_result.component.name)
pulumi.export("lactation_curves_app_name", curves_result.app.name)
pulumi.export("lactation_curves_app_url", curves_result.url)
pulumi.export("lactation_autoencoder_app_name", autoencoder_result.app.name)
pulumi.export("lactation_autoencoder_app_url", autoencoder_result.url)
pulumi.export("api_url", api_result.url)
