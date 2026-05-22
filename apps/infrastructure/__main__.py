"""Bovi platform Azure deployment.

Resources deployed:
  - Storage Account + Azure Files share (SQLite persistence for central API)
  - Log Analytics Workspace (shared backend for all App Insights)
  - App Service Plan (shared consumption plan for all Function Apps)
  - lactation-curves Function App + dedicated App Insights
  - lactation-autoencoder Function App + dedicated App Insights
  - Container Apps Environment + bovi-api + bovi-dashboard Container Apps
  - dedicated App Insights for the central API
"""

import hashlib
import os

import pulumi
import pulumi_azure_native.resources as resources
import pulumi_azure_native.web as web
from bovi_infra.resources.container_app import (
    AZURE_FILES_STORAGE_NAME,
    DATA_MOUNT_PATH,
    DATA_VOLUME_NAME,
    SQLITE_DATABASE_URL,
    ContainerAppArgs,
    StatelessContainerAppArgs,
    create_container_app,
    create_stateless_container_app,
)
from bovi_infra.resources.container_app_environment import (
    ContainerAppEnvironmentArgs,
    create_container_app_environment,
)
from bovi_infra.resources.container_app_job import (
    ContainerAppJobArgs,
    ContainerAppJobVolumeArgs,
    create_container_app_job,
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
api_image = (
    os.getenv("API_IMAGE")
    or config.get("apiImage")
    or "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
)
dashboard_image = (
    os.getenv("DASHBOARD_IMAGE")
    or config.get("dashboardImage")
    or "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
)
ghcr_username = os.getenv("GHCR_USERNAME")
ghcr_token = os.getenv("GHCR_TOKEN")
if bool(ghcr_username) != bool(ghcr_token):
    raise ValueError("GHCR_USERNAME and GHCR_TOKEN must be set together")

icar_storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME_ICAR")
icar_storage_account_key = os.getenv("STORAGE_ACCOUNT_KEY_ICAR")
icar_storage_container = os.getenv("STORAGE_ACCOUNT_CONTAINER_ICAR")
if not icar_storage_account_name or not icar_storage_account_key or not icar_storage_container:
    raise ValueError(
        "STORAGE_ACCOUNT_NAME_ICAR, STORAGE_ACCOUNT_KEY_ICAR, and "
        "STORAGE_ACCOUNT_CONTAINER_ICAR must be set for preset dataset blob access"
    )

# Storage accounts must be globally unique — derive a short suffix from the subscription ID
storage_suffix = hashlib.md5(subscription_id.encode()).hexdigest()[:6]
tags = {"environment": stack, "project": "bovi"}

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
storage_result = create_storage_account(
    "storage-account",
    StorageAccountArgs(
        resource_group_name=resource_group.name,
        location=location,
        account_name=f"bovi{stack}{storage_suffix}"[:24],
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
# Shared observability backend
# ---------------------------------------------------------------------------
log_analytics_result = create_log_analytics(
    "log-analytics",
    LogAnalyticsArgs(
        resource_group_name=resource_group.name,
        location=location,
        workspace_name=f"bovi-logs-{stack}",
        retention_in_days=31,
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# App Service Plan (shared by all Function Apps)
# ---------------------------------------------------------------------------
plan_result = create_app_service_plan(
    "functions-plan",
    AppServicePlanArgs(
        resource_group_name=resource_group.name,
        location=location,
        plan_name=f"functions-plan-{stack}",
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Lactation Curves Function App
# ---------------------------------------------------------------------------
curves_insights_result = create_app_insights(
    "curves-insights",
    AppInsightsArgs(
        resource_group_name=resource_group.name,
        location=location,
        resource_name=f"curves-insights-{stack}",
        workspace_resource_id=log_analytics_result.workspace.id,
        tags=tags,
    ),
)

curves_result = create_function_app(
    "curves-func",
    FunctionAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=f"curves-{stack}",
        server_farm_id=plan_result.plan.id,
        storage_connection_string=storage_result.connection_string,
        app_insights_connection_string=curves_insights_result.connection_string,
        cors_origins=[dashboard_origin],
        extra_app_settings=[
            web.NameValuePairArgs(name="MILKBOT_KEY", value=milkbot_key),
        ],
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Lactation Autoencoder Function App
# ---------------------------------------------------------------------------
autoencoder_insights_result = create_app_insights(
    "autoencoder-insights",
    AppInsightsArgs(
        resource_group_name=resource_group.name,
        location=location,
        resource_name=f"autoencoder-insights-{stack}",
        workspace_resource_id=log_analytics_result.workspace.id,
        tags=tags,
    ),
)

autoencoder_result = create_function_app(
    "autoencoder-func",
    FunctionAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=f"autoencoder-{stack}",
        server_farm_id=plan_result.plan.id,
        storage_connection_string=storage_result.connection_string,
        app_insights_connection_string=autoencoder_insights_result.connection_string,
        cors_origins=[dashboard_origin],
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Container App Environment + central FastAPI
# ---------------------------------------------------------------------------
api_insights_result = create_app_insights(
    "api-insights",
    AppInsightsArgs(
        resource_group_name=resource_group.name,
        location=location,
        resource_name=f"api-insights-{stack}",
        workspace_resource_id=log_analytics_result.workspace.id,
        tags=tags,
    ),
)

cae_result = create_container_app_environment(
    "container-app-env",
    ContainerAppEnvironmentArgs(
        resource_group_name=resource_group.name,
        location=location,
        env_name=f"bovi-env-{stack}",
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
        app_name=f"bovi-api-{stack}",
        environment_id=cae_result.id,
        environment_name=cae_result.name,
        image=api_image,
        port=8000,
        storage_account_name=storage_result.account.name,
        storage_account_key=storage_result.primary_key,
        files_share_name="bovidata",
        registry_server="ghcr.io" if ghcr_username and ghcr_token else None,
        registry_username=ghcr_username,
        registry_password=pulumi.Output.secret(ghcr_token) if ghcr_token else None,
        env={
            "APPLICATIONINSIGHTS_CONNECTION_STRING": api_insights_result.connection_string,
            "STORAGE_ACCOUNT_NAME_ICAR": icar_storage_account_name,
            "STORAGE_ACCOUNT_CONTAINER_ICAR": icar_storage_container,
        },
        secret_env={"STORAGE_ACCOUNT_KEY_ICAR": icar_storage_account_key},
        tags=tags,
    ),
)

api_migration_job_name = f"bovi-api-migrate-{stack}"
api_migration_job_result = create_container_app_job(
    "bovi-api-migrate",
    ContainerAppJobArgs(
        resource_group_name=resource_group.name,
        location=location,
        job_name=api_migration_job_name,
        environment_id=cae_result.id,
        image=api_image,
        command=["run-migrations"],
        container_name="bovi-api-migrate",
        env={
            "DATABASE_URL": SQLITE_DATABASE_URL,
            "APPLICATIONINSIGHTS_CONNECTION_STRING": api_insights_result.connection_string,
        },
        volumes=[
            ContainerAppJobVolumeArgs(
                name=DATA_VOLUME_NAME,
                storage_name=AZURE_FILES_STORAGE_NAME,
                mount_path=DATA_MOUNT_PATH,
            )
        ],
        registry_server="ghcr.io" if ghcr_username and ghcr_token else None,
        registry_username=ghcr_username,
        registry_password=pulumi.Output.secret(ghcr_token) if ghcr_token else None,
        depends_on=[api_result.environment_storage],
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Dashboard frontend
# ---------------------------------------------------------------------------
dashboard_result = create_stateless_container_app(
    "bovi-dashboard",
    StatelessContainerAppArgs(
        resource_group_name=resource_group.name,
        location=location,
        app_name=f"bovi-dashboard-{stack}",
        environment_id=cae_result.id,
        image=dashboard_image,
        port=3000,
        registry_server="ghcr.io" if ghcr_username and ghcr_token else None,
        registry_username=ghcr_username,
        registry_password=pulumi.Output.secret(ghcr_token) if ghcr_token else None,
        env={
            "NODE_ENV": "production",
            "NEXT_PUBLIC_API_URL": api_result.url,
        },
        tags=tags,
    ),
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
pulumi.export("resource_group_name", resource_group.name)
pulumi.export("storage_account_name", storage_result.account.name)
pulumi.export("curves_app_name", curves_result.app.name)
pulumi.export("curves_app_url", curves_result.url)
pulumi.export("autoencoder_app_name", autoencoder_result.app.name)
pulumi.export("autoencoder_app_url", autoencoder_result.url)
pulumi.export("api_url", api_result.url)
pulumi.export("api_migration_job_name", api_migration_job_result.name)
pulumi.export("dashboard_app_name", dashboard_result.container_app.name)
pulumi.export("dashboard_url", dashboard_result.url)
