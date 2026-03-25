"""Pulumi infrastructure for the Bovi dairy analytics platform.

Creates the following Azure resources:
  - Resource Group (existing, looked up)
  - Storage Account (backing store for Function Apps)
  - Log Analytics Workspace
  - Application Insights (monitoring)
  - App Service Plan (Linux Consumption Y1 tier)
  - Function App: lactation-curves (classical curve fitting + milkbot)
  - Function App: lactation-autoencoder (TF model predictions)
  - PostgreSQL Flexible Server (central API persistence)
"""

import hashlib

import pulumi
from pulumi_azure_native import (
    applicationinsights,
    dbforpostgresql,
    operationalinsights,
    resources,
    storage,
    web,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = pulumi.Config()
stack = pulumi.get_stack()

location = config.require("location")

prefix = f"bovi-{stack}"

tags = {
    "Environment": stack,
    "Project": "Bovi",
    "Owner": "Bovi-Analytics",
    "ManagedBy": "Pulumi",
}

# ---------------------------------------------------------------------------
# Resource Group (existing - looked up from Pulumi config)
# ---------------------------------------------------------------------------
subscription_id = config.require("subscriptionId")
suffix = hashlib.md5(subscription_id.encode()).hexdigest()[:6]
resource_group_name = config.require("resourceGroup")

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
# Storage Account
# ---------------------------------------------------------------------------
storage_account = storage.StorageAccount(
    "storage-account",
    resource_group_name=resource_group.name,
    location=location,
    account_name=f"{prefix.replace('-', '')}{suffix}",
    sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),
    kind=storage.Kind.STORAGE_V2,
    access_tier=storage.AccessTier.HOT,
    enable_https_traffic_only=True,
    minimum_tls_version=storage.MinimumTlsVersion.TLS1_2,
    allow_blob_public_access=False,
    tags=tags,
)

storage_keys = pulumi.Output.all(resource_group.name, storage_account.name).apply(
    lambda args: storage.list_storage_account_keys(
        resource_group_name=args[0],
        account_name=args[1],
    )
)
primary_storage_key = storage_keys.keys[0].value

storage_connection_string = pulumi.Output.all(storage_account.name, primary_storage_key).apply(
    lambda args: (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={args[0]};"
        f"AccountKey={args[1]};"
        f"EndpointSuffix=core.windows.net"
    )
)

# ---------------------------------------------------------------------------
# Log Analytics Workspace
# ---------------------------------------------------------------------------
log_analytics = operationalinsights.Workspace(
    "log-analytics",
    resource_group_name=resource_group.name,
    location=location,
    workspace_name=f"{prefix}-logs",
    sku=operationalinsights.WorkspaceSkuArgs(name="PerGB2018"),
    retention_in_days=31,
    tags=tags,
)

# ---------------------------------------------------------------------------
# Application Insights
# ---------------------------------------------------------------------------
app_insights = applicationinsights.Component(
    "app-insights",
    resource_group_name=resource_group.name,
    location=location,
    resource_name_=f"{prefix}-insights",
    kind="web",
    application_type=applicationinsights.ApplicationType.WEB,
    workspace_resource_id=log_analytics.id,
    retention_in_days=30,
    tags=tags,
)

# ---------------------------------------------------------------------------
# App Service Plan (Linux Consumption Y1 - shared by all Function Apps)
# ---------------------------------------------------------------------------
app_service_plan = web.AppServicePlan(
    "consumption-plan",
    resource_group_name=resource_group.name,
    location=location,
    name=f"{prefix}-plan",
    sku=web.SkuDescriptionArgs(
        name="Y1",
        tier="Dynamic",
    ),
    kind="linux",
    reserved=True,
    tags=tags,
)

# ---------------------------------------------------------------------------
# CORS: allow dashboard origin
# ---------------------------------------------------------------------------
dashboard_origin = config.get("dashboardOrigin") or "http://localhost:3000"

cors_settings = web.CorsSettingsArgs(
    allowed_origins=[
        "https://portal.azure.com",
        dashboard_origin,
    ],
)

# ---------------------------------------------------------------------------
# Shared Function App settings
# ---------------------------------------------------------------------------
milkbot_key = config.get_secret("milkbotKey") or ""


def _function_app_settings(
    extra_settings: list[web.NameValuePairArgs] | None = None,
) -> list[web.NameValuePairArgs]:
    """Return common app settings for Function Apps."""
    base = [
        web.NameValuePairArgs(name="AzureWebJobsStorage", value=storage_connection_string),
        web.NameValuePairArgs(name="FUNCTIONS_WORKER_RUNTIME", value="python"),
        web.NameValuePairArgs(name="FUNCTIONS_EXTENSION_VERSION", value="~4"),
        web.NameValuePairArgs(name="AzureWebJobsFeatureFlags", value="EnableWorkerIndexing"),
        web.NameValuePairArgs(
            name="APPLICATIONINSIGHTS_CONNECTION_STRING",
            value=app_insights.connection_string,
        ),
    ]
    if extra_settings:
        base.extend(extra_settings)
    return base


# ---------------------------------------------------------------------------
# Function App: lactation-curves
# ---------------------------------------------------------------------------
lactation_curves_app = web.WebApp(
    "lactation-curves-func",
    resource_group_name=resource_group.name,
    location=location,
    name=f"{prefix}-curves-{suffix}-func",
    kind="functionapp,linux",
    reserved=True,
    server_farm_id=app_service_plan.id,
    https_only=True,
    site_config=web.SiteConfigArgs(
        linux_fx_version="PYTHON|3.12",
        ftps_state=web.FtpsState.DISABLED,
        min_tls_version="1.2",
        app_settings=_function_app_settings([
            web.NameValuePairArgs(name="MILKBOT_KEY", value=milkbot_key),
        ]),
        cors=cors_settings,
    ),
    tags=tags,
)

# ---------------------------------------------------------------------------
# Function App: lactation-autoencoder
# ---------------------------------------------------------------------------
lactation_autoencoder_app = web.WebApp(
    "lactation-autoencoder-func",
    resource_group_name=resource_group.name,
    location=location,
    name=f"{prefix}-autoenc-{suffix}-func",
    kind="functionapp,linux",
    reserved=True,
    server_farm_id=app_service_plan.id,
    https_only=True,
    site_config=web.SiteConfigArgs(
        linux_fx_version="PYTHON|3.12",
        ftps_state=web.FtpsState.DISABLED,
        min_tls_version="1.2",
        app_settings=_function_app_settings(),
        cors=cors_settings,
    ),
    tags=tags,
)

# ---------------------------------------------------------------------------
# PostgreSQL Flexible Server (for the central API)
# ---------------------------------------------------------------------------
pg_admin_password = config.require_secret("pgAdminPassword")

postgres_server = dbforpostgresql.Server(
    "postgres-server",
    resource_group_name=resource_group.name,
    location=location,
    server_name=f"{prefix}-{suffix}-pg",
    version=dbforpostgresql.ServerVersion.SERVER_VERSION_16,
    sku=dbforpostgresql.SkuArgs(
        name="Standard_B1ms",
        tier=dbforpostgresql.SkuTier.BURSTABLE,
    ),
    storage=dbforpostgresql.StorageArgs(
        storage_size_gb=32,
    ),
    administrator_login="boviadmin",
    administrator_login_password=pg_admin_password,
    tags=tags,
)

postgres_db = dbforpostgresql.Database(
    "postgres-database",
    resource_group_name=resource_group.name,
    server_name=postgres_server.name,
    database_name="bovi",
    charset="UTF8",
    collation="en_US.utf8",
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
pulumi.export("resource_group_name", resource_group.name)
pulumi.export("lactation_curves_app_name", lactation_curves_app.name)
pulumi.export(
    "lactation_curves_app_url",
    lactation_curves_app.default_host_name.apply(lambda host: f"https://{host}"),
)
pulumi.export("lactation_autoencoder_app_name", lactation_autoencoder_app.name)
pulumi.export(
    "lactation_autoencoder_app_url",
    lactation_autoencoder_app.default_host_name.apply(lambda host: f"https://{host}"),
)
pulumi.export("postgres_server_name", postgres_server.name)
pulumi.export(
    "postgres_fqdn",
    postgres_server.fully_qualified_domain_name,
)
pulumi.export("app_insights_name", app_insights.name)
pulumi.export("storage_account_name", storage_account.name)
