"""Function App resources: App Service Plan and Azure Function Apps."""

import json
from dataclasses import dataclass, field

import pulumi
import pulumi_azure_native.web as web

from ..types import ResourceTags


@dataclass
class AppServicePlanArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    plan_name: str
    tags: ResourceTags | None = None


@dataclass
class AppServicePlanResult:
    plan: web.AppServicePlan


def create_app_service_plan(name: str, args: AppServicePlanArgs) -> AppServicePlanResult:
    plan = web.AppServicePlan(
        name,
        resource_group_name=args.resource_group_name,
        location=args.location,
        name=args.plan_name,
        sku=web.SkuDescriptionArgs(name="Y1", tier="Dynamic"),
        kind="linux",
        reserved=True,
        tags=args.tags,
    )
    return AppServicePlanResult(plan=plan)


@dataclass
class FunctionAppArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    app_name: str
    server_farm_id: pulumi.Input[str]
    storage_connection_string: pulumi.Input[str]
    app_insights_connection_string: pulumi.Input[str]
    cors_origins: list[str] = field(default_factory=list)
    extra_app_settings: list[web.NameValuePairArgs] = field(default_factory=list)
    tags: ResourceTags | None = None


@dataclass
class FunctionAppResult:
    app: web.WebApp
    url: pulumi.Output[str]


def create_function_app(name: str, args: FunctionAppArgs) -> FunctionAppResult:
    app_settings: list[web.NameValuePairArgs] = [
        web.NameValuePairArgs(name="AzureWebJobsStorage", value=args.storage_connection_string),
        web.NameValuePairArgs(name="FUNCTIONS_WORKER_RUNTIME", value="python"),
        web.NameValuePairArgs(name="FUNCTIONS_EXTENSION_VERSION", value="~4"),
        web.NameValuePairArgs(name="AzureWebJobsFeatureFlags", value="EnableWorkerIndexing"),
        web.NameValuePairArgs(
            name="APPLICATIONINSIGHTS_CONNECTION_STRING",
            value=args.app_insights_connection_string,
        ),
        *args.extra_app_settings,
    ]
    if args.cors_origins:
        app_settings.append(
            web.NameValuePairArgs(
                name="CORS_ORIGINS",
                value=json.dumps(args.cors_origins),
            )
        )
    cors = web.CorsSettingsArgs(allowed_origins=["https://portal.azure.com", *args.cors_origins])
    func_app = web.WebApp(
        name,
        resource_group_name=args.resource_group_name,
        location=args.location,
        name=args.app_name,
        kind="functionapp,linux",
        reserved=True,
        server_farm_id=args.server_farm_id,
        https_only=True,
        site_config=web.SiteConfigArgs(
            linux_fx_version="PYTHON|3.12",
            ftps_state=web.FtpsState.DISABLED,
            min_tls_version="1.2",
            app_settings=app_settings,
            cors=cors,
        ),
        tags=args.tags,
    )
    url = func_app.default_host_name.apply(lambda h: f"https://{h}")
    return FunctionAppResult(app=func_app, url=url)
