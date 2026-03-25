import os
import tomllib

import requests

from bovi_core.utils.dbfs_utils import with_dbutils
from bovi_core.utils.path_utils import (
    get_project_paths,
    get_project_root,
    get_project_src,
    get_run_config_path,
    read_project_name_from_toml,
)

# Re-export path utilities for backwards compatibility
__all__ = [
    "detect_environment",
    "get_toml_data",
    "get_project_paths",
    "get_project_root",
    "get_project_src",
    "read_project_name_from_toml",
    "get_run_config_path",
    "call_databricks_api",
    "create_secret_scope",
    "add_secret_to_scope",
    "list_secret_scopes",
    "list_secrets_in_scope",
    "delete_secret",
    "delete_secret_scope",
]


def detect_environment():
    """Automatically detect the current development environment"""
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return "databricks"
    elif os.path.exists("/dbfs"):
        return "databricks"
    elif "DATABRICKS_CONNECT" in os.environ:
        # Only use vscode_remote if DATABRICKS_CONNECT is explicitly set
        return "vscode_remote"
    elif os.environ.get("SPARK_REMOTE") and os.environ.get("DATABRICKS_HOST"):
        # Only use vscode_remote if both SPARK_REMOTE and DATABRICKS_HOST are set
        # This indicates a proper Databricks Connect setup
        return "vscode_remote"
    else:
        return "local"


def get_toml_data(project_file_path: str):
    # Load TOML (project config)
    if not os.path.exists(project_file_path):
        error_msg = (
            f"\n❌ Project file not found: {project_file_path}\n\n"
            f"💡 Make sure your project has a pyproject.toml at its root."
        )
        raise FileNotFoundError(error_msg)
    try:
        with open(project_file_path, "rb") as f:
            project_data = tomllib.load(f)
            return project_data
    except Exception as e:
        raise RuntimeError(f"Error loading project file: {e}")




@with_dbutils()
def call_databricks_api(method, endpoint_url, json_payload=None, dbutils=None):
    """
    Makes authenticated requests to Databricks REST API
    """
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    DATABRICKS_TOKEN = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    )

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        if method.upper() == "GET":
            response = requests.get(endpoint_url, headers=headers, json=json_payload)
        elif method.upper() == "POST":
            response = requests.post(endpoint_url, headers=headers, json=json_payload)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json() if response.content else None
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e.response.status_code} {e.response.text}")
        raise
    except Exception as e:
        print(f"General Error: {str(e)}")
        raise


def _get_workspace_url():
    """Get workspace URL directly from Spark"""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = SparkSession.builder.getOrCreate()
        return spark.conf.get("spark.databricks.workspaceUrl")
    except Exception:
        return None


@with_dbutils()
def create_secret_scope(scope_name: str, dbutils=None) -> bool:
    """Create a new Databricks secret scope"""
    scope_payload = {"scope": scope_name}
    workspace_url = _get_workspace_url()
    if not workspace_url:
        print("❌ Could not retrieve workspace URL")
        return False

    api_url = f"https://{workspace_url}/api/2.0/secrets/scopes/create"

    try:
        call_databricks_api("POST", api_url, scope_payload, dbutils=dbutils)
        print(f"✅ Secret scope '{scope_name}' created successfully")
        return True
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"ℹ️ Secret scope '{scope_name}' already exists")
            return True
        else:
            print(f"❌ Failed to create scope '{scope_name}': {str(e)}")
            return False


@with_dbutils()
def add_secret_to_scope(scope_name: str, key_name: str, secret_value: str, dbutils=None) -> bool:
    """Add a secret to an existing Databricks secret scope"""
    secret_payload = {
        "scope": scope_name,
        "key": key_name,
        "string_value": secret_value,
    }
    workspace_url = _get_workspace_url()
    if not workspace_url:
        print("❌ Could not retrieve workspace URL")
        return False

    api_url = f"https://{workspace_url}/api/2.0/secrets/put"

    try:
        call_databricks_api("POST", api_url, secret_payload, dbutils=dbutils)
        print(f"✅ Secret '{key_name}' added to scope '{scope_name}' successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to add secret '{key_name}' to scope '{scope_name}': {str(e)}")
        return False


@with_dbutils()
def list_secret_scopes(dbutils=None) -> list:
    """List all available secret scopes"""
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        scopes = dbutils.secrets.listScopes()
        return [scope.name for scope in scopes]
    except Exception as e:
        print(f"❌ Failed to list secret scopes: {str(e)}")
        return []


@with_dbutils()
def list_secrets_in_scope(scope_name: str, dbutils=None) -> list:
    """List all secrets in a given scope"""
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        secrets = dbutils.secrets.list(scope_name)
        return [secret.key for secret in secrets]
    except Exception as e:
        print(f"❌ Failed to list secrets in scope '{scope_name}': {str(e)}")
        return []


@with_dbutils()
def delete_secret(scope_name: str, key_name: str, dbutils=None) -> bool:
    """Delete a specific secret from a scope"""
    payload = {"scope": scope_name, "key": key_name}
    workspace_url = _get_workspace_url()
    if not workspace_url:
        print("❌ Could not retrieve workspace URL")
        return False

    api_url = f"https://{workspace_url}/api/2.0/secrets/delete"

    try:
        call_databricks_api("POST", api_url, payload, dbutils=dbutils)
        print(f"✅ Secret '{key_name}' deleted from scope '{scope_name}' successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to delete secret '{key_name}' from scope '{scope_name}': {str(e)}")
        return False


@with_dbutils()
def delete_secret_scope(scope_name: str, dbutils=None) -> bool:
    """Delete an entire secret scope"""
    payload = {"scope": scope_name}
    workspace_url = _get_workspace_url()
    if not workspace_url:
        print("❌ Could not retrieve workspace URL")
        return False

    api_url = f"https://{workspace_url}/api/2.0/secrets/scopes/delete"

    try:
        call_databricks_api("POST", api_url, payload, dbutils=dbutils)
        print(f"✅ Secret scope '{scope_name}' deleted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to delete scope '{scope_name}': {str(e)}")
        return False
