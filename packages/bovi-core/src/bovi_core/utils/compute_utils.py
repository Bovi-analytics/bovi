import json
import os
from pathlib import Path

from bovi_core.config import Config, with_config
from bovi_core.utils.env_utils import call_databricks_api


@with_config
def get_cluster_name_from_toml(config: Config):
    try:
        return config.project.databricks.cluster_name
    except AttributeError:
        return None


def load_cluster_config(cluster_name):
    compute_dir = Path(__file__).parent.parent.parent / "compute"
    for file in compute_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            if (
                data.get("cluster_name") == cluster_name
                or data.get("spec", {}).get("cluster_name") == cluster_name
            ):
                return data
    raise FileNotFoundError(f"No cluster config found for cluster_name: {cluster_name}")


@with_config
def get_workspace_url_from_toml(config: Config):
    try:
        return config.project.databricks.workspace_url
    except AttributeError:
        return None


def clusters_api(endpoint, workspace_url):
    return f"{workspace_url}/api/2.0/clusters/{endpoint}"


def api_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def get_api_token():
    # Try to get from env, fallback to Databricks CLI env var
    return os.environ.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_API_TOKEN")


def cluster_exists(cluster_name, workspace_url, token):
    url = clusters_api("list", workspace_url)
    resp = call_databricks_api("GET", url, dbutils=None)
    if not resp or "clusters" not in resp:
        return False, None
    for c in resp["clusters"]:
        if c.get("cluster_name") == cluster_name:
            return True, c["cluster_id"]
    return False, None


@with_config
def create_or_get_cluster(config: Config):
    cluster_name = get_cluster_name_from_toml(config)
    if not cluster_name:
        raise ValueError("Cluster name not found in pyproject.toml")
    cluster_conf = load_cluster_config(cluster_name)
    workspace_url = get_workspace_url_from_toml(config)
    if not workspace_url:
        raise ValueError("Workspace URL not found in pyproject.toml")
    token = get_api_token()
    if not token:
        raise ValueError("Databricks API token not found in environment")

    exists, cluster_id = cluster_exists(cluster_name, workspace_url, token)
    if exists:
        print(f"✅ Cluster '{cluster_name}' exists (ID: {cluster_id})")
        return cluster_id
    # Prepare minimal cluster spec for creation
    create_conf = cluster_conf.get("spec", cluster_conf)
    url = clusters_api("create", workspace_url)
    resp = call_databricks_api("POST", url, json_payload=create_conf, dbutils=None)
    if resp and "cluster_id" in resp:
        print(f"✅ Created new cluster: {cluster_name} (ID: {resp['cluster_id']})")
        return resp["cluster_id"]
    raise RuntimeError(f"❌ Error creating cluster: {resp}")
