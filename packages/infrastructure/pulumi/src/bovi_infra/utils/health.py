"""Health probe utilities for Azure Container Apps."""

from typing import Any


def build_health_probes(port: int, path: str | None = None) -> list[dict[str, Any]]:
    """Build liveness and readiness probes for a Container App."""
    probe_path = path or "/health"
    defaults = {"periodSeconds": 10, "timeoutSeconds": 2, "failureThreshold": 3}
    return [
        {
            "type": "Liveness",
            "httpGet": {"path": probe_path, "port": port},
            "initialDelaySeconds": 10,
            **defaults,
        },
        {
            "type": "Readiness",
            "httpGet": {"path": probe_path, "port": port},
            "initialDelaySeconds": 5,
            **defaults,
        },
    ]


def build_secret_env_vars(secret_keys: list[str]) -> list[dict[str, str]]:
    """Convert secret key names to Container App secret reference env var dicts."""
    return [
        {"name": key, "secretRef": key.lower().replace("_", "-")} for key in secret_keys
    ]  # pragma: allowlist secret
