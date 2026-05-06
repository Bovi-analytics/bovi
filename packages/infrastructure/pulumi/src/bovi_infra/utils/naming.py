"""Resource naming utilities for Azure resources."""


def build_resource_name(
    project: str,
    env: str,
    resource: str,
    max_length: int = 63,
    allow_hyphens: bool = True,
) -> str:
    """Build a consistent Azure resource name: {project}-{env}-{resource}."""
    name = f"{project}-{env}-{resource}".lower()

    if not allow_hyphens:
        name = name.replace("-", "")

    name = "".join(c for c in name if c.isalnum() or (c == "-" and allow_hyphens))
    name = name.strip("-")

    if len(name) > max_length:
        name = name[:max_length].rstrip("-")

    return name


def build_storage_account_name(project: str, env: str, suffix: str | None = None) -> str:
    """Build a valid Azure Storage account name (3-24 chars, lowercase, no hyphens)."""
    return build_resource_name(project, env, suffix or "stor", max_length=24, allow_hyphens=False)


def build_resource_tags(
    project: str,
    env: str,
    owner: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build standard resource tags applied to every Azure resource."""
    tags: dict[str, str] = {
        "Environment": env,
        "Project": project,
        "Owner": owner or "Bovi-Analytics",
        "ManagedBy": "Pulumi",
    }
    if extra:
        tags.update(extra)
    return tags
