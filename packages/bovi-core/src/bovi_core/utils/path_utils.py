"""
Path resolution utilities for the project.

This module provides centralized path resolution functions for:
- Finding project root (via pyproject.toml)
- Finding project source directory
- Resolving relative data paths
"""

import glob
import logging
import os
import tomllib
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def read_project_name_from_toml(toml_path: Path) -> str:
    """Read and return the project name from a pyproject.toml file"""
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("name", "unknown")
    except (ImportError, FileNotFoundError, KeyError, ValueError):
        return "unknown"


def _toml_matches_project(toml_path: Path, project_name: str) -> bool:
    """Check if a pyproject.toml declares the given project name."""
    return read_project_name_from_toml(toml_path) == project_name


def get_project_root(project_name: Optional[str] = None) -> str:
    """Find project root by looking for pyproject.toml.

    When ``project_name`` is provided the search skips any pyproject.toml
    whose ``project.name`` does not match.  This is essential in monorepo
    layouts where multiple pyproject.toml files exist at different levels.

    Searches in this order:
    1. Current directory
    2. Parent directories (walking up)
    3. Subdirectories (walking down)

    Args:
        project_name: If given, only accept a pyproject.toml whose
            ``project.name`` equals this value.

    Returns:
        str: Absolute path to project root

    Raises:
        ValueError: If no matching pyproject.toml found
    """
    current_path = Path.cwd()

    def _is_match(path: Path) -> bool:
        toml = path / "pyproject.toml"
        if not toml.exists():
            return False
        if project_name is not None:
            return _toml_matches_project(toml, project_name)
        return True

    # 1. Current directory
    if _is_match(current_path):
        return str(current_path)

    # 2. Walk up parent directories
    for path in current_path.parents:
        if _is_match(path):
            found_name = read_project_name_from_toml(path / "pyproject.toml")
            print(f"🔍 Found pyproject.toml in parent directory: {path}")
            print(f"      📋 Project name: {found_name}")
            print(f"      📁 Current working directory: {current_path}")
            print(f"      📁 Traversed to: {path}")
            return str(path)

    # 3. Walk down subdirectories
    for path in current_path.glob("**/*"):
        if path.is_dir() and _is_match(path):
            found_name = read_project_name_from_toml(path / "pyproject.toml")
            print(f"🔍 Found pyproject.toml in subdirectory: {path}")
            print(f"      📋 Project name: {found_name}")
            return str(path)

    target = f" for project '{project_name}'" if project_name else ""
    raise ValueError(
        f"Project root not found{target}. No matching pyproject.toml in the current "
        "directory, its parents, or its children."
    )


def get_project_src() -> str:
    """
    Find project src directory.

    Searches in this order:
    1. Current directory has 'src' child
    2. Parent directories have 'src' child
    3. Glob for 'src' in current directory

    Returns:
        str: Absolute path to src directory

    Raises:
        ValueError: If src directory not found
    """
    # First, check if we're already in the project root (has src as a child)
    current_path = Path.cwd()
    if (current_path / "src").exists():
        return str(current_path / "src")

    # If not, walk up the parent directories looking for src
    for path in current_path.parents:
        if (path / "src").exists():
            return str(path / "src")

    # Fallback: look for src in current directory
    src_dir = glob.glob("src")
    if src_dir:
        return str(Path(src_dir[0]).parent / "src")

    raise ValueError("Project src not found")


def get_project_paths(
    environment: str, project_root_path: str, project_data: Optional[dict] = None
) -> dict:
    """
    Get all project paths based on environment.

    Args:
        environment: Environment type (local, databricks, vscode_remote)
        project_root_path: Path to project root
        project_data: Optional project metadata containing workspace_user and project name

    Returns:
        Dict with project paths:
        - project_root_path
        - src_dir_path
        - notebooks_dir_path
        - data_dir_path
    """
    try:
        project_src = get_project_src()
    except ValueError:
        # In monorepo layouts there is no single top-level src directory.
        # Fall back to the project root so that downstream code still gets a
        # valid path.
        logger.debug("No top-level src directory found; falling back to project root")
        project_src = project_root_path
    print(f"      📁 project_src: {project_src}")
    # For Databricks, we can optionally override the project root if specific workspace
    # info is provided
    if environment == "databricks" and project_data:
        workspace_user = project_data.get("project", {}).get("workspace_user")
        project_name = project_data.get("project", {}).get("name")

        # Only override if both values are provided and we're in a Databricks environment
        if workspace_user and project_name:
            databricks_root = f"/Workspace/Users/{workspace_user}/projects/{project_name}"
            print(f"🔧 Using Databricks workspace path: {databricks_root}")
            project_root_path = databricks_root
        else:
            print(
                "ℹ️  Databricks environment detected but workspace_user or "
                "project_name not provided in project_data"
            )
            print(f"   📁 Using automatically detected project root: {project_root_path}")

    return {
        "project_root_path": project_root_path,
        "src_dir_path": project_src,  # Use the actual src directory path
        "notebooks_dir_path": os.path.join(project_root_path, "notebooks"),
        "data_dir_path": os.path.join(project_root_path, "data"),
    }


def _extract_version_number(version_name: str) -> int:
    """
    Extract numeric version from version folder name.

    Handles formats like 'v1', 'v11', 'v2' etc.
    Returns 0 for non-standard formats.

    Args:
        version_name: Version folder name (e.g., 'v1', 'v11')

    Returns:
        int: Extracted version number, or 0 if not parseable
    """
    import re

    match = re.match(r"v(\d+)", version_name)
    return int(match.group(1)) if match else 0


def get_run_config_path(
    project_root_path: str,
    experiment_name: str,
    config_file_name: str = "config.yaml",
    version: Optional[str] = None,
) -> str:
    """
    Get path to experiment run config file.

    Args:
        project_root_path: Path to project root
        experiment_name: Name of the experiment
        config_file_name: Name of config file (default: "config.yaml")
        version: Version folder name (e.g., "v1"). If None, uses latest
            version that has a config file.

    Returns:
        str: Path to run config file
    """
    versions_dir = Path(project_root_path) / "data" / "experiments" / experiment_name / "versions"

    if version is None and versions_dir.exists():
        # Find versions that have a config file, sorted by version number descending
        versions_with_config = []
        for d in versions_dir.iterdir():
            if d.is_dir():
                config_path = d / "config" / config_file_name
                if config_path.exists():
                    versions_with_config.append(d.name)

        if versions_with_config:
            # Sort by numeric version (v1, v2, v11 -> 1, 2, 11) and pick highest
            version = max(versions_with_config, key=_extract_version_number)
        else:
            # Fallback: pick highest version numerically even if no config exists
            all_versions = [d.name for d in versions_dir.iterdir() if d.is_dir()]
            version = max(all_versions, key=_extract_version_number) if all_versions else "v1"

    return str(versions_dir / (version or "v1") / "config" / config_file_name)


def make_path_absolute(path: Union[str, Path], project_root: Optional[Path] = None) -> Path:
    """
    Make a path absolute by resolving it relative to the project root.

    If the path is already absolute, it is returned as-is.
    Relative paths are resolved relative to the project root.

    Args:
        path: Path to make absolute (can be absolute or relative)
        project_root: Optional explicit project root. If None, auto-detected.

    Returns:
        Absolute Path object

    Example:
        >>> make_path_absolute("data/json/")
        Path("/Users/user/project/data/json")

        >>> make_path_absolute("/absolute/path/data")
        Path("/absolute/path/data")
    """
    path = Path(path)

    # If already absolute, return as-is
    if path.is_absolute():
        return path

    # Find project root if not provided
    if project_root is None:
        project_root = Path(get_project_root())

    # Resolve relative to project root
    resolved = project_root / path
    return resolved.resolve()


# Alias for backwards compatibility
resolve_data_path = make_path_absolute


def get_experiment_paths(
    project_root_path: str,
    experiment_name: str,
    version: str,
) -> dict:
    """
    Get all paths for a specific experiment version.

    Args:
        project_root_path: Path to project root
        experiment_name: Name of the experiment
        version: Version string (e.g., "v1", "1", or "1.0")

    Returns:
        Dict with experiment paths:
        - experiments_dir: Base experiments directory
        - dir: Main experiment version directory
        - config_dir: Config directory within version
        - weights_dir: Weights directory within version
    """
    # Normalize version string to "v{n}" format if needed
    version_str = version if version.startswith("v") else f"v{version}"

    experiments_dir = Path(project_root_path) / "data" / "experiments"
    version_dir = experiments_dir / experiment_name / "versions" / version_str

    return {
        "experiments_dir": str(experiments_dir),
        "dir": str(version_dir),
        "config_dir": str(version_dir / "config"),
        "weights_dir": str(version_dir / "weights"),
    }
