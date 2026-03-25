"""
Configuration utilities for author info extraction and validation.

This module provides:
- Centralized author info extraction from pyproject.toml
- File change tracking for Config singleton invalidation
- Experiment name extraction from versioned paths
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ConfigFileTracker:
    """Tracks file modification times to detect changes.

    Used by Config singleton to auto-reload when pyproject.toml
    or run config files are modified.
    """

    def __init__(self) -> None:
        self._file_mtimes: Dict[str, float] = {}

    def track_file(self, file_path: str) -> None:
        """Store the current mtime of a file."""
        path = Path(file_path)
        if path.exists():
            self._file_mtimes[file_path] = path.stat().st_mtime

    def has_changed(self, file_path: str) -> bool:
        """Check if file has been modified since last tracked."""
        path = Path(file_path)
        if not path.exists():
            return file_path in self._file_mtimes

        current_mtime = path.stat().st_mtime
        stored_mtime = self._file_mtimes.get(file_path)

        if stored_mtime is None:
            return True  # Never tracked = treat as changed

        return current_mtime > stored_mtime

    def any_changed(self) -> bool:
        """Check if any tracked file has changed."""
        return any(self.has_changed(f) for f in self._file_mtimes)

    def clear(self) -> None:
        """Clear all tracked files."""
        self._file_mtimes.clear()


class AuthorConfigError(ValueError):
    """Raised when author configuration is invalid or not set."""

    pass


class ProjectConfigError(ValueError):
    """Raised when project configuration is invalid."""

    pass


def get_repo_name(project_root: str) -> str:
    """
    Extract repository name from git remote or folder name.

    Priority:
    1. Git remote origin URL (if available)
    2. Folder name of project root

    Args:
        project_root: Path to project root directory

    Returns:
        Repository/project name
    """
    import subprocess

    # Try git remote first
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Handle: git@github.com:org/repo.git or https://github.com/org/repo.git
            repo_name = url.rstrip(".git").split("/")[-1]
            return repo_name
    except Exception:
        pass

    # Fallback to folder name
    return Path(project_root).name


def validate_project_name(project_name: str, project_root: str) -> str:
    """
    Validate that project name has been updated from template default.

    Args:
        project_name: Name from pyproject.toml [project] name field
        project_root: Path to project root (for repo name extraction)

    Returns:
        Validated project name

    Raises:
        ProjectConfigError: If project name is still the template default
    """
    if project_name == "bovi-models-template":
        suggested_name = get_repo_name(project_root)
        raise ProjectConfigError(
            f"\n❌ Project name in pyproject.toml is still set to 'bovi-models-template'.\n"
            f"   Please update [project] name = '{suggested_name}'\n"
            f"\n"
            f"   Suggested name based on repo: '{suggested_name}'\n"
        )

    return project_name


def get_author_info(project_config: Any) -> Tuple[str, str]:
    """
    Extract and validate author information from project config.

    Args:
        project_config: ConfigNode with project data (contains 'authors' attribute)

    Returns:
        Tuple of (author_name, author_email)

    Raises:
        AuthorConfigError: If authors field is missing or still has default placeholders
    """
    authors = getattr(project_config, "authors", None)

    if authors is None:
        raise AuthorConfigError(
            "\n❌ No 'authors' field found in pyproject.toml [project] section.\n"
            "   Please add:\n"
            "   authors = [{ name = 'Your Name', email = 'your@email.com' }]\n"
        )

    # Handle list of authors (standard pyproject.toml format)
    if isinstance(authors, list) and len(authors) > 0:
        first_author = authors[0]
        if isinstance(first_author, dict):
            author_name = first_author.get("name", "Your Name")
            author_email = first_author.get("email", "your.email@example.com")
        else:
            author_name = "Your Name"
            author_email = "your.email@example.com"
    else:
        # Handle ConfigNode or other object-style access
        author_name = getattr(authors, "name", "Your Name")
        author_email = getattr(authors, "email", "your.email@example.com")

    # Validate author is not default placeholder
    if author_email == "your.email@example.com" or author_name == "Your Name":
        raise AuthorConfigError(
            "\n❌ Author information in pyproject.toml is still set to default placeholders.\n"
            "   Please update [project] authors = [{ name = '...', email = '...' }]\n"
            "   with your actual name and email.\n"
            "\n"
            "   This is required for:\n"
            "   - Saving models to Unity Catalog\n"
            "   - MLflow experiment tracking\n"
        )

    return author_name, author_email


def extract_experiment_name_from_path(file_path: str) -> Optional[str]:
    """Extract experiment name from a config file path.

    Supports both legacy and versioned directory structures:
      - Legacy: data/experiments/{experiment}/config.yaml
      - Versioned: data/experiments/{experiment}/versions/v1/config/config.yaml

    Args:
        file_path: Path to a config file within an experiment directory.

    Returns:
        The experiment name extracted from the path,
        or None if 'experiments' is not found in the path.

    """
    path_parts = Path(file_path).parts
    if "experiments" in path_parts:
        exp_idx = path_parts.index("experiments")
        return path_parts[exp_idx + 1] if exp_idx + 1 < len(path_parts) else None
    return None
