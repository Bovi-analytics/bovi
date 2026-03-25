import os
import subprocess
import sys

from bovi_core.config import Config, with_config


@with_config
def run_ruff_check(config: Config, target_path=None):
    """Run Ruff check using config for paths and environment detection"""

    # Use config attributes for paths
    target = target_path or config.project.project_root
    project_file_path = config.project.pyproject_file_path  # This is pyproject.toml

    # Build ruff command
    cmd = [sys.executable, "-m", "ruff", "check", target]
    if os.path.exists(project_file_path):
        cmd.extend(["--config", project_file_path])

    print(f"🔍 Running Ruff check in {config.environment} environment")
    print(f"📁 Target: {target}")
    print(f"⚙️ Config file: {project_file_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.stdout:
            print("📝 Ruff Output:")
            print(result.stdout)

        if result.stderr:
            print("⚠️ Ruff Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ No issues found!")
        else:
            print(f"❌ Ruff found issues (exit code: {result.returncode})")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏱️ Ruff check timed out after 60 seconds")
        return False
    except FileNotFoundError:
        print("❌ Ruff not installed. Install with: pip install ruff")
        return False
    except Exception as e:
        print(f"❌ Error running ruff: {e}")
        return False


@with_config
def run_ruff_format(config: Config, target_path=None):
    """Run Ruff formatter using config for paths and environment detection"""

    # Use config attributes for paths
    target = target_path or config.project.project_root
    project_file_path = config.project.pyproject_file_path  # This is pyproject.toml

    # Build ruff format command
    cmd = [sys.executable, "-m", "ruff", "format", target]
    if os.path.exists(project_file_path):
        cmd.extend(["--config", project_file_path])

    print(f"🎨 Formatting code in {config.environment} environment")
    print(f"📁 Target: {target}")
    print(f"⚙️ Config file: {project_file_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.stdout:
            print("📝 Format Output:")
            print(result.stdout)

        if result.stderr:
            print("⚠️ Format Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ Formatting completed successfully!")
        else:
            print(f"❌ Formatting failed (exit code: {result.returncode})")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏱️ Ruff format timed out after 60 seconds")
        return False
    except FileNotFoundError:
        print("❌ Ruff not installed. Install with: pip install ruff")
        return False
    except Exception as e:
        print(f"❌ Error running ruff format: {e}")
        return False


@with_config
def run_ruff_fix(config: Config, target_path=None):
    """Run Ruff check with auto-fix using config for paths"""

    # Use config attributes for paths
    if config.project.project_root is None:
        raise ValueError("Config not fully initialized - project_root is None")
    target = target_path or config.project.project_root
    project_file_path = config.project.pyproject_file_path

    # Build ruff command with --fix flag
    cmd = [sys.executable, "-m", "ruff", "check", target, "--fix"]
    if os.path.exists(project_file_path):
        cmd.extend(["--config", project_file_path])

    print(f"🔧 Running Ruff check with auto-fix in {config.environment} environment")
    print(f"📁 Target: {target}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.stdout:
            print("📝 Ruff Fix Output:")
            print(result.stdout)

        if result.stderr:
            print("⚠️ Ruff Fix Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ Auto-fix completed successfully!")
        else:
            print(f"❌ Some issues couldn't be auto-fixed (exit code: {result.returncode})")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏱️ Ruff fix timed out after 60 seconds")
        return False
    except FileNotFoundError:
        print("❌ Ruff not installed. Install with: pip install ruff")
        return False
    except Exception as e:
        print(f"❌ Error running ruff fix: {e}")
        return False
