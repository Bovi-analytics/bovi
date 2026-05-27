#!/usr/bin/env python3
"""Run dashboard checks when frontend files changed in this checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from test_affected import REPO_ROOT, changed_paths, git_lines

DASHBOARD_ROOT = Path("apps/frontend/dashboard")
DASHBOARD_CHECK_COMMANDS = [
    ["bun", "run", "format:check"],
    ["bun", "run", "lint"],
    ["bun", "run", "typecheck"],
]


def local_changed_paths() -> set[str]:
    paths: set[str] = set()
    paths.update(git_lines(["diff", "--name-only", "--cached"]))
    paths.update(git_lines(["diff", "--name-only"]))
    paths.update(git_lines(["ls-files", "--others", "--exclude-standard"]))
    return {path for path in paths if path}


def select_dashboard_paths(paths: set[str]) -> list[str]:
    prefix = f"{DASHBOARD_ROOT}/"
    return sorted(path for path in paths if path == str(DASHBOARD_ROOT) or path.startswith(prefix))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        help="optional git ref to compare against; omitted means local working tree only",
    )
    parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="simulate a changed path; may be passed more than once",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the selected commands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.changed_path:
        paths = set(args.changed_path)
    elif args.base:
        paths = changed_paths(args.base)
    else:
        paths = local_changed_paths()

    dashboard_paths = select_dashboard_paths(paths)
    if not dashboard_paths:
        print("No affected dashboard files selected for frontend checks.")
        if paths:
            print("Changed files:")
            for path in sorted(paths):
                print(f"  {path}")
        return 0

    print("Affected dashboard files:")
    for path in dashboard_paths:
        print(f"  {path}")
    print("Selected frontend commands:")
    for command in DASHBOARD_CHECK_COMMANDS:
        print(" ".join(command))

    if args.dry_run:
        return 0

    if shutil.which("bun") is None:
        print("bun is required to run dashboard checks, but it was not found.")
        return 1

    if not (REPO_ROOT / DASHBOARD_ROOT / "node_modules").is_dir():
        print("Dashboard node_modules not found; run `bun install` in apps/frontend/dashboard.")
        return 1

    for command in DASHBOARD_CHECK_COMMANDS:
        result = subprocess.run(command, cwd=REPO_ROOT / DASHBOARD_ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
