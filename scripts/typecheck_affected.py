#!/usr/bin/env python3
"""Run BasedPyright for Python files changed in this checkout."""

from __future__ import annotations

import argparse
import subprocess

from test_affected import REPO_ROOT, changed_paths, git_lines

TYPECHECK_CONFIG_PATHS = {
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
}


def is_python_file(path: str) -> bool:
    return path.endswith(".py") and (REPO_ROOT / path).is_file()


def select_typecheck_files(paths: set[str]) -> tuple[list[str], bool]:
    if TYPECHECK_CONFIG_PATHS.intersection(paths):
        return [], True

    return sorted(path for path in paths if is_python_file(path)), False


def local_changed_paths() -> set[str]:
    paths: set[str] = set()
    paths.update(git_lines(["diff", "--name-only", "--cached"]))
    paths.update(git_lines(["diff", "--name-only"]))
    paths.update(git_lines(["ls-files", "--others", "--exclude-standard"]))
    return {path for path in paths if path}


def build_basedpyright_command(paths: list[str], full_check: bool) -> list[str]:
    command = ["uv", "run", "basedpyright"]
    if not full_check:
        command.extend(paths)
    return command


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
    parser.add_argument("--dry-run", action="store_true", help="print the selected command")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.changed_path:
        paths = set(args.changed_path)
    elif args.base:
        paths = changed_paths(args.base)
    else:
        paths = local_changed_paths()
    python_files, full_check = select_typecheck_files(paths)

    if not full_check and not python_files:
        print("No affected Python files selected for BasedPyright.")
        if paths:
            print("Changed files:")
            for path in sorted(paths):
                print(f"  {path}")
        return 0

    command = build_basedpyright_command(python_files, full_check)
    print("Selected BasedPyright command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
