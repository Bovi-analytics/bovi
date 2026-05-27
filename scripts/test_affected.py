#!/usr/bin/env python3
"""Run the pytest subset that matches the files changed in this checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FAST_TARGETS = [
    "packages/bovi-core/tests",
    "packages/models/lactationcurve/tests",
    "apps/backend/api/tests",
    "apps/backend/models/lactation-curves/tests",
    "apps/backend/models/lactation-autoencoder/tests/test_schemas.py",
    "apps/backend/models/lactation-autoencoder/tests/test_startup.py",
]

HEALTH_TARGETS = [
    "apps/backend/models/lactation-curves/tests/test_lactation_curves_health.py",
    "apps/backend/models/lactation-autoencoder/tests/test_startup.py",
]


def run_git(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def git_lines(args: list[str]) -> set[str]:
    result = run_git(args)
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def ref_exists(ref: str) -> bool:
    return run_git(["rev-parse", "--verify", "--quiet", ref]).returncode == 0


def changed_paths(base: str) -> set[str]:
    paths: set[str] = set()

    if ref_exists(base):
        paths.update(git_lines(["diff", "--name-only", f"{base}...HEAD"]))
    else:
        print(
            f"warning: base ref {base!r} not found; using local working tree only",
            file=sys.stderr,
        )

    paths.update(git_lines(["diff", "--name-only", "--cached"]))
    paths.update(git_lines(["diff", "--name-only"]))
    paths.update(git_lines(["ls-files", "--others", "--exclude-standard"]))
    return {path for path in paths if path}


def add_target(targets: set[str], path: str) -> None:
    if (REPO_ROOT / path).exists():
        targets.add(path)


def select_tests(paths: set[str]) -> tuple[set[str], bool, bool, list[str]]:
    targets: set[str] = set()
    allow_torch = False
    allow_tensorflow = False
    notes: list[str] = []

    for path in sorted(paths):
        if path in {"pyproject.toml", "uv.lock"}:
            targets.update(FAST_TARGETS)
            continue

        if path == "justfile" or path.startswith(".github/workflows/"):
            targets.update(HEALTH_TARGETS)
            continue

        if path in {"scripts/test_affected.py", "scripts/typecheck_affected.py"} or path.startswith(
            "scripts/tests/"
        ):
            add_target(targets, "scripts/tests/test_test_affected.py")
            targets.update(HEALTH_TARGETS)
            continue

        if path.startswith("scripts/"):
            targets.update(HEALTH_TARGETS)
            continue

        if path.startswith("apps/infrastructure/") or path.startswith("packages/infrastructure/"):
            targets.update(HEALTH_TARGETS)
            continue

        if path.startswith("apps/backend/api/"):
            add_target(targets, "apps/backend/api/tests")
            continue

        if path.startswith("apps/backend/models/lactation-curves/"):
            add_target(targets, "apps/backend/models/lactation-curves/tests")
            continue

        if path.startswith("apps/backend/models/lactation-autoencoder/"):
            allow_tensorflow = True
            add_target(targets, "apps/backend/models/lactation-autoencoder/tests")
            continue

        if path.startswith("packages/models/lactationcurve/"):
            add_target(targets, "packages/models/lactationcurve/tests")
            add_target(targets, "apps/backend/models/lactation-curves/tests")
            continue

        if path.startswith("packages/models/lactation-autoencoder/"):
            allow_tensorflow = True
            add_target(targets, "packages/models/lactation-autoencoder/tests")
            add_target(targets, "apps/backend/models/lactation-autoencoder/tests")
            continue

        if path.startswith("packages/models/bovi-yolo/"):
            allow_torch = True
            add_target(targets, "packages/models/bovi-yolo/tests")
            continue

        if path.startswith("packages/bovi-core/src/bovi_core/ml/dataloaders/loaders/pytorch"):
            allow_torch = True
            add_target(
                targets,
                "packages/bovi-core/tests/bovi_core/ml/dataloaders/loaders/test_pytorch_loader.py",
            )
            continue

        if path.startswith("packages/bovi-core/src/bovi_core/ml/dataloaders/loaders/tensorflow"):
            allow_tensorflow = True
            add_target(
                targets,
                "packages/bovi-core/tests/bovi_core/ml/dataloaders/loaders/test_tensorflow_loader.py",
            )
            continue

        if path.startswith("packages/bovi-core/src/bovi_core/ml/dataloaders/transforms/"):
            allow_torch = True
            allow_tensorflow = True
            add_target(
                targets,
                "packages/bovi-core/tests/bovi_core/ml/dataloaders/transforms/test_vision_transforms.py",
            )
            add_target(
                targets,
                "packages/bovi-core/tests/bovi_core/ml/dataloaders/transforms/test_timeseries_transforms.py",
            )
            continue

        if path.startswith("packages/bovi-core/src/"):
            add_target(targets, "packages/bovi-core/tests")
            continue

        if path.startswith("packages/bovi-core/tests/"):
            add_target(targets, path)
            if "pytorch" in path:
                allow_torch = True
            if "tensorflow" in path:
                allow_tensorflow = True
            if "vision" in path:
                allow_torch = True
                allow_tensorflow = True
            continue

        if path.startswith("apps/frontend/dashboard/"):
            notes.append(
                "dashboard files changed; run the dashboard's bun checks separately when needed"
            )

    return targets, allow_torch, allow_tensorflow, notes


def dependency_markers_for_target(target: str) -> frozenset[str]:
    if target.endswith("test_schemas.py") or target.endswith("test_startup.py"):
        return frozenset()

    markers: set[str] = set()
    if "pytorch" in target or "bovi-yolo" in target:
        markers.add("torch")
    if "tensorflow" in target or "lactation-autoencoder/tests" in target:
        markers.add("tensorflow")
    if "test_vision_transforms.py" in target:
        markers.update({"tensorflow", "torch"})
    return frozenset(markers)


def marker_expression(args: argparse.Namespace, allowed_markers: Iterable[str]) -> str:
    if args.all_markers:
        return ""

    allowed = set(allowed_markers)
    excluded = set()
    if not args.include_slow:
        excluded.add("slow")
    if not args.include_azure:
        excluded.add("azure")
    if not args.include_model_weights:
        excluded.add("model_weights")
    if not args.include_multiprocessing:
        excluded.add("multiprocessing")
    if "torch" not in allowed:
        excluded.add("torch")
    if "tensorflow" not in allowed:
        excluded.add("tensorflow")

    return " and ".join(f"not {marker}" for marker in sorted(excluded))


def contains_target(parent: str, child: str) -> bool:
    return parent != child and child.startswith(f"{parent.rstrip('/')}/")


def collapse_redundant_targets(targets: set[str]) -> set[str]:
    collapsed = set(targets)
    for target in targets:
        target_markers = dependency_markers_for_target(target)
        for candidate_parent in targets:
            if not contains_target(candidate_parent, target):
                continue
            parent_markers = dependency_markers_for_target(candidate_parent)
            if parent_markers.issuperset(target_markers):
                collapsed.discard(target)
                break
    return collapsed


def build_pytest_commands(targets: set[str], args: argparse.Namespace) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for target in sorted(collapse_redundant_targets(targets)):
        allowed_markers = dependency_markers_for_target(target)
        marker_expr = marker_expression(args, allowed_markers)
        groups[marker_expr].append(target)

    commands: list[list[str]] = []
    for marker_expr, group_targets in groups.items():
        cmd = ["uv", "run", "pytest", "-c", "pyproject.toml", "-v", *group_targets]
        if marker_expr:
            cmd.extend(["-m", marker_expr])
        commands.append(cmd)
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="git ref to compare against")
    parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="simulate a changed path; may be passed more than once",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the selected pytest command")
    parser.add_argument("--fast", action="store_true", help="run the standard fast Python subset")
    parser.add_argument("--include-slow", action="store_true", help="include tests marked slow")
    parser.add_argument("--include-azure", action="store_true", help="include tests marked azure")
    parser.add_argument(
        "--include-model-weights",
        action="store_true",
        help="include tests that require model weight artifacts",
    )
    parser.add_argument(
        "--include-multiprocessing",
        action="store_true",
        help="include multiprocessing worker tests",
    )
    parser.add_argument("--all-markers", action="store_true", help="do not filter by marker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.fast:
        targets = set(FAST_TARGETS)
        notes: list[str] = []
        paths: set[str] = set()
    else:
        paths = set(args.changed_path) if args.changed_path else changed_paths(args.base)
        targets, _allow_torch, _allow_tensorflow, notes = select_tests(paths)

    for note in notes:
        print(f"note: {note}", file=sys.stderr)

    if not targets:
        print("No affected Python pytest targets selected.")
        if paths:
            print("Changed files:")
            for path in sorted(paths):
                print(f"  {path}")
        return 0

    commands = build_pytest_commands(targets, args)

    print("Selected pytest command:")
    for cmd in commands:
        print(" ".join(cmd))

    if args.dry_run:
        return 0

    for cmd in commands:
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
