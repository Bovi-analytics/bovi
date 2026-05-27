"""Tests for the affected pytest target selector."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "test_affected.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("test_affected", _SCRIPT_PATH)
assert _SPEC is not None
test_affected = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(test_affected)

_TYPECHECK_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "typecheck_affected.py"
_TYPECHECK_SPEC = importlib.util.spec_from_file_location(
    "typecheck_affected", _TYPECHECK_SCRIPT_PATH
)
assert _TYPECHECK_SPEC is not None
typecheck_affected = importlib.util.module_from_spec(_TYPECHECK_SPEC)
assert _TYPECHECK_SPEC.loader is not None
_TYPECHECK_SPEC.loader.exec_module(typecheck_affected)


def _args(**overrides: bool) -> argparse.Namespace:
    defaults = {
        "all_markers": False,
        "include_azure": False,
        "include_model_weights": False,
        "include_multiprocessing": False,
        "include_slow": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_pytest_commands_groups_targets_with_same_marker_expression() -> None:
    commands = test_affected.build_pytest_commands(
        {
            "apps/backend/models/lactation-curves/tests/test_health.py",
            "apps/backend/models/lactation-autoencoder/tests/test_startup.py",
        },
        _args(),
    )

    assert commands == [
        [
            "uv",
            "run",
            "pytest",
            "-c",
            "pyproject.toml",
            "-v",
            "apps/backend/models/lactation-autoencoder/tests/test_startup.py",
            "apps/backend/models/lactation-curves/tests/test_health.py",
            "-m",
            "not azure and not model_weights and not multiprocessing and not slow "
            "and not tensorflow and not torch",
        ]
    ]


def test_build_pytest_commands_keeps_different_marker_expressions_separate() -> None:
    commands = test_affected.build_pytest_commands(
        {
            "apps/backend/models/lactation-curves/tests/test_health.py",
            "apps/backend/models/lactation-autoencoder/tests",
        },
        _args(),
    )

    assert commands == [
        [
            "uv",
            "run",
            "pytest",
            "-c",
            "pyproject.toml",
            "-v",
            "apps/backend/models/lactation-autoencoder/tests",
            "-m",
            "not azure and not model_weights and not multiprocessing and not slow and not torch",
        ],
        [
            "uv",
            "run",
            "pytest",
            "-c",
            "pyproject.toml",
            "-v",
            "apps/backend/models/lactation-curves/tests/test_health.py",
            "-m",
            "not azure and not model_weights and not multiprocessing and not slow "
            "and not tensorflow and not torch",
        ],
    ]


def test_select_tests_includes_script_tests_for_runner_changes() -> None:
    targets, allow_torch, allow_tensorflow, notes = test_affected.select_tests(
        {"scripts/test_affected.py"}
    )

    assert "scripts/tests/test_test_affected.py" in targets
    assert set(test_affected.HEALTH_TARGETS).issubset(targets)
    assert allow_torch is False
    assert allow_tensorflow is False
    assert notes == []


def test_select_tests_includes_script_tests_for_typecheck_runner_changes() -> None:
    targets, allow_torch, allow_tensorflow, notes = test_affected.select_tests(
        {"scripts/typecheck_affected.py"}
    )

    assert "scripts/tests/test_test_affected.py" in targets
    assert set(test_affected.HEALTH_TARGETS).issubset(targets)
    assert allow_torch is False
    assert allow_tensorflow is False
    assert notes == []


def test_select_typecheck_files_returns_changed_python_files() -> None:
    files, full_check = typecheck_affected.select_typecheck_files(
        {
            "scripts/test_affected.py",
            "justfile",
            "apps/infrastructure/.env.example",
        }
    )

    assert files == ["scripts/test_affected.py"]
    assert full_check is False


def test_select_typecheck_files_uses_full_check_for_config_changes() -> None:
    files, full_check = typecheck_affected.select_typecheck_files(
        {"pyproject.toml", "scripts/test_affected.py"}
    )

    assert files == []
    assert full_check is True


def test_build_basedpyright_command_uses_files_for_affected_check() -> None:
    assert typecheck_affected.build_basedpyright_command(["a.py", "b.py"], False) == [
        "uv",
        "run",
        "basedpyright",
        "a.py",
        "b.py",
    ]


def test_build_basedpyright_command_omits_files_for_full_check() -> None:
    assert typecheck_affected.build_basedpyright_command(["a.py"], True) == [
        "uv",
        "run",
        "basedpyright",
    ]


def test_local_changed_paths_includes_cached_unstaged_and_untracked(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_git_lines(args: list[str]) -> set[str]:
        calls.append(args)
        return {
            ("diff", "--name-only", "--cached"): {"staged.py"},
            ("diff", "--name-only"): {"unstaged.py"},
            ("ls-files", "--others", "--exclude-standard"): {"untracked.py"},
        }[tuple(args)]

    monkeypatch.setattr(typecheck_affected, "git_lines", fake_git_lines)

    assert typecheck_affected.local_changed_paths() == {
        "staged.py",
        "unstaged.py",
        "untracked.py",
    }
    assert calls == [
        ["diff", "--name-only", "--cached"],
        ["diff", "--name-only"],
        ["ls-files", "--others", "--exclude-standard"],
    ]
