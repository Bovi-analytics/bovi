"""Tests for the affected pytest target selector."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "test_affected.py"
_SPEC = importlib.util.spec_from_file_location("test_affected", _SCRIPT_PATH)
assert _SPEC is not None
test_affected = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(test_affected)


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


def test_select_tests_includes_script_runner_tests_for_runner_changes() -> None:
    targets, allow_torch, allow_tensorflow, notes = test_affected.select_tests(
        {"scripts/test_affected.py"}
    )

    assert "scripts/tests/test_test_affected.py" in targets
    assert set(test_affected.HEALTH_TARGETS).issubset(targets)
    assert allow_torch is False
    assert allow_tensorflow is False
    assert notes == []
