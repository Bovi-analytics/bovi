"""Shared fixtures for bestpred-py tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def source11_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "source11_current"


@pytest.fixture
def source10_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "source10_current"


@pytest.fixture
def source15_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "source15_current"


@pytest.fixture
def source14_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "source14_current"


@pytest.fixture
def source24_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "source24_current"


@pytest.fixture
def legacy_fixture_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "legacy_manual_expected"
