"""Shared fixtures for lactation curve tests."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def milkbot_key() -> str:
    """Return the MilkBot API key from environment."""
    key = os.getenv("milkbot_key")
    if not key:
        pytest.skip("milkbot_key not found in environment")
    return key


@pytest.fixture
def key() -> str:
    """Fixture providing the MilkBot API key."""
    return milkbot_key()


@pytest.fixture
def test_data_dir() -> Path:
    """Return test data directory containing csv/json fixtures."""
    return PACKAGE_ROOT / "tests" / "test_data"


@pytest.fixture
def reference_lactations(test_data_dir: Path) -> pd.DataFrame:
    """Return reusable reference lactations for covariance-fitting tests."""
    with (test_data_dir / "reference_lactations.json").open(encoding="utf-8") as file:
        records = json.load(file)
    return pd.DataFrame(records)
