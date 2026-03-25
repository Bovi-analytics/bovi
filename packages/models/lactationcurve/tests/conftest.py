"""Shared fixtures for lactation curve tests."""

import os

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def milkbot_key() -> str:
    """Return the MilkBot API key from environment."""
    key = os.getenv("milkbot_key")
    if not key:
        raise ValueError("milkbot_key not found in environment. Check your .env file.")
    return key


@pytest.fixture
def key() -> str:
    """Fixture providing the MilkBot API key."""
    return milkbot_key()
