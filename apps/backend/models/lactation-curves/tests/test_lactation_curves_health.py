"""Tests for the health endpoint."""

import httpx


def test_health(api: httpx.Client):
    r = api.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_platform_health(api: httpx.Client):
    r = api.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
