"""Tests for central API model proxy routes."""

import asyncio
from typing import cast

from bovi_api.routes import proxy
from fastapi import Request
from fastapi.responses import JSONResponse


def test_proxy_curves_islc_forwards_to_model_app(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
        calls.append((base_url, path))
        return JSONResponse(
            status_code=200,
            content={"results": [{"test_id": 1, "total_305_yield": 100.0}]},
        )

    monkeypatch.setattr(proxy, "_proxy_post", fake_proxy_post)

    response = asyncio.run(
        proxy.proxy_curves_islc(cast(Request, object())),
    )

    assert response.status_code == 200
    assert response.body == b'{"results":[{"test_id":1,"total_305_yield":100.0}]}'
    assert calls == [("http://localhost:8001", "/islc")]


def test_proxy_curves_best_predict_forwards_to_model_app(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
        calls.append((base_url, path))
        return JSONResponse(
            status_code=200,
            content={"results": [{"test_id": 1, "total_305_yield": 100.0}]},
        )

    monkeypatch.setattr(proxy, "_proxy_post", fake_proxy_post)

    response = asyncio.run(
        proxy.proxy_curves_best_predict(cast(Request, object())),
    )

    assert response.status_code == 200
    assert response.body == b'{"results":[{"test_id":1,"total_305_yield":100.0}]}'
    assert calls == [("http://localhost:8001", "/best-predict")]


def test_proxy_curves_characteristic_batch_forwards_to_model_app(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
        calls.append((base_url, path))
        return JSONResponse(
            status_code=200,
            content={"results": [{"id": "cow-1", "value": 100.0}]},
        )

    monkeypatch.setattr(proxy, "_proxy_post", fake_proxy_post)

    response = asyncio.run(
        proxy.proxy_curves_characteristic_batch(cast(Request, object())),
    )

    assert response.status_code == 200
    assert response.body == b'{"results":[{"id":"cow-1","value":100.0}]}'
    assert calls == [("http://localhost:8001", "/characteristic/batch")]
