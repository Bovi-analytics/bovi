"""Tests for central API model proxy routes."""

from bovi_api.routes import proxy
from fastapi import Request
from fastapi.responses import JSONResponse


def test_proxy_curves_islc_forwards_to_model_app(client, monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
        calls.append((base_url, path))
        return JSONResponse(
            status_code=200,
            content={"results": [{"test_id": 1, "total_305_yield": 100.0}]},
        )

    monkeypatch.setattr(proxy, "_proxy_post", fake_proxy_post)

    response = client.post(
        "/curves/islc",
        json={"dim": [10, 30], "milkrecordings": [20.0, 25.0]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["total_305_yield"] == 100.0
    assert calls == [("http://localhost:8001", "/islc")]


def test_proxy_curves_best_predict_forwards_to_model_app(client, monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_proxy_post(base_url: str, path: str, request: Request) -> JSONResponse:
        calls.append((base_url, path))
        return JSONResponse(
            status_code=200,
            content={"results": [{"test_id": 1, "total_305_yield": 100.0}]},
        )

    monkeypatch.setattr(proxy, "_proxy_post", fake_proxy_post)

    response = client.post(
        "/curves/best-predict",
        json={"dim": [10, 30], "milkrecordings": [20.0, 25.0]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["total_305_yield"] == 100.0
    assert calls == [("http://localhost:8001", "/best-predict")]
