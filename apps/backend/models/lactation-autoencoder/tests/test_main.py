"""API endpoint tests - HTTP contract only."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_platform_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestValidationEndpoint:
    def test_validation_error_does_not_echo_large_input(self):
        response = client.post(
            "/predict",
            json={"dim": [0, 305], "milkrecordings": [25.0, 27.0]},
        )

        assert response.status_code == 422
        body = response.json()
        assert "request_id" in body
        assert body["detail"]
        assert "input" not in body["detail"][0]
        assert "305" not in response.text


@pytest.mark.model_weights
class TestPredictEndpoint:
    def test_predict_minimal(self):
        milk: list[float | None] = [None] * 304
        for i in range(18):
            milk[i] = 15.0 + i * 1.5
        response = client.post("/predict", json={"milk": milk})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 304
        assert all(isinstance(v, float) for v in data["predictions"])

    def test_predict_with_all_params(self):
        milk = [25.0] * 200
        response = client.post(
            "/predict",
            json={
                "milk": milk,
                "events": ["calving"] + ["pad"] * 199,
                "parity": 2,
                "herd_id": 2942694,
                "imputation_method": "linear",
            },
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_predict_with_periodic_records(self):
        response = client.post(
            "/predict",
            json={
                "dim": [10, 40, 70, 100],
                "milkrecordings": [30.0, 38.0, 35.0, 32.0],
                "parity": 2,
            },
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_predict_custom_herd_stats(self):
        response = client.post(
            "/predict",
            json={
                "milk": [25.0] * 100,
                "herd_stats": [0.5] * 10,
            },
        )
        assert response.status_code == 200

    def test_short_milk_padded(self):
        response = client.post("/predict", json={"milk": [25.0] * 50})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 304

    def test_empty_milk_rejected(self):
        assert client.post("/predict", json={"milk": []}).status_code == 422

    def test_bad_imputation_rejected(self):
        response = client.post(
            "/predict", json={"milk": [25.0] * 100, "imputation_method": "invalid"}
        )
        assert response.status_code == 422

    def test_wrong_herd_stats_length(self):
        response = client.post("/predict", json={"milk": [25.0] * 100, "herd_stats": [0.5] * 5})
        assert response.status_code == 422


@pytest.mark.model_weights
class TestBatchEndpoint:
    def test_batch_predict(self):
        response = client.post(
            "/predict/batch",
            json={
                "items": [{"milk": [25.0] * 200, "parity": 1}, {"milk": [30.0] * 150, "parity": 2}],
            },
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2
        assert all(len(r["predictions"]) == 304 for r in response.json()["results"])

    def test_batch_empty_rejected(self):
        assert client.post("/predict/batch", json={"items": []}).status_code == 422
