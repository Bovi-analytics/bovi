"""Tests for fitting result persistence endpoints."""

VALID_RESULT = {
    "model_type": "wood",
    "source_app": "lactation-curves",
    "input_data": {"cow_id": "cow-1"},
    "output_data": {"yield": 42.0},
    "metadata_extra": {"unit": "kg"},
}


def test_store_result(client):
    response = client.post("/results/", json=VALID_RESULT)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] is not None
    assert data["model_type"] == "wood"
    assert data["source_app"] == "lactation-curves"
    assert data["input_data"] == {"cow_id": "cow-1"}


def test_get_result(client):
    created = client.post("/results/", json=VALID_RESULT).json()
    response = client.get(f"/results/{created['id']}")
    assert response.status_code == 200
    assert response.json()["output_data"] == {"yield": 42.0}


def test_list_results_filters_by_model_type(client):
    client.post("/results/", json=VALID_RESULT)
    client.post("/results/", json={**VALID_RESULT, "model_type": "milkbot"})

    response = client.get("/results/?model_type=wood")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["model_type"] == "wood"


def test_get_result_not_found(client):
    response = client.get("/results/999")
    assert response.status_code == 404
