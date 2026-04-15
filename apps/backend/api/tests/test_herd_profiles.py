"""Tests for the herd profiles CRUD API."""

VALID_PROFILE = {
    "name": "Test Herd",
    "description": "A test herd profile",
    "achieved_21_milk": 0.53,
    "achieved_305_milk": 0.50,
    "achieved_75_milk": 0.55,
    "achieved_milk": 0.41,
    "days_dry": 0.39,
    "days_in_milk": 0.44,
    "days_open": 0.38,
    "days_pregnant": 0.62,
    "historic_calving_interval": 0.54,
    "quality_sequence": 0.33,
}


def test_create_profile(client):
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Herd"
    assert data["id"] is not None
    assert "created_at" in data  # may be None under SQLite


def test_list_profiles_empty(client):
    response = client.get("/herd-profiles/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_profiles_after_create(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.get("/herd-profiles/")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.get(f"/herd-profiles/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Herd"


def test_get_profile_not_found(client):
    response = client.get("/herd-profiles/999")
    assert response.status_code == 404


def test_update_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    updated = {**VALID_PROFILE, "name": "Updated Herd", "achieved_21_milk": 0.70}
    response = client.put(f"/herd-profiles/{created['id']}", json=updated)
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Herd"
    assert response.json()["achieved_21_milk"] == 0.70


def test_delete_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.delete(f"/herd-profiles/{created['id']}")
    assert response.status_code == 204
    assert client.get(f"/herd-profiles/{created['id']}").status_code == 404


def test_duplicate_name_returns_409(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 409


def test_invalid_stat_value_returns_422(client):
    invalid = {**VALID_PROFILE, "achieved_21_milk": 1.5}  # exceeds 1.0
    response = client.post("/herd-profiles/", json=invalid)
    assert response.status_code == 422
