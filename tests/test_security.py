from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


client = TestClient(app)


def test_missing_api_key():
    response = client.get("/api/v1/health")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_invalid_api_key():
    response = client.get(
        "/api/v1/health",
        headers={"X-API-Key": "wrong-api-key"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_unexpected_extra_field():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "unexpected_field": "not allowed"
    }

    response = client.post(
        "/api/v1/predict",
        headers={"X-API-Key": settings.API_KEY},
        json=payload
    )

    assert response.status_code == 422