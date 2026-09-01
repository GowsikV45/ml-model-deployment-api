from fastapi.testclient import TestClient

from app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_predict():
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        response = client.post(
            "/api/v1/predict",
            json=payload,
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "confidence" in data
        assert "request_id" in data

        assert isinstance(data["prediction"], int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["request_id"], str)


def test_predict_invalid_input():
    with TestClient(app) as client:
        payload = {
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        response = client.post(
            "/api/v1/predict",
            json=payload,
        )

        assert response.status_code == 422


def test_predict_batch():
    with TestClient(app) as client:
        payload = {
            "inputs": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
                {
                    "sepal_length": 6.2,
                    "sepal_width": 3.4,
                    "petal_length": 5.4,
                    "petal_width": 2.3,
                },
            ]
        }

        response = client.post(
            "/api/v1/predict-batch",
            json=payload,
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data
        assert len(data["predictions"]) == 2

        for prediction in data["predictions"]:
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "request_id" in prediction


def test_predict_batch_max_size():
    with TestClient(app) as client:
        inputs = []

        for _ in range(100):
            inputs.append(
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            )

        payload = {
            "inputs": inputs
        }

        response = client.post(
            "/api/v1/predict-batch",
            json=payload,
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data["predictions"]) == 100


def test_predict_batch_over_max_size():
    with TestClient(app) as client:
        inputs = []

        for _ in range(101):
            inputs.append(
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            )

        payload = {
            "inputs": inputs
        }

        response = client.post(
            "/api/v1/predict-batch",
            json=payload,
        )

        assert response.status_code == 422


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/api/v1/model-info")

        assert response.status_code == 200

        data = response.json()

        assert data["model_type"] == "RandomForestClassifier"
        assert data["version"] == "1.0.0"
        assert data["training_date"] == "2026-09-01"

        assert data["expected_features"] == [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

