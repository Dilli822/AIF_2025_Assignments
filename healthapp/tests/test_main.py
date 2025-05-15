from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Disease Predictor API"}


def test_predict_valid():
    response = client.post(
        "/predict",
        json={"symptoms": "chronic fatigue, muscle pain, shortness of breath"},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_empty_input():
    response = client.post("/predict", json={"symptoms": ""})
    assert response.status_code == 200
    assert "prediction" in response.json()
