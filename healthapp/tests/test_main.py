from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Disease Predictor API"}


def test_predict():
    # Example input for the Symptoms model
    test_input = {"symptom1": "fever", "symptom2": "cough"}

    # If `schemas.Symptoms` is a Pydantic model, make sure to match the fields accordingly
    response = client.post("/predict", json=test_input)

    # Check that the status code is 200
    assert response.status_code == 200

    # Check that the response contains a prediction
    assert "prediction" in response.json()
    # Optionally,  can add more checks depending on your output
    # For example:
    # assert response.json()["prediction"] == "disease_type_1"
