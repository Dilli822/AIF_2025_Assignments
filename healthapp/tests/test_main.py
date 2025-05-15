from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_accepts_any_symptoms():
    sample_data = {
        "fever": False,
        "cough": False,
        "fatigue": True,
        "headache": True,
        "skin_rash": False,
        "itching": False,
        "dischromic_patches": False,
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_returns_200_with_valid_input():
    sample_data = {
        "fever": True,
        "cough": True,
        "fatigue": False,
        "headache": False,
        "skin_rash": True,
        "itching": False,
        "dischromic_patches": True,
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_handles_different_symptoms():
    sample_data = {
        "fever": True,
        "cough": False,
        "fatigue": False,
        "headache": True,
        "skin_rash": True,
        "itching": True,
        "dischromic_patches": False,
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "prediction" in response.json()
