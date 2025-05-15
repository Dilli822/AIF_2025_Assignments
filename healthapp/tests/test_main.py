from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_endpoint_accepts_input():
    # User can send any symptoms # sample
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
