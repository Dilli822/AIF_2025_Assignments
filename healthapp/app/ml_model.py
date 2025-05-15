# healthapp/app/ml_model.py
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/disease_predict.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
