import joblib
import os

# Set paths to the model and encoder
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../models/disease_predict.joblib")
)
ENCODER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../models/label_encoder.joblib")
)

try:
    rf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    print("✅ Model and LabelEncoder loaded successfully.")
except FileNotFoundError:
    print("❌ Model or Encoder file not found.")
    rf, le = None, None
except Exception as e:
    print(f"❌ Error loading model/encoder: {e}")
    rf, le = None, None
