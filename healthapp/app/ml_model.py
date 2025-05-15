import os
import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache
from .logging_config import logger
from app import schemas

# ================
# Paths & Defaults
# ================

DEFAULT_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", "./models/disease_predict.joblib")
)
DEFAULT_ENCODER_PATH = Path(
    os.environ.get("ENCODER_PATH", "./models/label_encoder.joblib")
)


# ========================
# Model & Encoder Loaders
# ========================


@lru_cache(maxsize=1)
def load_model():
    if not DEFAULT_MODEL_PATH.exists():
        logger.error(f"❌ Model file not found at {DEFAULT_MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {DEFAULT_MODEL_PATH}")
    try:
        model = joblib.load(DEFAULT_MODEL_PATH)
        logger.info("✅ Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise


@lru_cache(maxsize=1)
def load_encoder():
    if not DEFAULT_ENCODER_PATH.exists():
        logger.error(f"❌ Encoder file not found at {DEFAULT_ENCODER_PATH}")
        raise FileNotFoundError(f"Encoder not found at {DEFAULT_ENCODER_PATH}")
    try:
        encoder = joblib.load(DEFAULT_ENCODER_PATH)
        logger.info("✅ Encoder loaded successfully.")
        return encoder
    except Exception as e:
        logger.error(f"❌ Failed to load encoder: {e}", exc_info=True)
        raise


# ========================
# Preprocessing Functions
# ========================


def preprocess_user_input(symptom_input: schemas.Symptoms):
    symptoms_dict = symptom_input.model_dump()
    active_symptoms = [symptom for symptom, present in symptoms_dict.items() if present]

    if not active_symptoms:
        logger.warning("⚠️ No symptoms provided.")
        return None

    input_vector = [
        1 if symptom in active_symptoms else 0 for symptom in SYMPTOM_COLUMNS
    ]
    return np.array(input_vector).reshape(1, -1)


# ==================
# Prediction Handler
# ==================


def predict_disease(symptom_input: schemas.Symptoms) -> str:
    """Predict disease from symptom input."""
    model = load_model()
    encoder = load_encoder()
    input_vector = preprocess_user_input(symptom_input)

    if input_vector is None:
        logger.warning("⚠️ No valid symptoms to predict.")
        return "Insufficient symptom data for prediction."

    try:
        prediction = model.predict(input_vector)
        predicted_label = encoder.inverse_transform(prediction)[0]
        logger.info(f"✅ Predicted disease: {predicted_label}")
        return predicted_label
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        return "Prediction error occurred."


# ==================
# Symptom Feature List
# ==================

SYMPTOM_COLUMNS = [
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_ urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic _patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload.1",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]
