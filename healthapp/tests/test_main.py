from app import ml_model
from app import schemas
import numpy as np


class MockSymptoms(schemas.Symptoms):
    skin_rash: bool = True
    fatigue: bool = True
    headache: bool = True
    vomiting: bool = False
    cough: bool = False


def test_preprocessing_vector_shape():
    input_data = MockSymptoms()
    vector = ml_model.preprocess_user_input(input_data, ml_model.symptom_columns)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1, len(ml_model.symptom_columns))


def test_predict_returns_string():
    input_data = MockSymptoms()
    result = ml_model.predict(input_data)
    assert isinstance(result, str)
