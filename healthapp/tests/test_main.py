from unittest.mock import patch, MagicMock
from app import ml_model
from app import schemas


class MockSymptoms(schemas.Symptoms):
    skin_rash: bool = True
    fatigue: bool = True


@patch("app.ml_model.joblib.load")
def test_predict_returns_string(mock_joblib_load):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_label_encoder = MagicMock()
    mock_label_encoder.inverse_transform.return_value = ["flu"]

    # joblib.load() will be called twice — once for model, once for encoder
    mock_joblib_load.side_effect = [mock_model, mock_label_encoder]

    result = ml_model.predict(MockSymptoms())
    assert isinstance(result, str)
    assert result == "flu"
