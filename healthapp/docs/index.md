# Disease Predictor

This project predicts the disease based on symptoms provided by the user using a machine learning model.

---

## API Endpoints

### POST `/predict`

This endpoint predicts the most probable disease based on the symptoms input.

Example:[127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

#### Request

- **URL**: `/predict`
- **Method**: `POST`
- **Headers**:

  - `Content-Type`: `application/json`
- **Body (JSON)**:

```json
{
  "itching": 1,
  "skin_rash": 1,
  "nodal_skin_eruptions": 0,
  "continuous_sneezing": 0,
  "shivering": 0,
  "chills": 0,
  "joint_pain": 0,
  "stomach_pain": 0,
  "acidity": 0
}
```

- **OR Request Body (JSON)**:

```json
{
  "fever": false,
  "cough": false,
  "fatigue": false,
  "headache": false,
  "skin_rash": true,
  "itching": true,
  "dischromic_patches": true
}

```

- **Response Body (JSON)**:

```json
{
  "prediction": "Fungal infection"
}

```

 **(Disclaimer):**

* All the Credit goes to Kaushil268 Kaggle User.
* I donot own the Code. It is just an AI Assist Tool that makes predictions based on the symptoms. This system can make mistakes.

**References**

[1] Kaushil268, "Disease Prediction with Random Forest," Kaggle, Mar. 15, 2023. [Online]. Available: https://www.kaggle.com/code/kaushil268/disease-prediction-with-random-forest. [Accessed: May 14, 2025].
