from fastapi import FastAPI
from app import ml_model
from app import schemas

app = FastAPI(title="Disease Predictor API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Predictor API"}


@app.post("/predict")
def predict(input: schemas.Symptoms):
    prediction = ml_model.predict_disease(input)
    return {"prediction": prediction}
