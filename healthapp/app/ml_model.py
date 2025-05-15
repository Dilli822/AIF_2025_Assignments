import joblib

model_path = "../models/disease_predict.joblib"
# Load the model
model = joblib.load(model_path)

print("Model loaded successfully!")
print(model)
