import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
df = pd.read_csv("../data/Training.csv", index_col=0)
df = df.drop(df.columns[-1], axis=1)  # Drop last null column

# Encode the target labels
le = LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])

# Split features and target
X_train = df.drop("prognosis", axis=1)
y_train = df["prognosis"]

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Store the symptom column names
symptom_columns = X_train.columns.tolist()


# Preprocess user input
def preprocess_user_input(input_string: str):
    input_symptoms = [
        sym.strip().lower().replace(" ", "_") for sym in input_string.split(",")
    ]
    input_vector = [
        1 if symptom in input_symptoms else 0 for symptom in symptom_columns
    ]
    return np.array(input_vector).reshape(1, -1)


# Predict disease
def predict(symptoms: str):
    input_vector = preprocess_user_input(symptoms)
    prediction = rf.predict(input_vector)
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label


model_path = "../models/disease_predict.joblib"

# Load the model
model = joblib.load(model_path)

print("Model loaded successfully!")
print(model)
