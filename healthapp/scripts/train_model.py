import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/Training.csv", index_col=0)
df_test = pd.read_csv("data/Testing.csv", index_col=0)

"""# Data Preprocessing and Visualization"""

disease_counts = df["prognosis"].value_counts()

plt.figure(figsize=(12, 8))
bars = plt.bar(disease_counts.index, disease_counts.values, color="skyblue")
plt.xticks(rotation=90)

# write each bar with its value
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.5,
        int(yval),
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.title("Frequency of Each Disease")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.tight_layout()


df = df.drop(df.columns[-1], axis=1)

"""all value in this coulmn is null so we can drop"""

print(df)

"""all of disease has the same number!"""

le = LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])
df_test["prognosis"] = le.transform(df_test["prognosis"])  # use transform only!

# 1. Split features and labels
X_train = df.drop("prognosis", axis=1)
y_train = df["prognosis"]
X_test = df_test.drop("prognosis", axis=1)
y_test = df_test["prognosis"]

"""# ٌِRandom Forest Modesl"""

# 2. Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test = rf.predict(X_test)

# 3. Training Metrics
print("\n🔹 Training Metrics:")
print("Training Accuracy:", accuracy_score(y_train, y_pred_rf_train))
print(
    "Training Precision:", precision_score(y_train, y_pred_rf_train, average="weighted")
)
print("Training Recall:", recall_score(y_train, y_pred_rf_train, average="weighted"))
print("Training F1 Score:", f1_score(y_train, y_pred_rf_train, average="weighted"))

# 4. Testing Metrics
print("\n🔸 Testing Metrics:")
print("Testing Accuracy:", accuracy_score(y_test, y_pred_rf_test))
print("Testing Precision:", precision_score(y_test, y_pred_rf_test, average="weighted"))
print("Testing Recall:", recall_score(y_test, y_pred_rf_test, average="weighted"))
print("Testing F1 Score:", f1_score(y_test, y_pred_rf_test, average="weighted"))

# 5. Classification Reports
print("\n🔹 Classification Report for Training Data:")
print(classification_report(y_train, y_pred_rf_train, target_names=le.classes_))

print("\n🔸 Classification Report for Testing Data:")
print(classification_report(y_test, y_pred_rf_test, target_names=le.classes_))

# 6. Confusion Matrix for Test Data
cm = confusion_matrix(y_test, y_pred_rf_test)
plt.figure(figsize=(15, 12))

# Annotating confusion matrix with values and labels
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()
plt.show()

symptom_columns = X_train.columns.tolist()

# Select a random index from the testing set
random_index = random.randint(0, len(X_test) - 1)

# Get the corresponding input features and label
random_x = X_test.iloc[random_index]
random_y = y_test.iloc[random_index]

# Print the label (both encoded and decoded) and features
print("\n🎯 Random Entry from Test Set")
print(f"Encoded Label (y): {random_y}")
print(f"Decoded Label    : {le.inverse_transform([random_y])[0]}")
print("\nSymptoms Present (x):")

# Print only the symptoms that are present (value == 1)
present_symptoms = [col for col in X_test.columns if random_x[col] == 1]
for symptom in present_symptoms:
    print(f" - {symptom.replace('_', ' ').capitalize()}")


def preprocess_user_input(input_string):
    input_symptoms = [
        sym.strip().lower().replace(" ", "_") for sym in input_string.split(",")
    ]
    input_vector = [
        1 if symptom in input_symptoms else 0 for symptom in symptom_columns
    ]
    return np.array(input_vector).reshape(1, -1)


def predict_disease(input_string):
    input_vector = preprocess_user_input(input_string)
    prediction = rf.predict(input_vector)
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label


user_input = "chronic fatigue, muscle pain, shortness of breath, chest pain, severe headaches, dizziness"
predicted_disease = predict_disease(user_input)
print("Predicted Disease:", predicted_disease)

user_input = input("Enter your symptoms separated by commas: ")
predicted_disease = predict_disease(user_input)
print("Predicted Disease:", predicted_disease)
