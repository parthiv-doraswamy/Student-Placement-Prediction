import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("placementdata.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# Remove StudentID
df.drop("StudentID", axis=1, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

df["ExtracurricularActivities"] = encoder.fit_transform(
    df["ExtracurricularActivities"]
)

df["PlacementTraining"] = encoder.fit_transform(
    df["PlacementTraining"]
)

df["PlacementStatus"] = encoder.fit_transform(
    df["PlacementStatus"]
)

# Features and Target
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Sample Prediction

sample_student = pd.DataFrame({
    "CGPA":[9.2],
    "Internships":[2],
    "Projects":[3],
    "Workshops/Certifications":[2],
    "AptitudeTestScore":[88],
    "SoftSkillsRating":[4.5],
    "ExtracurricularActivities":[1],
    "PlacementTraining":[1],
    "SSC_Marks":[85],
    "HSC_Marks":[90]
})

result = model.predict(sample_student)

if result[0] == 1:
    print("\nPrediction: Student is likely to be PLACED")
else:
    print("\nPrediction: Student is likely to be NOT PLACED")