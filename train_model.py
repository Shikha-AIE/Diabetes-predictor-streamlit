# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("C:/Users/GJPC/Desktop/datasets/diabetes.csv")

# Split into features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
