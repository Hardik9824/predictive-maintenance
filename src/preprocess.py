# File: /predictive_maintenance/src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data/machine_data.csv")

# Select features and target
X = df[["temperature", "vibration", "pressure"]]
y = df["failure"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Output shapes
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
