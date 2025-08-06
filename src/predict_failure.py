import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("models/maintenance_model.pkl")

# Example: Load or simulate new sensor data to predict failure
new_data = pd.DataFrame({
    "temperature": [75.2, 79.1, 71.3],
    "vibration": [0.023, 0.034, 0.017],
    "pressure": [1.10, 0.89, 1.05]
})

# Preprocess new data
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)  # Note: use the same scaler from training in production

# Make predictions
predictions = model.predict(new_data_scaled)

# Output results
for i, pred in enumerate(predictions):
    result = "Failure Expected" if pred == 1 else "Normal Operation"
    print(f"Sample {i + 1}: {result}")
