# File: /predictive_maintenance/main.ipynb (Jupyter Notebook)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Simulate Machine Data
np.random.seed(42)
n = 1000
data = {
    "timestamp": pd.date_range(start='2025-01-01', periods=n, freq='H'),
    "temperature": np.random.normal(loc=75, scale=5, size=n),
    "vibration": np.random.normal(loc=0.02, scale=0.01, size=n),
    "pressure": np.random.normal(loc=1.0, scale=0.2, size=n),
    "failure": np.random.choice([0, 1], size=n, p=[0.97, 0.03])
}
df = pd.DataFrame(data)

# Step 3: Save Data to CSV
# File: /predictive_maintenance/data/machine_data.csv (Generated file)
df.to_csv("data/machine_data.csv", index=False)

# Step 4: Preprocess Data
# File: /predictive_maintenance/src/preprocess.py
X = df[["temperature", "vibration", "pressure"]]
y = df["failure"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Model
# File: /predictive_maintenance/src/train_model.py
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
# File: /predictive_maintenance/src/predict_failure.py
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize Failure Trend
plt.figure(figsize=(12, 5))
df['failure_rolling'] = df['failure'].rolling(24).mean()
plt.plot(df['timestamp'], df['failure_rolling'], label='Failure Probability (Rolling)')
plt.xlabel('Time')
plt.ylabel('Failure Rate')
plt.title('Rolling Failure Probability Over Time')
plt.legend()
plt.grid()
plt.show()
