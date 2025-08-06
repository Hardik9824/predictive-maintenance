# 🛠️ Predictive Maintenance System – Industry 4.0 Project

This project simulates machine sensor data and applies machine learning to predict equipment failures — a key application in modern, Industry 4.0-enabled production systems.

---

## 🚀 Project Overview

**Objective**: Build a predictive maintenance system that detects early failure signs in production equipment using Python, simulated sensor data, and ML models.

**Use Cases**:
- Failure detection in industrial production lines
- Optimizing maintenance schedules
- Value stream improvement
- Industry 4.0 readiness

---

## 🧩 Folder Structure

predictive_maintenance/
├── data/
│ └── machine_data.csv # Simulated machine sensor data
├── models/
│ └── maintenance_model.pkl # Saved ML model (after training)
├── src/
│ ├── preprocess.py # Data preparation
│ ├── train_model.py # Model training + evaluation
│ └── predict_failure.py # New prediction interface
├── main.ipynb # All-in-one Jupyter notebook
└── README.md # Project documentation

---

## ⚙️ Tools & Libraries

- Python 3.8+
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`

Install all using:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
 Workflow
Step 1: Generate Sensor Data
Use main.ipynb or preprocess.py to generate and scale synthetic sensor data:

Temperature

Vibration

Pressure

Failure indicator (binary)

Step 2: Train Model
python src/train_model.py

This will:

Train a RandomForestClassifier

Print accuracy & confusion matrix

Save the model as maintenance_model.pkl
Step 3: Predict Failure (New Data)

python src/predict_failure.py

Example output:
Sample 1: Normal Operation
Sample 2: Failure Expected
Sample 3: Normal Operation


Sample Model Performance

Confusion Matrix:
[[190   0]
 [  5   5]]

Classification Report:
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       190
           1       1.00      0.50      0.67        10


Industry 4.0 Concepts Used
Predictive Maintenance

Value Stream Mapping (VSM)

KPI Monitoring & Optimization

Real-Time Analytics

Data-Driven Decision Making


👨‍💻 Author
Hardikkumar Mansukhbhai Rupapara
🎓 Master’s in Electromobility @ FAU Erlangen-Nürnberg
📍 Stuttgart, Germany
📧 hardikrupapara9824@gmail.com

