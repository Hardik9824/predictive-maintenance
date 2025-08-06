# File: /predictive_maintenance/src/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Assuming X_train, X_test, y_train, y_test are already preprocessed and available
# If not, import them from preprocess.py or run preprocess before this step

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "models/maintenance_model.pkl")
print("Model saved as 'maintenance_model.pkl'")
