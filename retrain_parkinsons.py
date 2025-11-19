import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models/parkinsons', exist_ok=True)

# Load the dataset (you'll need to provide the actual dataset)
# For now, creating a sample compatible model
np.random.seed(42)
n_samples = 195
n_features = 22

# Generate sample data (replace with actual Parkinson's dataset)
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'models/parkinsons/parkinsons_xgboost_basic.pkl')
joblib.dump(scaler, 'models/parkinsons/parkinsons_xgboost_scaler.pkl')

# Save feature names (optional)
feature_names = [f'feature_{i}' for i in range(n_features)]
joblib.dump(feature_names, 'models/parkinsons/parkinsons_xgboost_features.pkl')

print("✅ Parkinson's model retrained and saved successfully!")
print(f"Model accuracy: {model.score(X_test_scaled, y_test):.2f}")