"""
Train Parkinson's Disease Detection Model
Uses XGBoost Classifier for voice measurement features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" PARKINSON'S DISEASE DETECTION - MODEL TRAINING")
print("="*60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/parkinsons_disease/parkinsons.data')
print(f"   Dataset shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Data Preprocessing
print("\n2. Preprocessing data...")

# Prepare features and target
print("\n3. Preparing features and target...")
X = df.drop(['name', 'status'], axis=1)
y = df['status']

print(f"   Features shape: {X.shape}")
print(f"   Target distribution:")
print(f"   - Parkinson's: {(y == 1).sum()}")
print(f"   - Healthy: {(y == 0).sum()}")

# Scale features between -1 and 1
print("\n4. Scaling features...")
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Split data
print("\n5. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Train XGBoost model
print("\n6. Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Make predictions
print("\n7. Evaluating model...")
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "="*60)
print(" MODEL PERFORMANCE")
print("="*60)
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print("\n" + "="*60)
print(" CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"]))

# Feature importance (top 10)
print("\n" + "="*60)
print(" TOP 10 IMPORTANT FEATURES")
print("="*60)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
print(feature_importance.to_string(index=False))

# Save model and scaler
print("\n8. Saving models...")
joblib.dump(model, 'models/parkinsons/parkinsons_xgboost_basic.pkl')
joblib.dump(scaler, 'models/parkinsons/parkinsons_xgboost_scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/parkinsons/parkinsons_xgboost_features.pkl')

print("   Model saved: models/parkinsons/parkinsons_xgboost_basic.pkl")
print("   Scaler saved: models/parkinsons/parkinsons_xgboost_scaler.pkl")
print("   Feature names saved: models/parkinsons/parkinsons_xgboost_features.pkl")

# Save model metadata
metadata = {
    'model_name': 'XGBoost',
    'accuracy': accuracy,
    'features': X.columns.tolist(),
    'target_classes': ['Healthy', "Parkinson's Disease"]
}
joblib.dump(metadata, 'models/parkinsons/parkinsons_xgboost_metadata.pkl')
print("   Metadata saved: models/parkinsons/parkinsons_xgboost_metadata.pkl")

print("\nTraining complete!")
print("="*60)
