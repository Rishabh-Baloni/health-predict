"""
Train Liver Disease Prediction Model
Uses Random Forest Classifier with best parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" LIVER DISEASE PREDICTION - MODEL TRAINING")
print("="*60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/liver_disease/indian_liver_patient.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Data Preprocessing
print("\n2. Preprocessing data...")

# Handle missing values
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)

# Encode gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Prepare features and target
print("\n3. Preparing features and target...")
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Map target: 1 = Liver Disease, 2 = No Disease -> 1 = Disease, 0 = No Disease
y = y.map({1: 1, 2: 0})

print(f"   Features shape: {X.shape}")
print(f"   Target distribution:")
print(f"   - Liver Disease: {(y == 1).sum()}")
print(f"   - No Disease: {(y == 0).sum()}")

# Scale features
print("\n4. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
print("\n5. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Train Random Forest model
print("\n6. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
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
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Liver Disease']))

# Feature importance
print("\n" + "="*60)
print(" FEATURE IMPORTANCE")
print("="*60)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# Save model and scaler
print("\n8. Saving models...")
joblib.dump(model, 'models/liver/liver_randomforest_basic.pkl')
joblib.dump(scaler, 'models/liver/liver_randomforest_scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/liver/liver_randomforest_features.pkl')

print("   ✓ Model saved: models/liver/liver_randomforest_basic.pkl")
print("   ✓ Scaler saved: models/liver/liver_randomforest_scaler.pkl")
print("   ✓ Feature names saved: models/liver/liver_randomforest_features.pkl")

# Save model metadata
metadata = {
    'model_name': 'Random Forest',
    'accuracy': accuracy,
    'features': X.columns.tolist(),
    'target_classes': ['No Disease', 'Liver Disease']
}
joblib.dump(metadata, 'models/liver/liver_randomforest_metadata.pkl')
print("   ✓ Metadata saved: models/liver/liver_randomforest_metadata.pkl")

print("\n✅ Training complete!")
print("="*60)
