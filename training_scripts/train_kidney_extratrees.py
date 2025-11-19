"""
Train Kidney Disease Prediction Models
Compares multiple ML algorithms and saves the best performing model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" KIDNEY DISEASE PREDICTION - MODEL TRAINING")
print("="*60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/kidney_disease/kidney_disease.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Data Preprocessing
print("\n2. Preprocessing data...")

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert numeric columns
numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
# For numeric columns - fill with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns - fill with mode
categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features
print("\n3. Encoding categorical features...")
# Create separate encoders for each column to maintain consistency
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Handle target variable separately
target_le = LabelEncoder()
df['classification'] = target_le.fit_transform(df['classification'].astype(str))
print(f"   Target classes: {target_le.classes_}")
print(f"   Target mapping: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")

# Prepare features and target
print("\n4. Preparing features and target...")
X = df.drop(['id', 'classification'], axis=1)
y = df['classification']

print(f"   Features shape: {X.shape}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Scale features
print("\n5. Scaling features...")
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Split data
print("\n6. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Train and evaluate models
print("\n7. Training and evaluating models...")
print("="*60)

results = []
best_model = None
best_accuracy = 0

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Display results summary
print("\n" + "="*60)
print(" RESULTS SUMMARY")
print("="*60)
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

# Save best model and scaler
print("\n" + "="*60)
print(f" BEST MODEL: {best_model_name}")
print(f" Accuracy: {best_accuracy:.4f}")
print("="*60)

print("\n8. Saving models...")
joblib.dump(best_model, 'models/kidney/kidney_extratrees_basic.pkl')
joblib.dump(scaler, 'models/kidney/kidney_extratrees_scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/kidney/kidney_extratrees_features.pkl')

print("   ✓ Best model saved: models/kidney/kidney_extratrees_basic.pkl")
print("   ✓ Scaler saved: models/kidney/kidney_extratrees_scaler.pkl")
print("   ✓ Feature names saved: models/kidney/kidney_extratrees_features.pkl")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': best_accuracy,
    'features': X.columns.tolist(),
    'target_classes': target_le.classes_.tolist(),
    'target_mapping': dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))
}
joblib.dump(metadata, 'models/kidney/kidney_extratrees_metadata.pkl')
joblib.dump(target_le, 'models/kidney/kidney_extratrees_target_encoder.pkl')
print("   ✓ Metadata saved: models/kidney/kidney_extratrees_metadata.pkl")
print("   ✓ Target encoder saved: models/kidney/kidney_extratrees_target_encoder.pkl")

print("\n✅ Training complete!")
print("="*60)
