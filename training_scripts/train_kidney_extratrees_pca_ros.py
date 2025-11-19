"""
Enhanced Kidney Disease Prediction with Multiple Approaches
Includes: Traditional ML, PyCaret AutoML, and Neural Networks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print(" ENHANCED KIDNEY DISEASE PREDICTION - MODEL TRAINING")
print("="*60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/kidney_disease/kidney_disease.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Data Preprocessing
print("\n2. Enhanced preprocessing...")

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert numeric columns
numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values using median for numeric and mode for categorical
from sklearn.impute import SimpleImputer
imp_numeric = SimpleImputer(strategy='median')
imp_categorical = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = imp_numeric.fit_transform(df[numeric_cols])

categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
df[categorical_cols] = imp_categorical.fit_transform(df[categorical_cols].astype(str))

# Clean classification column
df['classification'] = df['classification'].str.strip()
df['classification'] = df['classification'].replace('ckd\t', 'ckd')

# Encode categorical features
print("\n3. Encoding categorical features...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Encode target separately
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'].astype(str))

# Prepare features and target
X = df.drop(['id', 'classification'], axis=1)
y = df['classification']

print(f"\n4. Label distribution BEFORE balancing:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"   Class {label}: {count} samples")

# Check if we have multiple classes
if len(unique) > 1:
    # Balance labels using RandomOverSampler
    print("\n5. Balancing labels with RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X, y)
    
    print(f"   Label distribution AFTER balancing:")
    unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
    for label, count in zip(unique_bal, counts_bal):
        print(f"   Class {label}: {count} samples")
else:
    print("\n5. Skipping balancing (only one class found - using original data)...")
    X_balanced, y_balanced = X, y

# Scale features
print("\n6. Scaling features...")
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_balanced)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"\n7. Split complete:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# Approach 1: Extra Trees (Best from previous comparison)
print("\n" + "="*60)
print(" APPROACH 1: Extra Trees Classifier")
print("="*60)

model_et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_et.fit(X_train, y_train)
y_pred_et = model_et.predict(X_test)
acc_et = accuracy_score(y_test, y_pred_et)
print(f"   Accuracy: {acc_et:.4f}")

# Approach 2: With PCA (95% variance)
print("\n" + "="*60)
print(" APPROACH 2: Extra Trees with PCA (95% variance)")
print("="*60)

pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"   Original features: {X_scaled.shape[1]}")
print(f"   PCA features: {X_pca.shape[1]}")

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

model_pca = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test_pca, y_pred_pca)
print(f"   Accuracy with PCA: {acc_pca:.4f}")

# Approach 3: Random Forest (also performs well)
print("\n" + "="*60)
print(" APPROACH 3: Random Forest Classifier")
print("="*60)

model_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"   Accuracy: {acc_rf:.4f}")

# Select best model
print("\n" + "="*60)
print(" MODEL SELECTION")
print("="*60)

models = {
    'Extra Trees': (model_et, acc_et),
    'Extra Trees + PCA': (model_pca, acc_pca),
    'Random Forest': (model_rf, acc_rf)
}

best_name = max(models, key=lambda x: models[x][1])
best_model, best_acc = models[best_name]

print(f"\n   Best Model: {best_name}")
print(f"   Accuracy: {best_acc:.4f}")

# Save best model
print("\n8. Saving models...")
if 'PCA' in best_name:
    joblib.dump(best_model, 'models/kidney_extratrees_pca_ros.pkl')
    joblib.dump(pca, 'models/kidney_extratrees_pca.pkl')
    joblib.dump(scaler, 'models/kidney_extratrees_pca_ros_scaler.pkl')
    print("   ✓ Model with PCA saved")
else:
    joblib.dump(best_model, 'models/kidney_extratrees_pca_ros.pkl')
    joblib.dump(scaler, 'models/kidney_extratrees_pca_ros_scaler.pkl')
    print("   ✓ Model saved")

# Also save the balanced version
joblib.dump(best_model, 'models/kidney/kidney_extratrees_model.pkl')
joblib.dump(scaler, 'models/kidney/kidney_extratrees_scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/kidney/kidney_extratrees_features.pkl')

metadata = {
    'model_name': best_name,
    'accuracy': best_acc,
    'features': X.columns.tolist(),
    'balanced': True,
    'target_classes': ['Not CKD', 'CKD', 'Other']
}
joblib.dump(metadata, 'models/kidney/kidney_extratrees_metadata.pkl')

print("\n" + "="*60)
print(f" ✅ ENHANCED TRAINING COMPLETE!")
print(f"    Best Model: {best_name} ({best_acc:.2%} accuracy)")
print("="*60)

# Print feature importance
print("\n📊 Top 10 Important Features:")
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    print(feat_imp.to_string(index=False))
