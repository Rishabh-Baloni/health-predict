"""
Kidney Disease Prediction - PyCaret AutoML Approach
Uses PyCaret's automated machine learning capabilities
"""
import pandas as pd
import numpy as np
from pycaret.classification import *

print("=" * 60)
print(" KIDNEY DISEASE - PYCARET AUTOML TRAINING")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/kidney_disease/kidney_disease.csv')
print(f"   Dataset shape: {df.shape}")

# Data preprocessing
print("\n2. Preprocessing data...")

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Clean classification column
df['classification'] = df['classification'].str.strip()
df['classification'] = df['classification'].replace('ckd\t', 'ckd')

print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Class distribution:")
print(df['classification'].value_counts())

# Setup PyCaret
print("\n3. Setting up PyCaret environment...")
exp = setup(
    data=df,
    target='classification',
    session_id=42,
    normalize=True,
    transformation=True,
    fix_imbalance=True,  # Handle class imbalance (uses SMOTE by default)
    verbose=False
)

# Compare models
print("\n4. Comparing multiple models...")
print("   This may take a few minutes...")
best_models = compare_models(n_select=3, sort='Accuracy')

print("\n5. Top 3 Models:")
for i, model in enumerate(best_models, 1):
    print(f"   {i}. {type(model).__name__}")

# Get the best model
best_model = best_models[0]
print(f"\n6. Best Model: {type(best_model).__name__}")

# Tune the best model
print("\n7. Tuning hyperparameters...")
tuned_model = tune_model(best_model, optimize='Accuracy', n_iter=10)

# Evaluate model
print("\n8. Model Evaluation:")
evaluate_model(tuned_model)

# Make predictions
print("\n9. Making predictions on test set...")
predictions = predict_model(tuned_model)
print(f"   Predictions shape: {predictions.shape}")

# Save model
print("\n10. Saving model...")
save_model(tuned_model, 'models/kidney/kidney_pycaret_model')
print("    ✓ Model saved to models/kidney/kidney_pycaret_model.pkl")

# Feature importance
try:
    print("\n11. Feature Importance:")
    plot_model(tuned_model, plot='feature', save=True)
    print("    ✓ Feature importance plot saved")
except:
    print("    Feature importance not available for this model")

print("\n" + "=" * 60)
print(" ✅ PYCARET TRAINING COMPLETE!")
print(f"    Best Model: {type(tuned_model).__name__}")
print("=" * 60)
