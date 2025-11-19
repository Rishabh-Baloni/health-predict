"""
Liver Disease Prediction - PyCaret AutoML Approach
"""
import pandas as pd
from pycaret.classification import *

print("=" * 60)
print(" LIVER DISEASE - PYCARET AUTOML TRAINING")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/liver_disease/indian_liver_patient.csv')
print(f"   Dataset shape: {df.shape}")

# Preprocessing
print("\n2. Preprocessing data...")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df.fillna(df.median(), inplace=True)

print(f"   Class distribution:")
print(df['Dataset'].value_counts())

# Setup PyCaret
print("\n3. Setting up PyCaret environment...")
exp = setup(
    data=df,
    target='Dataset',
    session_id=42,
    normalize=True,
    transformation=True,
    fix_imbalance=True,
    verbose=False
)

# Compare models
print("\n4. Comparing models...")
best_models = compare_models(n_select=3, sort='Accuracy')

print("\n5. Top 3 Models:")
for i, model in enumerate(best_models, 1):
    print(f"   {i}. {type(model).__name__}")

# Tune best model
best_model = best_models[0]
print(f"\n6. Tuning {type(best_model).__name__}...")
tuned_model = tune_model(best_model, optimize='Accuracy')

# Evaluate
print("\n7. Model Evaluation:")
evaluate_model(tuned_model)

# Save
print("\n8. Saving model...")
save_model(tuned_model, 'models/liver/liver_pycaret_model')
print("    ✓ Model saved")

print("\n" + "=" * 60)
print(" ✅ PYCARET TRAINING COMPLETE!")
print(f"    Best Model: {type(tuned_model).__name__}")
print("=" * 60)
