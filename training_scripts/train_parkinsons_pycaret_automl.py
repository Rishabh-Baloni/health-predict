"""
Parkinson's Disease Prediction - PyCaret AutoML Approach
"""
import pandas as pd
from pycaret.classification import *

print("=" * 60)
print(" PARKINSON'S DISEASE - PYCARET AUTOML TRAINING")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/parkinsons_disease/parkinsons.data')
print(f"   Dataset shape: {df.shape}")

# Drop name column
df = df.drop('name', axis=1)

print(f"\n2. Class distribution:")
print(df['status'].value_counts())

# Setup PyCaret
print("\n3. Setting up PyCaret environment...")
exp = setup(
    data=df,
    target='status',
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
save_model(tuned_model, 'models/parkinsons/parkinsons_pycaret_model')
print("    ✓ Model saved")

print("\n" + "=" * 60)
print(" ✅ PYCARET TRAINING COMPLETE!")
print(f"    Best Model: {type(tuned_model).__name__}")
print("=" * 60)
