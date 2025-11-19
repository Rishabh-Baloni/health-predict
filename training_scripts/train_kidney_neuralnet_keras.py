"""
Kidney Disease Prediction - Neural Network Approach
Uses Keras/TensorFlow for deep learning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

print("=" * 60)
print(" KIDNEY DISEASE - NEURAL NETWORK TRAINING")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/kidney_disease/kidney_disease.csv')
print(f"   Dataset shape: {df.shape}")

# Preprocessing
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

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.drop('classification')
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df['classification'] = le_target.fit_transform(df['classification'])

print(f"\n3. Label distribution:")
print(df['classification'].value_counts())

# Separate features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Balance labels
print("\n4. Balancing labels with RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)
print(f"   After balancing: {pd.Series(y_balanced).value_counts().to_dict()}")

# Apply PCA for dimensionality reduction
print("\n5. Applying PCA (95% variance)...")
pca = PCA(0.95)
X_pca = pca.fit_transform(X_balanced)
print(f"   Original features: {X_balanced.shape[1]}")
print(f"   PCA features: {X_pca.shape[1]}")
print(f"   Variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# Scale features
print("\n6. Scaling features...")
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_pca)

# Split data
print("\n7. Splitting dataset (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Build Neural Network
print("\n8. Building Neural Network...")
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n   Model Architecture:")
model.summary()

# Train model
print("\n9. Training Neural Network...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=0
)

print(f"   Training completed in {len(history.history['loss'])} epochs")

# Evaluate model
print("\n10. Evaluating model...")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n   Training Accuracy: {train_acc:.4f}")
print(f"   Testing Accuracy: {test_acc:.4f}")
print(f"   Test Loss: {test_loss:.4f}")

# Save model
print("\n11. Saving models...")
model.save('models/kidney/kidney_neuralnet_model.h5')
joblib.dump(scaler, 'models/kidney/kidney_neuralnet_scaler.pkl')
joblib.dump(pca, 'models/kidney/kidney_neuralnet_pca.pkl')
joblib.dump(le_target, 'models/kidney/kidney_neuralnet_label_encoder.pkl')

print("    ✓ Neural network saved")
print("    ✓ Scaler saved")
print("    ✓ PCA saved")
print("    ✓ Label encoder saved")

print("\n" + "=" * 60)
print(" ✅ NEURAL NETWORK TRAINING COMPLETE!")
print(f"    Test Accuracy: {test_acc:.4f}")
print(f"    Model: models/kidney/kidney_neuralnet_model.h5")
print("=" * 60)
