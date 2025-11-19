# Naming Convention Guide - HealthPredict ML

## 📝 File Naming Standards

All files follow a clear, descriptive naming pattern that immediately tells you:
- **Disease**: kidney, liver, or parkinsons
- **Algorithm**: extratrees, randomforest, xgboost, pycaret, neuralnet
- **Variant**: basic, pca_ros (PCA + RandomOverSampler), automl
- **Component**: model, scaler, features, metadata

---

## 🫘 Kidney Disease Models

### Basic Extra Trees Model (100% Accuracy)
- `kidney_extratrees_basic.pkl` - Trained Extra Trees Classifier
- `kidney_extratrees_scaler.pkl` - MinMaxScaler for features
- `kidney_extratrees_features.pkl` - List of feature names
- `kidney_extratrees_metadata.pkl` - Model metrics and info

### Enhanced Extra Trees with PCA + RandomOverSampler (100% Accuracy)
- `kidney_extratrees_pca_ros.pkl` - Extra Trees with dimensionality reduction
- `kidney_extratrees_pca_ros_scaler.pkl` - Scaler for enhanced model
- `kidney_extratrees_pca.pkl` - PCA transformer (95% variance)

### PyCaret AutoML Model (99.64% Accuracy)
- `kidney_pycaret_decisiontree.pkl` - Best model from PyCaret (Decision Tree)

### Neural Network Model (Keras)
- `kidney_neuralnet_keras.h5` - Keras Sequential model
- `kidney_neuralnet_scaler.pkl` - StandardScaler
- `kidney_neuralnet_pca.pkl` - PCA transformer

---

## 🍺 Liver Disease Models

### Basic Random Forest Model (70% Accuracy)
- `liver_randomforest_basic.pkl` - Trained Random Forest Classifier
- `liver_randomforest_scaler.pkl` - MinMaxScaler for features
- `liver_randomforest_features.pkl` - List of feature names
- `liver_randomforest_metadata.pkl` - Model metrics and info

### PyCaret AutoML Model (70.13% Accuracy)
- `liver_pycaret_xgboost.pkl` - Best real model from PyCaret (XGBoost)

---

## 🧠 Parkinson's Disease Models

### Basic XGBoost Model (92.31% Accuracy)
- `parkinsons_xgboost_basic.pkl` - Trained XGBoost Classifier
- `parkinsons_xgboost_scaler.pkl` - MinMaxScaler for features
- `parkinsons_xgboost_features.pkl` - List of feature names
- `parkinsons_xgboost_metadata.pkl` - Model metrics and info

### PyCaret AutoML Model (88.35% Accuracy)
- `parkinsons_pycaret_lightgbm.pkl` - Best model from PyCaret (LightGBM)

---

## 🎓 Training Scripts

### Kidney Disease Training
- `train_kidney_extratrees.py` - Train basic Extra Trees model
- `train_kidney_extratrees_pca_ros.py` - Train with PCA + RandomOverSampler
- `train_kidney_pycaret_automl.py` - PyCaret AutoML training
- `train_kidney_neuralnet_keras.py` - Keras Neural Network training

### Liver Disease Training
- `train_liver_randomforest.py` - Train basic Random Forest model
- `train_liver_pycaret_automl.py` - PyCaret AutoML training

### Parkinson's Disease Training
- `train_parkinsons_xgboost.py` - Train basic XGBoost model
- `train_parkinsons_pycaret_automl.py` - PyCaret AutoML training

---

## 📋 Naming Pattern Breakdown

### Model Files
```
{disease}_{algorithm}_{variant}.pkl
```
**Examples:**
- `kidney_extratrees_basic.pkl` → Kidney + Extra Trees + Basic version
- `kidney_extratrees_pca_ros.pkl` → Kidney + Extra Trees + PCA & RandomOverSampler
- `liver_randomforest_basic.pkl` → Liver + Random Forest + Basic version
- `parkinsons_xgboost_basic.pkl` → Parkinson's + XGBoost + Basic version

### Scaler Files
```
{disease}_{algorithm}_{variant}_scaler.pkl
```
**Examples:**
- `kidney_extratrees_scaler.pkl`
- `liver_randomforest_scaler.pkl`
- `parkinsons_xgboost_scaler.pkl`

### Metadata Files
```
{disease}_{algorithm}_metadata.pkl
```
**Examples:**
- `kidney_extratrees_metadata.pkl`
- `liver_randomforest_metadata.pkl`
- `parkinsons_xgboost_metadata.pkl`

### Feature Files
```
{disease}_{algorithm}_features.pkl
```
**Examples:**
- `kidney_extratrees_features.pkl`
- `liver_randomforest_features.pkl`
- `parkinsons_xgboost_features.pkl`

### PyCaret Models
```
{disease}_pycaret_{best_algorithm}.pkl
```
**Examples:**
- `kidney_pycaret_decisiontree.pkl` → PyCaret chose Decision Tree
- `liver_pycaret_xgboost.pkl` → PyCaret chose XGBoost
- `parkinsons_pycaret_lightgbm.pkl` → PyCaret chose LightGBM

### Training Scripts
```
train_{disease}_{algorithm}[_{variant}].py
```
**Examples:**
- `train_kidney_extratrees.py` → Train basic Extra Trees
- `train_kidney_extratrees_pca_ros.py` → Train with PCA + ROS
- `train_kidney_pycaret_automl.py` → Train using PyCaret AutoML
- `train_liver_randomforest.py` → Train basic Random Forest
- `train_parkinsons_xgboost.py` → Train basic XGBoost

---

## 🎯 Benefits of This Naming System

### ✅ Self-Documenting
- File name tells you exactly what it contains
- No need to open files to understand their purpose
- Easy to find the right model/script

### ✅ Algorithm Transparency
- Immediately see which ML algorithm is used
- Know if it's a basic model or enhanced version
- Identify PyCaret AutoML results

### ✅ Version Control Friendly
- Clear distinction between model variants
- Easy to track different approaches
- Simple to compare performance

### ✅ Scalable
- Easy to add new algorithms
- Pattern supports multiple variants
- Consistent across all diseases

### ✅ Professional Standards
- Follows industry best practices
- Clear for team collaboration
- Resume-worthy organization

---

## 📊 Quick Reference Table

| Disease | Algorithm | Variant | Accuracy | File Name |
|---------|-----------|---------|----------|-----------|
| Kidney | Extra Trees | Basic | 100% | `kidney_extratrees_basic.pkl` |
| Kidney | Extra Trees | PCA+ROS | 100% | `kidney_extratrees_pca_ros.pkl` |
| Kidney | Decision Tree | PyCaret | 99.64% | `kidney_pycaret_decisiontree.pkl` |
| Liver | Random Forest | Basic | 70% | `liver_randomforest_basic.pkl` |
| Liver | XGBoost | PyCaret | 70.13% | `liver_pycaret_xgboost.pkl` |
| Parkinson's | XGBoost | Basic | 92.31% | `parkinsons_xgboost_basic.pkl` |
| Parkinson's | LightGBM | PyCaret | 88.35% | `parkinsons_pycaret_lightgbm.pkl` |

---

## 💡 Usage Examples

### Loading a Specific Model
```python
import joblib

# Load basic kidney model
kidney_model = joblib.load('models/kidney/kidney_extratrees_basic.pkl')
kidney_scaler = joblib.load('models/kidney/kidney_extratrees_scaler.pkl')

# Load PyCaret liver model
liver_model = joblib.load('models/liver/liver_pycaret_xgboost.pkl')

# Load enhanced kidney model
kidney_enhanced = joblib.load('models/kidney/kidney_extratrees_pca_ros.pkl')
```

### Running Training Scripts
```powershell
# Train basic models
python training_scripts/train_kidney_extratrees.py
python training_scripts/train_liver_randomforest.py
python training_scripts/train_parkinsons_xgboost.py

# Train enhanced versions
python training_scripts/train_kidney_extratrees_pca_ros.py

# Train AutoML models
python training_scripts/train_kidney_pycaret_automl.py
python training_scripts/train_liver_pycaret_automl.py
python training_scripts/train_parkinsons_pycaret_automl.py
```

---

## 🔄 Adding New Models

When adding new models, follow the pattern:

1. **Choose descriptive algorithm name**: `logistic`, `svm`, `gradientboost`, etc.
2. **Add variant if applicable**: `_tuned`, `_optimized`, `_ensemble`, etc.
3. **Save with pattern**: `{disease}_{algorithm}_{variant}.pkl`
4. **Name script consistently**: `train_{disease}_{algorithm}_{variant}.py`

**Example for new Kidney SVM model:**
- Model: `kidney_svm_rbf.pkl`
- Scaler: `kidney_svm_rbf_scaler.pkl`
- Script: `train_kidney_svm_rbf.py`
