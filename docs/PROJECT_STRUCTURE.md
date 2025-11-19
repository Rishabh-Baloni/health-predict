# HealthPredict ML - Project Structure

## 📁 Organized Directory Layout

```
HealthPredict/
│
├── 📄 app.py                          # Main Streamlit web application
├── 📄 requirements.txt                # Python dependencies
├── 📄 setup.ps1                       # Automated setup script
├── 📄 .gitignore                      # Git ignore rules
├── 📄 Rules.txt                       # Project rules and guidelines
│
├── 📁 data/                           # All datasets organized by disease
│   ├── kidney/
│   │   ├── kidney_disease.csv
│   │   ├── Model+Deployment.ipynb
│   │   ├── Neural+Networks+To+predict+Kidney+Disease.ipynb
│   │   ├── predicting-chronic-kidney-disease.ipynb
│   │   └── Pycaret+to+predict+Kidney+diseases.ipynb
│   │
│   ├── liver/
│   │   ├── indian_liver_patient.csv
│   │   ├── Model+Deployment (1).ipynb
│   │   └── PyCaret_Liver_Disease_prediction.ipynb
│   │
│   └── parkinsons/
│       ├── parkinsons.csv
│       └── Detecting_Parkinson's_Disease_.ipynb
│
├── 📁 models/                         # All trained models organized by disease
│   ├── kidney/
│   │   ├── kidney_disease_model.pkl              # Basic Extra Trees (100%)
│   │   ├── kidney_disease_scaler.pkl             # Feature scaler
│   │   ├── kidney_disease_metadata.pkl           # Model metadata
│   │   ├── kidney_disease_features.pkl           # Feature names
│   │   ├── kidney_disease_model_enhanced.pkl     # Enhanced with PCA+ROS (100%)
│   │   ├── kidney_disease_scaler_enhanced.pkl    # Enhanced scaler
│   │   └── kidney_pycaret_model.pkl              # PyCaret AutoML (99.64%)
│   │
│   ├── liver/
│   │   ├── liver_disease_model.pkl               # Random Forest (70%)
│   │   ├── liver_disease_scaler.pkl              # Feature scaler
│   │   ├── liver_disease_metadata.pkl            # Model metadata
│   │   ├── liver_disease_features.pkl            # Feature names
│   │   └── liver_pycaret_model.pkl               # PyCaret AutoML (70.13%)
│   │
│   └── parkinsons/
│       ├── parkinsons_model.pkl                  # XGBoost (92.31%)
│       ├── parkinsons_scaler.pkl                 # Feature scaler
│       ├── parkinsons_metadata.pkl               # Model metadata
│       ├── parkinsons_features.pkl               # Feature names
│       └── parkinsons_pycaret_model.pkl          # PyCaret AutoML (88.35%)
│
├── 📁 training_scripts/               # All model training scripts
│   ├── train_kidney_model.py                     # Basic kidney model
│   ├── train_kidney_model_enhanced.py            # Enhanced with PCA+RandomOverSampler
│   ├── train_kidney_pycaret.py                   # PyCaret AutoML for kidney
│   ├── train_kidney_neural.py                    # Neural Network for kidney
│   │
│   ├── train_liver_model.py                      # Basic liver model
│   ├── train_liver_pycaret.py                    # PyCaret AutoML for liver
│   │
│   ├── train_parkinsons_model.py                 # Basic Parkinson's model
│   └── train_parkinsons_pycaret.py               # PyCaret AutoML for Parkinson's
│
├── 📁 docs/                           # Documentation and reports
│   ├── README.md                                 # Main project documentation
│   ├── PROJECT_SHOWCASE.md                       # Resume highlight points
│   ├── MODEL_COMPARISON_RESULTS.md               # Comprehensive model comparison
│   └── PROJECT_STRUCTURE.md                      # This file
│
├── 📁 notebooks/                      # Jupyter notebooks (if any custom analysis)
│
└── 📁 venv/                           # Python virtual environment
    └── (Python packages and dependencies)
```

## 🎯 Key Features of This Structure

### ✅ Clean Separation of Concerns
- **Data**: All datasets and original notebooks in one place
- **Models**: Trained models organized by disease type
- **Scripts**: Training code separate from application code
- **Docs**: All documentation centralized

### ✅ Scalability
- Easy to add new diseases (just create new subfolder)
- Multiple model versions for each disease
- Clear naming conventions

### ✅ Professional Standards
- Follows Python project best practices
- Easy navigation for collaborators
- Version control friendly (.gitignore included)
- Self-documenting structure

### ✅ Workflow Support
1. **Data Collection**: `data/` folder with organized datasets
2. **Model Training**: `training_scripts/` with disease-specific scripts
3. **Model Storage**: `models/` with organized subdirectories  
4. **Deployment**: `app.py` at root level for easy access
5. **Documentation**: `docs/` for all project docs

## 📊 Model Organization

Each disease folder in `models/` contains:
- **Basic Model**: Initial trained model (.pkl)
- **Scaler**: Feature preprocessing scaler (.pkl)
- **Metadata**: Model performance metrics (.pkl)
- **Features**: Feature names list (.pkl)
- **Advanced Models**: Enhanced versions (PyCaret, Neural Net, etc.)

## 🚀 Usage

### To Train Models:
```powershell
cd training_scripts
python train_kidney_model.py
python train_liver_model.py
python train_parkinsons_model.py
```

### To Run Web App:
```powershell
streamlit run app.py
```

### To View Documentation:
```powershell
cd docs
# Open any .md file in your preferred Markdown viewer
```

## 📝 Notes

- All paths in code are now relative to maintain portability
- Training scripts save to `../models/{disease}/`  
- App loads from `models/{disease}/`
- Original notebooks preserved in `data/` for reference
