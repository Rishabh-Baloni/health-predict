# 🏥 HealthPredict - AI-Powered Multi-Disease Detection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![scikit-learn](https://img.shields.io/badge/sklearn-1.2.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An advanced machine learning platform for early detection of chronic diseases**

[Live Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Disease Models](#-disease-models)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🌟 Overview

HealthPredict is a comprehensive web-based platform that leverages state-of-the-art machine learning algorithms to assist in the early detection of three major chronic diseases:

- 🫘 **Chronic Kidney Disease (CKD)**
- 🫀 **Liver Disease**
- 🧠 **Parkinson's Disease**

The platform provides healthcare professionals and researchers with a fast, accurate preliminary assessment tool using clinical parameters and biomarkers.

---

## ✨ Features

### 🎯 Core Capabilities

- **Multi-Disease Detection**: Three specialized AI models for different chronic diseases
- **High Accuracy**: Up to 100% accuracy on kidney disease detection
- **User-Friendly Interface**: Intuitive Streamlit-based web interface
- **Real-Time Predictions**: Instant results with confidence scores
- **Visual Analytics**: Interactive charts and confidence gauges
- **Sample Data Testing**: Quick test functionality with pre-loaded samples
- **Clinical Recommendations**: Evidence-based suggestions for each prediction

### 🔬 Advanced Features

- **AutoML Integration**: PyCaret for automated model optimization
- **Deep Learning**: Neural network models for complex pattern recognition
- **Feature Engineering**: PCA, SMOTE, and advanced preprocessing
- **Model Comparison**: Multiple algorithms tested and compared
- **Scalable Architecture**: Easy to add new disease models

---

## 🫘 Disease Models

### 1. Chronic Kidney Disease (CKD)

| Metric | Value |
|--------|-------|
| **Algorithm** | Extra Trees Classifier |
| **Accuracy** | 100% |
| **Features** | 24 clinical parameters |
| **Dataset** | 400 patients |

**Key Features:**
- Blood tests (Glucose, Urea, Creatinine, Hemoglobin)
- Urine tests (Specific Gravity, Albumin, Sugar)
- Clinical history (Hypertension, Diabetes, Anemia)

### 2. Liver Disease

| Metric | Value |
|--------|-------|
| **Algorithm** | Random Forest Classifier |
| **Accuracy** | 70% |
| **Features** | 10 liver function tests |
| **Dataset** | 583 patients |

**Key Features:**
- Bilirubin levels (Total & Direct)
- Liver enzymes (ALT, AST, Alkaline Phosphotase)
- Protein markers (Albumin, Total Proteins, A/G Ratio)

### 3. Parkinson's Disease

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost Classifier |
| **Accuracy** | 92.31% |
| **Features** | 22 voice measurements |
| **Dataset** | 195 voice recordings |

**Key Features:**
- Frequency variations (Jitter, Shimmer)
- Vocal fold vibrations
- Noise-to-harmonics ratio
- Nonlinear dynamical complexity

---

## 💻 Tech Stack

### Machine Learning

- **scikit-learn** 1.2.2 - Core ML algorithms
- **XGBoost** 1.7.6 - Gradient boosting
- **PyCaret** 3.2.0 - AutoML framework
- **TensorFlow** 2.12.0 - Deep learning
- **Keras** 2.15.0 - Neural networks
- **Imbalanced-learn** 0.12.4 - SMOTE, RandomOverSampler

### Data Processing

- **Pandas** 1.5.3 - Data manipulation
- **NumPy** 1.23.5 - Numerical computing
- **Joblib** 1.2.0 - Model serialization

### Visualization & Deployment

- **Streamlit** 1.28.0 - Web interface
- **Plotly** 5.17.0 - Interactive charts

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/healthpredict.git
cd healthpredict
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train Models (Optional)

If models are not included, train them first:

```bash
# Train all basic models
python training_scripts/train_kidney_extratrees.py
python training_scripts/train_liver_randomforest.py
python training_scripts/train_parkinsons_xgboost.py

# Train advanced models (optional)
python training_scripts/train_kidney_pycaret_automl.py
python training_scripts/train_liver_pycaret_automl.py
python training_scripts/train_parkinsons_pycaret_automl.py
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Quick Start

1. **Select Disease Module**: Choose from the sidebar menu
2. **Enter Clinical Data**: Fill in patient parameters
3. **Get Prediction**: Click the predict button
4. **View Results**: Analyze confidence scores and recommendations

### Sample Data Testing

- Check the "🎲 Use sample patient data" box
- Pre-filled values will populate automatically
- Click predict to see instant results

### Interpreting Results

- **Confidence Score**: Model certainty (0-100%)
- **Positive Detection**: Red alert with clinical recommendations
- **Negative Detection**: Green confirmation with health tips
- **Gauge Visualization**: Interactive confidence meter

---

## 📊 Model Performance

### Kidney Disease Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Extra Trees** | **100%** | **100%** | **100%** | **100%** |
| Logistic Regression | 98.75% | 98.79% | 98.75% | 98.75% |
| KNN | 98.75% | 98.79% | 98.75% | 98.75% |
| Random Forest | 97.50% | 97.60% | 97.50% | 97.48% |
| XGBoost | 97.50% | 97.60% | 97.50% | 97.48% |

### Advanced Techniques Applied

- ✅ PCA for dimensionality reduction (95% variance)
- ✅ RandomOverSampler for class imbalance
- ✅ PyCaret AutoML for optimization
- ✅ Neural Networks with Keras
- ✅ Cross-validation and hyperparameter tuning

---

## 📁 Project Structure

```
healthpredict/
├── data/                          # Datasets
│   ├── kidney/
│   │   └── kidney_disease.csv
│   ├── liver/
│   │   └── indian_liver_patient.csv
│   └── parkinsons/
│       └── parkinsons.data
│
├── models/                        # Trained models
│   ├── kidney/
│   │   ├── kidney_extratrees_basic.pkl
│   │   ├── kidney_extratrees_scaler.pkl
│   │   ├── kidney_pycaret_decisiontree.pkl
│   │   └── ...
│   ├── liver/
│   │   ├── liver_randomforest_basic.pkl
│   │   └── ...
│   └── parkinsons/
│       ├── parkinsons_xgboost_basic.pkl
│       └── ...
│
├── training_scripts/              # Model training scripts
│   ├── train_kidney_extratrees.py
│   ├── train_kidney_pycaret_automl.py
│   ├── train_liver_randomforest.py
│   ├── train_parkinsons_xgboost.py
│   └── ...
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── PROJECT_SHOWCASE.md
│   ├── MODEL_COMPARISON_RESULTS.md
│   └── NAMING_CONVENTIONS.md
│
├── app.py                         # Main Streamlit application
├── app_enhanced.py                # Enhanced UI version
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

---

## 🖼️ Screenshots

### Home Page
![Home Page](docs/screenshots/home.png)

### Kidney Disease Prediction
![Kidney Prediction](docs/screenshots/kidney.png)

### Results with Confidence Gauge
![Results](docs/screenshots/results.png)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT: FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This tool is designed as an **educational project** and **research prototype**. It should NOT be used for:

- ❌ Final medical diagnosis
- ❌ Treatment decisions
- ❌ Replacement of professional medical advice
- ❌ Clinical decision-making without verification

**Always consult qualified healthcare professionals** for medical advice, diagnosis, and treatment. The predictions provided by this system are probabilistic estimates based on machine learning models and may not reflect actual medical conditions.

### Limitations

- Models are trained on limited datasets
- May not generalize to all populations
- Accuracy varies by disease type
- No real-time clinical validation
- Not FDA approved or clinically validated

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle
- Inspiration: Healthcare AI research community
- Libraries: scikit-learn, Streamlit, PyCaret teams

---

<div align="center">

**Made with ❤️ for healthcare innovation**

⭐ Star this repo if you find it helpful!

</div>
