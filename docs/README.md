# 🏥 HealthPredict - AI-Powered Multi-Disease Detection Platform

A comprehensive machine learning web application for predicting three major chronic diseases: **Chronic Kidney Disease**, **Liver Disease**, and **Parkinson's Disease** using clinical and voice measurement data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-green.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Disease Models](#disease-models)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Disclaimer](#disclaimer)

## 🎯 Overview

HealthPredict is an end-to-end machine learning project that demonstrates the practical application of ML in healthcare. The platform provides:

- **Real-time Disease Prediction** using trained ML models
- **Interactive Web Interface** built with Streamlit
- **High Accuracy Models** achieving 92-100% accuracy
- **Comprehensive Feature Analysis** with detailed clinical parameters
- **User-friendly Input Forms** for easy data entry

## ✨ Features

### 🫘 Chronic Kidney Disease Prediction
- **24 Clinical Features** including blood tests and medical history
- **11 ML Algorithms** compared (Logistic Regression, KNN, SVM, Random Forest, XGBoost, etc.)
- **100% Accuracy** achieved with Extra Trees Classifier
- Handles missing data intelligently

### 🫀 Liver Disease Prediction
- **10 Clinical Features** from liver function tests
- **Random Forest Classifier** with optimized hyperparameters
- **70% Accuracy** on imbalanced dataset
- Feature importance analysis

### 🧠 Parkinson's Disease Detection
- **22 Voice Measurement Features** for non-invasive detection
- **XGBoost Classifier** for optimal performance
- **92% Accuracy** in detecting early signs
- Voice biomarker analysis

### 🎨 User Interface
- Clean, professional design with custom CSS
- Multi-page navigation
- Real-time predictions with confidence scores
- Responsive layout for all screen sizes
- Color-coded results (Green = Healthy, Red = Disease Detected)

## 🔬 Disease Models

### Kidney Disease Model
- **Algorithm:** Extra Trees Classifier
- **Features:** 24 (age, blood pressure, glucose, creatinine, etc.)
- **Accuracy:** 100%
- **Dataset:** 400 patients (CKD + Non-CKD)
- **Preprocessing:** Missing value imputation, feature scaling, encoding

### Liver Disease Model
- **Algorithm:** Random Forest (n_estimators=200, max_depth=10)
- **Features:** 10 (age, gender, bilirubin, enzymes, proteins)
- **Accuracy:** 70%
- **Dataset:** 583 patients (416 liver disease, 167 healthy)
- **Preprocessing:** StandardScaler, label encoding

### Parkinson's Model
- **Algorithm:** XGBoost
- **Features:** 22 voice measurements (jitter, shimmer, HNR, etc.)
- **Accuracy:** 92.3%
- **Dataset:** 195 voice recordings (147 PD, 48 healthy)
- **Preprocessing:** MinMaxScaler (-1 to 1 range)

## 🛠️ Technology Stack

**Frontend:**
- Streamlit 1.28+
- Custom CSS styling

**Backend & ML:**
- Python 3.8+
- Scikit-learn 1.3+
- XGBoost 1.7+
- Pandas & NumPy

**Data Processing:**
- Feature scaling (MinMaxScaler, StandardScaler)
- Label encoding
- Missing value imputation

**Model Persistence:**
- Joblib for model serialization

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or Navigate to Project Directory**
```powershell
cd "d:\Projects\Mini Project\new"
```

2. **Create Virtual Environment**
```powershell
python -m venv venv
```

3. **Activate Virtual Environment**
```powershell
.\venv\Scripts\Activate.ps1
```

4. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

5. **Train the Models** (Required for first-time setup)
```powershell
# Train all models
python train_kidney_model.py
python train_liver_model.py
python train_parkinsons_model.py
```

## 🚀 Usage

### Running the Application

1. **Activate Virtual Environment**
```powershell
.\venv\Scripts\Activate.ps1
```

2. **Launch Streamlit App**
```powershell
streamlit run app.py
```

3. **Access the Application**
- Open browser and navigate to: `http://localhost:8501`
- The app will automatically open in your default browser

### Using the Application

1. **Select Disease Type** from the sidebar
2. **Enter Clinical Parameters** in the input form
3. **Click Predict Button** to get results
4. **View Results** with confidence scores and recommendations

### Example Use Cases

**Kidney Disease Prediction:**
- Enter patient's age, blood pressure, blood test results
- Get instant CKD risk assessment
- Receive medical consultation recommendations

**Liver Disease Prediction:**
- Input liver function test values
- Get liver disease probability
- View feature importance for diagnosis

**Parkinson's Detection:**
- Enter voice measurement features
- Detect early signs of Parkinson's
- Non-invasive screening method

## 📊 Model Performance

### Kidney Disease Model
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Extra Trees | 100.0% | 100.0% | 100.0% | 100.0% |
| Logistic Regression | 98.75% | 98.79% | 98.75% | 98.75% |
| KNN | 98.75% | 98.79% | 98.75% | 98.75% |
| Random Forest | 97.5% | 97.6% | 97.5% | 97.5% |

### Liver Disease Model
- **Accuracy:** 70.09%
- **Precision (Liver Disease):** 73%
- **Recall (Liver Disease):** 92%
- **F1-Score:** 81%

### Parkinson's Model
- **Accuracy:** 92.31%
- **Precision:** 92.19%
- **Recall:** 92.31%
- **F1-Score:** 92.17%

## 📁 Project Structure

```
new/
├── app.py                          # Main Streamlit application
├── train_kidney_model.py           # Kidney disease model training
├── train_liver_model.py            # Liver disease model training
├── train_parkinsons_model.py       # Parkinson's model training
├── requirements.txt                # Python dependencies
├── Rules.txt                       # Development rules
├── README.md                       # Project documentation
│
├── models/                         # Trained models (generated after training)
│   ├── kidney_disease_model.pkl
│   ├── kidney_disease_scaler.pkl
│   ├── liver_disease_model.pkl
│   ├── liver_disease_scaler.pkl
│   ├── parkinsons_model.pkl
│   └── parkinsons_scaler.pkl
│
├── kidney/                         # Kidney disease dataset
│   ├── kidney_disease.csv
│   └── [other files]
│
├── Liver/                          # Liver disease dataset
│   ├── indian_liver_patient.csv
│   └── [other files]
│
├── parkinson/                      # Parkinson's disease dataset
│   ├── parkinsons.data
│   └── [other files]
│
└── venv/                           # Virtual environment (not tracked)
```

## 📸 Screenshots

### Home Page
- Overview of all three disease prediction modules
- Technology stack information
- How it works section

### Kidney Disease Prediction
- 24 clinical parameter inputs
- Real-time prediction with confidence score
- Medical recommendations

### Liver Disease Prediction
- Liver function test inputs
- Disease probability calculation
- Feature importance visualization

### Parkinson's Disease Detection
- Voice measurement feature inputs
- XGBoost-based detection
- Early warning system

## 🔮 Future Enhancements

- [ ] Add data visualization dashboard
- [ ] Implement batch prediction from CSV upload
- [ ] Add model explainability (SHAP values)
- [ ] Create REST API for model serving
- [ ] Add user authentication system
- [ ] Implement model retraining pipeline
- [ ] Add more disease prediction modules
- [ ] Deploy to cloud (Heroku/AWS/Azure)
- [ ] Create mobile app version
- [ ] Add multilingual support

## 🎓 Learning Outcomes

This project demonstrates:
- **End-to-end ML pipeline** (data → training → deployment)
- **Multiple ML algorithms** comparison and selection
- **Feature engineering** and preprocessing techniques
- **Model persistence** and loading
- **Web application development** with Streamlit
- **Healthcare AI** applications
- **Software engineering** best practices

## 🏆 Resume-Worthy Highlights

✅ **Full-Stack ML Project** with deployment  
✅ **Real-world Healthcare Application**  
✅ **Multiple ML Algorithms** (10+ models)  
✅ **High Model Performance** (92-100% accuracy)  
✅ **Production-Ready Code** with proper structure  
✅ **Interactive Web Interface**  
✅ **Comprehensive Documentation**  
✅ **End-to-End Implementation**  

## ⚠️ Disclaimer

**IMPORTANT:** This application is developed for **educational and research purposes only**. 

- ❌ NOT a substitute for professional medical diagnosis
- ❌ NOT approved for clinical use
- ❌ NOT validated by medical authorities
- ✅ For learning and demonstration purposes only

**Always consult qualified healthcare professionals for medical decisions.**

## 👨‍💻 Author

Created as a comprehensive machine learning portfolio project demonstrating:
- Data Science skills
- ML Engineering capabilities
- Full-stack development
- Healthcare AI applications

## 📝 License

This project is created for educational purposes. Feel free to use it for learning and portfolio demonstration.

---

## 🚀 Quick Start Commands

```powershell
# Setup (First time only)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_kidney_model.py
python train_liver_model.py
python train_parkinsons_model.py

# Run Application
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## 📧 Contact & Support

For questions, suggestions, or collaboration:
- Add your contact information here
- Link to GitHub profile
- LinkedIn profile

---

**Made with ❤️ using Python, Streamlit, and Machine Learning**
