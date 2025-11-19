import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="HealthPredict - AI Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 30px;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏥 HealthPredict")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🫘 Kidney Disease", "🫀 Liver Disease", "🧠 Parkinson's Disease", "📊 About Models"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**HealthPredict** uses advanced machine learning algorithms "
    "to predict the likelihood of chronic diseases based on clinical parameters."
)

# Helper function to load models
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Error loading {model_path}: {str(e)}")
        return None

# Kidney Disease Prediction Page
if page == "🫘 Kidney Disease":
    st.title("🫘 Chronic Kidney Disease Prediction")
    st.markdown("Enter the patient's clinical parameters to predict Chronic Kidney Disease (CKD)")
    
    # Load model (Extra Trees Classifier - 100% accuracy)
    model = load_model('models/kidney/kidney_extratrees_basic.pkl')
    scaler = load_model('models/kidney/kidney_extratrees_scaler.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_kidney_extratrees.py`")
    else:
        st.success("✅ Model loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Information")
            age = st.number_input("Age (years)", 1, 120, 50)
            bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
            al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
            su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
        
        with col2:
            st.subheader("Blood Tests")
            ba = st.selectbox("Bacteria", ["present", "notpresent"])
            bgr = st.number_input("Blood Glucose Random (mgs/dl)", 20, 500, 120)
            bu = st.number_input("Blood Urea (mgs/dl)", 1, 400, 40)
            sc = st.number_input("Serum Creatinine (mgs/dl)", 0.1, 20.0, 1.2, 0.1)
            sod = st.number_input("Sodium (mEq/L)", 50, 200, 140)
            pot = st.number_input("Potassium (mEq/L)", 1.0, 20.0, 4.5, 0.1)
            hemo = st.number_input("Hemoglobin (gms)", 1.0, 20.0, 14.0, 0.1)
            pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
        
        with col3:
            st.subheader("Additional Parameters")
            wc = st.number_input("White Blood Cell Count (cells/cumm)", 1000, 30000, 8000)
            rc = st.number_input("Red Blood Cell Count (millions/cmm)", 1.0, 10.0, 4.5, 0.1)
            htn = st.selectbox("Hypertension", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])
        
        if st.button("🔍 Predict Kidney Disease", type="primary"):
            # Encode categorical variables
            rbc_enc = 1 if rbc == "abnormal" else 0
            pc_enc = 1 if pc == "abnormal" else 0
            pcc_enc = 1 if pcc == "present" else 0
            ba_enc = 1 if ba == "present" else 0
            htn_enc = 1 if htn == "yes" else 0
            dm_enc = 1 if dm == "yes" else 0
            cad_enc = 1 if cad == "yes" else 0
            appet_enc = 1 if appet == "poor" else 0
            pe_enc = 1 if pe == "yes" else 0
            ane_enc = 1 if ane == "yes" else 0
            
            # Create feature array
            features = np.array([[age, bp, sg, al, su, rbc_enc, pc_enc, pcc_enc, ba_enc, bgr, 
                                bu, sc, sod, pot, hemo, pcv, wc, rc, htn_enc, dm_enc, cad_enc, 
                                appet_enc, pe_enc, ane_enc]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            if prediction == 2:  # CKD
                st.error(f"### ⚠️ Chronic Kidney Disease Detected")
                st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                st.warning("⚠️ **Recommendation:** Immediate consultation with a nephrologist is advised.")
            else:
                st.success(f"### ✅ No Chronic Kidney Disease Detected")
                st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                st.info("💡 **Recommendation:** Maintain a healthy lifestyle and regular checkups.")

# Liver Disease Prediction Page
elif page == "🫀 Liver Disease":
    st.title("🫀 Liver Disease Prediction")
    st.markdown("Enter the patient's liver function test parameters to predict liver disease")
    
    # Load model (Random Forest - 70% accuracy)
    model = load_model('models/liver/liver_randomforest_basic.pkl')
    scaler = load_model('models/liver/liver_randomforest_scaler.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_liver_randomforest.py`")
    else:
        st.success("✅ Model loaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.number_input("Age (years)", 1, 120, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.1, 100.0, 1.0, 0.1)
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 50.0, 0.3, 0.1)
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", 10, 2000, 200)
        
        with col2:
            st.subheader("Liver Function Tests")
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase (IU/L)", 1, 2000, 30)
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (IU/L)", 1, 2000, 35)
            total_protiens = st.number_input("Total Proteins (g/dL)", 1.0, 15.0, 7.0, 0.1)
            albumin = st.number_input("Albumin (g/dL)", 0.5, 10.0, 4.0, 0.1)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.1, 5.0, 1.0, 0.1)
        
        if st.button("🔍 Predict Liver Disease", type="primary"):
            # Encode gender
            gender_enc = 1 if gender == "Male" else 0
            
            # Create feature array
            features = np.array([[age, gender_enc, total_bilirubin, direct_bilirubin, 
                                alkaline_phosphotase, alamine_aminotransferase, 
                                aspartate_aminotransferase, total_protiens, albumin, ag_ratio]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:  # Liver Disease
                st.error(f"### ⚠️ Liver Disease Detected")
                st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                st.warning("⚠️ **Recommendation:** Consult a hepatologist for further evaluation.")
            else:
                st.success(f"### ✅ No Liver Disease Detected")
                st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                st.info("💡 **Recommendation:** Maintain a healthy lifestyle and avoid excessive alcohol.")

# Parkinson's Disease Prediction Page
elif page == "🧠 Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Detection")
    st.markdown("Enter voice measurement features to detect Parkinson's Disease")
    
    # Load model (XGBoost - 92.31% accuracy)
    model = load_model('models/parkinsons/parkinsons_xgboost_basic.pkl')
    scaler = load_model('models/parkinsons/parkinsons_xgboost_scaler.pkl')
    features_list = load_model('models/parkinsons/parkinsons_xgboost_features.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_parkinsons_xgboost.py`")
    else:
        st.success("✅ Model loaded successfully!")
        
        st.info("📝 **Note:** This model uses 22 voice measurement features. You can either enter values manually or upload a CSV file.")
        
        input_method = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])
        
        if input_method == "Manual Entry":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Frequency Measures")
                mdvp_fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 150.0, 0.1)
                mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 180.0, 0.1)
                mdvp_flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 100.0, 0.1)
                mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.005, 0.001)
                mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001, 0.00003, 0.000001)
                mdvp_rap = st.number_input("MDVP:RAP", 0.0, 0.1, 0.003, 0.001)
                mdvp_ppq = st.number_input("MDVP:PPQ", 0.0, 0.1, 0.003, 0.001)
                jitter_ddp = st.number_input("Jitter:DDP", 0.0, 0.1, 0.008, 0.001)
            
            with col2:
                st.subheader("Shimmer Measures")
                mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.03, 0.001)
                mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 2.0, 0.3, 0.01)
                shimmer_apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.015, 0.001)
                shimmer_apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.1, 0.017, 0.001)
                mdvp_apq = st.number_input("MDVP:APQ", 0.0, 0.1, 0.024, 0.001)
                shimmer_dda = st.number_input("Shimmer:DDA", 0.0, 0.1, 0.045, 0.001)
            
            with col3:
                st.subheader("Other Measures")
                nhr = st.number_input("NHR", 0.0, 1.0, 0.025, 0.001)
                hnr = st.number_input("HNR", 5.0, 35.0, 20.0, 0.1)
                rpde = st.number_input("RPDE", 0.2, 0.8, 0.5, 0.01)
                dfa = st.number_input("DFA", 0.5, 0.9, 0.7, 0.01)
                spread1 = st.number_input("spread1", -8.0, -2.0, -5.0, 0.1)
                spread2 = st.number_input("spread2", 0.0, 0.5, 0.2, 0.01)
                d2 = st.number_input("D2", 1.0, 4.0, 2.5, 0.1)
                ppe = st.number_input("PPE", 0.0, 0.7, 0.2, 0.01)
            
            if st.button("🔍 Detect Parkinson's Disease", type="primary"):
                # Create feature array
                features = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, 
                                    mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp,
                                    mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5,
                                    mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, 
                                    spread2, d2, ppe]])
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                if prediction == 1:  # Parkinson's
                    st.error(f"### ⚠️ Parkinson's Disease Detected")
                    st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                    st.warning("⚠️ **Recommendation:** Consult a neurologist for comprehensive evaluation.")
                else:
                    st.success(f"### ✅ No Parkinson's Disease Detected")
                    st.write(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
                    st.info("💡 **Recommendation:** Continue regular health monitoring.")
        else:
            uploaded_file = st.file_uploader("Upload CSV file with voice features", type=['csv'])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(data.head())
                
                if st.button("🔍 Predict from CSV"):
                    st.info("CSV prediction feature coming soon!")

# Home Page
elif page == "🏠 Home":
    st.markdown('<div class="main-header">🏥 HealthPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Multi-Disease Detection Platform</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    st.header("Welcome to HealthPredict")
    st.write("""
    HealthPredict is an advanced machine learning platform designed to assist in the early detection 
    of chronic diseases. Using state-of-the-art algorithms and clinical data, our system can predict 
    the likelihood of three major chronic conditions.
    """)
    
    # Features
    st.header("🎯 Available Disease Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🫘 Chronic Kidney Disease")
        st.write("""
        - **24 Clinical Features**
        - **11 ML Algorithms**
        - **98%+ Accuracy**
        - Detects CKD using blood tests and clinical parameters
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🫀 Liver Disease")
        st.write("""
        - **10 Clinical Features**
        - **PyCaret AutoML**
        - **High Accuracy**
        - Predicts liver disease from liver function tests
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("🧠 Parkinson's Disease")
        st.write("""
        - **24 Voice Features**
        - **XGBoost Algorithm**
        - **95%+ Accuracy**
        - Detects PD from voice measurements
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.header("🔬 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1️⃣ Input Data")
        st.write("Enter patient clinical parameters")
    
    with col2:
        st.markdown("### 2️⃣ Processing")
        st.write("Data is preprocessed and normalized")
    
    with col3:
        st.markdown("### 3️⃣ Prediction")
        st.write("ML models analyze the data")
    
    with col4:
        st.markdown("### 4️⃣ Results")
        st.write("Get prediction with confidence score")
    
    st.markdown("---")
    
    # Technology Stack
    st.header("💻 Technology Stack")
    st.write("""
    - **Machine Learning:** Scikit-learn, XGBoost, PyCaret, TensorFlow
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn, Plotly
    - **Web Framework:** Streamlit
    - **Deployment:** Python 3.8+
    """)
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. 
    It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions.
    """)

# About Models Page
elif page == "📊 About Models":
    st.title("📊 Model Information")
    
    st.markdown("---")
    
    # Kidney Disease Model
    st.header("🫘 Chronic Kidney Disease Model")
    st.write("""
    **Dataset:** 400 patients with 24 clinical attributes
    
    **Features Used:**
    - Age, Blood Pressure, Specific Gravity
    - Albumin, Sugar, Red Blood Cells
    - Blood Glucose, Blood Urea, Serum Creatinine
    - Sodium, Potassium, Hemoglobin
    - And 12 more clinical parameters
    
    **Algorithms Compared:**
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    - Decision Tree
    - Random Forest
    - XGBoost
    - Extra Trees
    - AdaBoost
    - Gaussian Naive Bayes
    - Neural Network
    
    **Best Model:** Random Forest / Extra Trees Classifier
    """)
    
    st.markdown("---")
    
    # Liver Disease Model
    st.header("🫀 Liver Disease Model")
    st.write("""
    **Dataset:** 583 patients (416 liver patients, 167 non-liver patients)
    
    **Features Used:**
    - Age, Gender
    - Total Bilirubin, Direct Bilirubin
    - Alkaline Phosphatase
    - Alamine Aminotransferase
    - Aspartate Aminotransferase
    - Total Proteins, Albumin
    - Albumin/Globulin Ratio
    
    **Method:** PyCaret AutoML
    
    **Preprocessing:**
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    """)
    
    st.markdown("---")
    
    # Parkinson's Disease Model
    st.header("🧠 Parkinson's Disease Model")
    st.write("""
    **Dataset:** 195 voice recordings (147 with PD, 48 healthy)
    
    **Features Used:**
    - 24 voice measurement features
    - Jitter, Shimmer variations
    - Noise-to-Harmonics Ratio
    - Fundamental frequency measures
    - And more vocal biomarkers
    
    **Algorithm:** XGBoost (eXtreme Gradient Boosting)
    
    **Why XGBoost?**
    - Excellent performance on tabular data
    - Handles complex patterns
    - Built-in regularization
    - Fast training and prediction
    """)
    
    st.markdown("---")
    
    st.info("""
    📈 **Model Performance Metrics:**
    All models are evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC Score
    - Confusion Matrix
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed for Educational Purposes**")
st.sidebar.markdown("© 2025 HealthPredict")
