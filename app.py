
import streamlit as st
st.set_page_config(
    page_title="HealthPredict - AI Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
import joblib
import sys
import xgboost
import sklearn


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Enhanced Custom CSS with gradients and modern design
st.markdown("""
    <style>
    .info-box:empty {
        display: none !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        max-width: 1400px;
        padding: 0 1rem;
    }
    
    /* Make columns stretch to equal height */
    .row-widget.stHorizontal {
        align-items: stretch !important;
    }
    
    .row-widget.stHorizontal > div {
        display: flex !important;
        flex-direction: column !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        padding-bottom: 1.5rem;
        animation: fadeIn 1.5s ease-in;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        flex: 1;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .feature-card h3 {
        font-size: 1.15rem;
        margin-bottom: 0.8rem;
        color: #1f1f1f;
        font-weight: 600;
        line-height: 1.3;
    }
    
    .feature-card p {
        color: #333;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0.3rem 0;
    }
    
    .gradient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        flex: 1;
        height: 100%;
    }
    
    .gradient-card h3 {
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .gradient-card p {
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    .gradient-card h3 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .gradient-card p {
        font-size: 0.95rem;
        margin: 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    
    .info-box h2, .info-box h3, .info-box h4 {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    
    .info-box p, .info-box li {
        color: #2d2d2d !important;
        font-weight: 400;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1rem;
        }
        .feature-card h3, .gradient-card h3 {
            font-size: 1rem;
        }
    }
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .tooltip {
        background-color: #667eea;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to create confidence gauge
def create_confidence_gauge(confidence, title="Confidence"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': title, 'font': {'size': 24, 'color': '#667eea'}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "#764ba2", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Poppins"}
    )
    
    return fig

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
        st.markdown('<div class="info-box">✅ Model loaded successfully! Extra Trees Classifier with 100% accuracy</div>', unsafe_allow_html=True)
        
        # Sample Data Buttons
        st.markdown("### 🎯 Quick Test")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            use_sample_ckd = st.button("📊 Load Sample: CKD Patient", use_container_width=True)
        with col_btn2:
            use_sample_healthy = st.button("📊 Load Sample: Healthy Patient", use_container_width=True)
        
        # Initialize session state for values
        if 'kidney_values' not in st.session_state:
            st.session_state.kidney_values = {}
        
        # Sample data
        if use_sample_ckd:
            st.session_state.kidney_values = {
                'age': 65, 'bp': 90, 'sg': 1.015, 'al': 4, 'su': 0, 'rbc': 'abnormal',
                'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent', 'bgr': 150,
                'bu': 55, 'sc': 1.8, 'sod': 135, 'pot': 4.8, 'hemo': 10.5, 'pcv': 32,
                'wc': 9200, 'rc': 3.8, 'htn': 'yes', 'dm': 'yes', 'cad': 'no',
                'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
            }
            st.success("✅ Loaded sample CKD patient data")
        
        if use_sample_healthy:
            st.session_state.kidney_values = {
                'age': 40, 'bp': 80, 'sg': 1.025, 'al': 0, 'su': 0, 'rbc': 'normal',
                'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent', 'bgr': 140,
                'bu': 10, 'sc': 1.2, 'sod': 135, 'pot': 5.0, 'hemo': 15.0, 'pcv': 48,
                'wc': 10400, 'rc': 4.5, 'htn': 'no', 'dm': 'no', 'cad': 'no',
                'appet': 'good', 'pe': 'no', 'ane': 'no'
            }
            st.success("✅ Loaded sample healthy patient data")
        
        st.markdown("---")
        
        st.subheader("Basic Information")
        age = st.number_input("Age (years) 👤", 1, 120, 
                            st.session_state.kidney_values.get('age', 50),
                            help="Patient's age in years")
        bp = st.number_input("Blood Pressure (mm/Hg) 🫀", 50, 200, 
                           st.session_state.kidney_values.get('bp', 80),
                           help="Blood pressure measurement")
        sg = st.selectbox("Specific Gravity 💧", [1.005, 1.010, 1.015, 1.020, 1.025],
                        index=[1.005, 1.010, 1.015, 1.020, 1.025].index(
                            st.session_state.kidney_values.get('sg', 1.020)),
                        help="Urine specific gravity")
        al = st.selectbox("Albumin 🧪", [0, 1, 2, 3, 4, 5],
                        index=st.session_state.kidney_values.get('al', 0),
                        help="Albumin level in urine")
        su = st.selectbox("Sugar 🍬", [0, 1, 2, 3, 4, 5],
                        index=st.session_state.kidney_values.get('su', 0),
                        help="Sugar level in urine")
        rbc = st.selectbox("Red Blood Cells 🔴", ["normal", "abnormal"],
                         index=0 if st.session_state.kidney_values.get('rbc', 'normal')=='normal' else 1,
                         help="RBC in urine test")
        pc = st.selectbox("Pus Cell 🦠", ["normal", "abnormal"],
                        index=0 if st.session_state.kidney_values.get('pc', 'normal')=='normal' else 1,
                        help="Pus cells in urine")
        pcc = st.selectbox("Pus Cell Clumps 🧫", ["present", "notpresent"],
                         index=0 if st.session_state.kidney_values.get('pcc', 'notpresent')=='present' else 1,
                         help="Presence of pus cell clumps")
        ba = st.selectbox("Bacteria 🦠", ["present", "notpresent"],
                        index=0 if st.session_state.kidney_values.get('ba', 'notpresent')=='present' else 1,
                        help="Bacteria in urine")
        
        st.subheader("Blood Tests")
        bgr = st.number_input("Blood Glucose Random (mgs/dl) 🩸", 20, 500, 
                            st.session_state.kidney_values.get('bgr', 120),
                            help="Random blood glucose level")
        bu = st.number_input("Blood Urea (mgs/dl) 🧬", 1, 400, 
                           st.session_state.kidney_values.get('bu', 40),
                           help="Blood urea measurement")
        sc = st.number_input("Serum Creatinine (mgs/dl) ⚗️", 0.1, 20.0, 
                           float(st.session_state.kidney_values.get('sc', 1.2)), 0.1,
                           help="Serum creatinine level")
        sod = st.number_input("Sodium (mEq/L) 🧂", 50, 200, 
                            st.session_state.kidney_values.get('sod', 140),
                            help="Sodium level in blood")
        pot = st.number_input("Potassium (mEq/L) 🥔", 1.0, 20.0, 
                            float(st.session_state.kidney_values.get('pot', 4.5)), 0.1,
                            help="Potassium level in blood")
        hemo = st.number_input("Hemoglobin (gms) 🩸", 1.0, 20.0, 
                             float(st.session_state.kidney_values.get('hemo', 14.0)), 0.1,
                             help="Hemoglobin count")
        pcv = st.number_input("Packed Cell Volume 📊", 10, 60, 
                            st.session_state.kidney_values.get('pcv', 40),
                            help="Packed cell volume percentage")
        
        st.subheader("Additional Parameters")
        wc = st.number_input("White Blood Cell Count (cells/cumm) ⚪", 1000, 30000, 
                           st.session_state.kidney_values.get('wc', 8000),
                           help="WBC count")
        rc = st.number_input("Red Blood Cell Count (millions/cmm) 🔴", 1.0, 10.0, 
                           float(st.session_state.kidney_values.get('rc', 4.5)), 0.1,
                           help="RBC count")
        htn = st.selectbox("Hypertension 💔", ["yes", "no"],
                         index=0 if st.session_state.kidney_values.get('htn', 'no')=='yes' else 1,
                         help="History of hypertension")
        dm = st.selectbox("Diabetes Mellitus 🍭", ["yes", "no"],
                        index=0 if st.session_state.kidney_values.get('dm', 'no')=='yes' else 1,
                        help="History of diabetes")
        cad = st.selectbox("Coronary Artery Disease 🫀", ["yes", "no"],
                         index=0 if st.session_state.kidney_values.get('cad', 'no')=='yes' else 1,
                         help="Coronary artery disease history")
        appet = st.selectbox("Appetite 🍽️", ["good", "poor"],
                           index=0 if st.session_state.kidney_values.get('appet', 'good')=='good' else 1,
                           help="Patient's appetite")
        pe = st.selectbox("Pedal Edema 🦶", ["yes", "no"],
                        index=0 if st.session_state.kidney_values.get('pe', 'no')=='yes' else 1,
                        help="Swelling in feet")
        ane = st.selectbox("Anemia 🩸", ["yes", "no"],
                         index=0 if st.session_state.kidney_values.get('ane', 'no')=='yes' else 1,
                         help="Presence of anemia")
        
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
            
            confidence = prediction_proba[prediction]*100
            
            # Display confidence gauge
            col_gauge, col_result = st.columns([1, 1])
            
            with col_gauge:
                fig = create_confidence_gauge(confidence, "Prediction Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_result:
                if prediction in [0, 1]:  # CKD (ckd=0, ckd\t=1, notckd=2)
                    st.markdown(f'<div class="gradient-card"><h2>⚠️ Chronic Kidney Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
                    st.warning("⚠️ **Recommendation:** Immediate consultation with a nephrologist is advised.")
                else:  # prediction == 2 (notckd)
                    st.markdown(f'<div class="success-box"><h2>✅ No Chronic Kidney Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="info-box">✅ Model loaded successfully - Ready for prediction!</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sample data buttons
        st.subheader("🧪 Quick Test Samples")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            use_sample_liver = st.button("📊 Load Sample: Liver Disease Patient", use_container_width=True)
        with col_btn2:
            use_sample_healthy = st.button("📊 Load Sample: Healthy Patient", use_container_width=True)
        
        # Initialize session state
        if 'liver_values' not in st.session_state:
            st.session_state.liver_values = {}
        
        # Sample data: Liver disease patient (elevated enzymes and bilirubin)
        if use_sample_liver:
            st.session_state.liver_values = {
                'age': 55, 'gender': 'Male',
                'total_bilirubin': 2.5, 'direct_bilirubin': 0.9,
                'alkaline_phosphotase': 280, 'alamine_aminotransferase': 120,
                'aspartate_aminotransferase': 150, 'total_protiens': 6.2,
                'albumin': 2.8, 'ag_ratio': 0.82
            }
            st.success("✅ Loaded sample liver disease patient data")
        
        # Sample data: Healthy patient (normal liver function)
        elif use_sample_healthy:
            st.session_state.liver_values = {
                'age': 35, 'gender': 'Female',
                'total_bilirubin': 0.7, 'direct_bilirubin': 0.2,
                'alkaline_phosphotase': 175, 'alamine_aminotransferase': 28,
                'aspartate_aminotransferase': 32, 'total_protiens': 7.5,
                'albumin': 4.2, 'ag_ratio': 1.27
            }
            st.success("✅ Loaded sample healthy patient data")
        
        st.markdown("---")
        
        st.subheader("Patient Information")
        age = st.number_input("Age (years) 👤", 1, 120, value=st.session_state.liver_values.get('age', 45), help="Patient's age in years")
        gender = st.selectbox("Gender ⚧️", ["Male", "Female"], index=0 if st.session_state.liver_values.get('gender', 'Male') == 'Male' else 1, help="Patient's biological gender")

        st.subheader("Liver Function Tests")
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL) 🟡", 0.1, 100.0, value=st.session_state.liver_values.get('total_bilirubin', 1.0), step=0.1, help="Total bilirubin level in blood (normal: 0.1-1.2 mg/dL)")
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL) 🟠", 0.1, 50.0, value=st.session_state.liver_values.get('direct_bilirubin', 0.3), step=0.1, help="Direct bilirubin level (normal: 0.0-0.3 mg/dL)")
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L) ⚗️", 10, 2000, value=st.session_state.liver_values.get('alkaline_phosphotase', 200), help="ALP enzyme level (normal: 44-147 IU/L)")
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (IU/L) 🧪", 1, 2000, value=st.session_state.liver_values.get('alamine_aminotransferase', 30), help="ALT/SGPT enzyme level (normal: 7-56 IU/L)")
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (IU/L) 🔬", 1, 2000, value=st.session_state.liver_values.get('aspartate_aminotransferase', 35), help="AST/SGOT enzyme level (normal: 10-40 IU/L)")
        total_protiens = st.number_input("Total Proteins (g/dL) 🥩", 1.0, 15.0, value=st.session_state.liver_values.get('total_protiens', 7.0), step=0.1, help="Total protein in blood (normal: 6.0-8.3 g/dL)")
        albumin = st.number_input("Albumin (g/dL) 🧬", 0.5, 10.0, value=st.session_state.liver_values.get('albumin', 4.0), step=0.1, help="Albumin protein level (normal: 3.5-5.5 g/dL)")
        ag_ratio = st.number_input("Albumin/Globulin Ratio 📊", 0.1, 5.0, value=st.session_state.liver_values.get('ag_ratio', 1.0), step=0.1, help="A/G ratio (normal: 1.0-2.5)")
        
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
            
            confidence = prediction_proba[prediction]*100
            
            # Display confidence gauge and result
            col_gauge, col_result = st.columns([1, 1])
            
            with col_gauge:
                fig = create_confidence_gauge(confidence, "Prediction Confidence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_result:
                if prediction == 1:  # Liver Disease
                    st.markdown(f'<div class="gradient-card"><h2>⚠️ Liver Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
                    st.warning("⚠️ **Recommendation:** Consult a hepatologist for further evaluation.")
                else:
                    st.markdown(f'<div class="success-box"><h2>✅ No Liver Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="info-box">✅ Model loaded successfully - Ready for prediction!</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.info("📝 **Note:** This model uses 22 voice measurement features. You can either enter values manually or upload a CSV file.")
        
        input_method = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])
        
        if input_method == "Manual Entry":
            # Sample data buttons
            st.markdown('<span style="font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:0.5em;">🧪 Quick Test Samples</span>', unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                use_sample_pd = st.button("📊 Load Sample: Parkinson's Patient", use_container_width=True)
            with col_btn2:
                use_sample_healthy = st.button("📊 Load Sample: Healthy Person", use_container_width=True)
            
            # Initialize session state
            if 'parkinsons_values' not in st.session_state:
                st.session_state.parkinsons_values = {}
            
            # Sample data: Parkinson's patient (high jitter/shimmer, low HNR)
            if use_sample_pd:
                st.session_state.parkinsons_values = {
                    'mdvp_fo': 197.0, 'mdvp_fhi': 206.0, 'mdvp_flo': 192.0,
                    'mdvp_jitter_percent': 0.00784, 'mdvp_jitter_abs': 0.00007,
                    'mdvp_rap': 0.00370, 'mdvp_ppq': 0.00554, 'jitter_ddp': 0.01109,
                    'mdvp_shimmer': 0.04374, 'mdvp_shimmer_db': 0.426, 'shimmer_apq3': 0.02182,
                    'shimmer_apq5': 0.03130, 'mdvp_apq': 0.02971, 'shimmer_dda': 0.06545,
                    'nhr': 0.02211, 'hnr': 21.033, 'rpde': 0.414783, 'dfa': 0.815285,
                    'spread1': -4.813031, 'spread2': 0.266482, 'd2': 2.301442, 'ppe': 0.284654
                }
                st.success("✅ Loaded sample Parkinson's patient voice data")
            
            # Sample data: Healthy person (low jitter/shimmer, high HNR)
            elif use_sample_healthy:
                st.session_state.parkinsons_values = {
                    'mdvp_fo': 119.992, 'mdvp_fhi': 157.302, 'mdvp_flo': 74.997,
                    'mdvp_jitter_percent': 0.00289, 'mdvp_jitter_abs': 0.00001,
                    'mdvp_rap': 0.00166, 'mdvp_ppq': 0.00168, 'jitter_ddp': 0.00498,
                    'mdvp_shimmer': 0.01888, 'mdvp_shimmer_db': 0.164, 'shimmer_apq3': 0.00954,
                    'shimmer_apq5': 0.01070, 'mdvp_apq': 0.01767, 'shimmer_dda': 0.02862,
                    'nhr': 0.00777, 'hnr': 26.775, 'rpde': 0.422229, 'dfa': 0.741367,
                    'spread1': -7.348300, 'spread2': 0.177551, 'd2': 1.743867, 'ppe': 0.085569
                }
                st.success("✅ Loaded sample healthy person voice data")
            
            st.markdown("---")
            
            st.markdown('<span style="font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:0.5em;">Frequency Measures <span style="font-size:2rem;">🎵</span></span>', unsafe_allow_html=True)
            st.markdown('<span style="font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:0.5em;">Jitter Measures <span style="font-size:2rem;">🪁</span></span>', unsafe_allow_html=True)
            mdvp_fo = st.number_input("Fo 📢", 50.0, 300.0,
                                     value=st.session_state.parkinsons_values.get('mdvp_fo', 150.0),
                                     step=0.1,
                                     help="Average vocal fundamental frequency (normal: 80-260 Hz)")
            mdvp_fhi = st.number_input("Fhi ⬆️", 100.0, 600.0,
                                      value=st.session_state.parkinsons_values.get('mdvp_fhi', 180.0),
                                      step=0.1,
                                      help="Maximum vocal fundamental frequency")
            mdvp_flo = st.number_input("Flo ⬇️", 50.0, 300.0,
                                      value=st.session_state.parkinsons_values.get('mdvp_flo', 100.0),
                                      step=0.1,
                                      help="Minimum vocal fundamental frequency")
            mdvp_jitter_percent = st.number_input("Jitter% 📊", 0.0, 1.0,
                                                 value=st.session_state.parkinsons_values.get('mdvp_jitter_percent', 0.005),
                                                 step=0.001, format="%.5f",
                                                 help="Measure of variation in frequency (lower is better)")
            mdvp_jitter_abs = st.number_input("JitterAbs 🔍", 0.0, 0.001,
                                             value=st.session_state.parkinsons_values.get('mdvp_jitter_abs', 0.00003),
                                             step=0.000001, format="%.8f",
                                             help="Absolute jitter in microseconds")
            mdvp_rap = st.number_input("RAP 📈", 0.0, 0.1,
                                      value=st.session_state.parkinsons_values.get('mdvp_rap', 0.003),
                                      step=0.001, format="%.5f",
                                      help="Relative Amplitude Perturbation")
            mdvp_ppq = st.number_input("PPQ 📉", 0.0, 0.1,
                                      value=st.session_state.parkinsons_values.get('mdvp_ppq', 0.003),
                                      step=0.001, format="%.5f",
                                      help="Five-point Period Perturbation Quotient")
            jitter_ddp = st.number_input("DDP 📐", 0.0, 0.1,
                                        value=st.session_state.parkinsons_values.get('jitter_ddp', 0.008),
                                        step=0.001, format="%.5f",
                                        help="Average absolute difference of differences")
            
            st.markdown('<span style="font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:0.5em;">Shimmer Measures <span style="font-size:2rem;">🌊</span></span>', unsafe_allow_html=True)
            mdvp_shimmer = st.number_input("Shimmer 📊", 0.0, 1.0,
                                          value=st.session_state.parkinsons_values.get('mdvp_shimmer', 0.03),
                                          step=0.001, format="%.5f",
                                          help="Measure of variation in amplitude")
            mdvp_shimmer_db = st.number_input("ShimmerdB 🔊", 0.0, 2.0,
                                         value=st.session_state.parkinsons_values.get('mdvp_shimmer_db', 0.3),
                                         step=0.01,
                                         help="Shimmer in decibels")
            shimmer_apq3 = st.number_input("APQ3 3️⃣", 0.0, 0.1,
                                          value=st.session_state.parkinsons_values.get('shimmer_apq3', 0.015),
                                          step=0.001, format="%.5f",
                                          help="Three-point Amplitude Perturbation Quotient")
            shimmer_apq5 = st.number_input("APQ5 5️⃣", 0.0, 0.1,
                                          value=st.session_state.parkinsons_values.get('shimmer_apq5', 0.017),
                                          step=0.001, format="%.5f",
                                          help="Five-point Amplitude Perturbation Quotient")
            mdvp_apq = st.number_input("APQ 📏", 0.0, 0.1,
                                      value=st.session_state.parkinsons_values.get('mdvp_apq', 0.024),
                                      step=0.001, format="%.5f",
                                      help="11-point Amplitude Perturbation Quotient")
            shimmer_dda = st.number_input("DDA 〰️", 0.0, 0.1,
                                         value=st.session_state.parkinsons_values.get('shimmer_dda', 0.045),
                                         step=0.001, format="%.5f",
                                         help="Average absolute difference between amplitudes")
            
            st.markdown('<span style="font-size:1.5rem;font-weight:700;display:flex;align-items:center;gap:0.5em;">Other Measures <span style="font-size:2rem;">⚡</span></span>', unsafe_allow_html=True)
            nhr = st.number_input("NHR 🔉", 0.0, 1.0,
                                 value=st.session_state.parkinsons_values.get('nhr', 0.025),
                                 step=0.001, format="%.5f",
                                 help="Noise-to-Harmonics Ratio (lower is better)")
            hnr = st.number_input("HNR 🎼", 5.0, 35.0,
                                 value=st.session_state.parkinsons_values.get('hnr', 20.0),
                                 step=0.1,
                                 help="Harmonics-to-Noise Ratio (higher is better)")
            rpde = st.number_input("RPDE 🔄", 0.2, 0.8,
                                  value=st.session_state.parkinsons_values.get('rpde', 0.5),
                                  step=0.01,
                                  help="Recurrence Period Density Entropy")
            dfa = st.number_input("DFA 📈", 0.5, 0.9,
                                 value=st.session_state.parkinsons_values.get('dfa', 0.7),
                                 step=0.01,
                                 help="Detrended Fluctuation Analysis")
            spread1 = st.number_input("Spread1 📊", -8.0, -2.0,
                                     value=st.session_state.parkinsons_values.get('spread1', -5.0),
                                     step=0.1,
                                     help="Nonlinear measure of fundamental frequency variation")
            spread2 = st.number_input("Spread2 📉", 0.0, 0.5,
                                     value=st.session_state.parkinsons_values.get('spread2', 0.2),
                                     step=0.01,
                                     help="Second nonlinear measure of variation")
            d2 = st.number_input("D2 🌀", 1.0, 4.0,
                                value=st.session_state.parkinsons_values.get('d2', 2.5),
                                step=0.1,
                                help="Correlation dimension")
            ppe = st.number_input("PPE 🎯", 0.0, 0.7,
                                 value=st.session_state.parkinsons_values.get('ppe', 0.2),
                                 step=0.01,
                                 help="Pitch Period Entropy")
            
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
                
                confidence = prediction_proba[prediction]*100
                
                # Display confidence gauge and result
                col_gauge, col_result = st.columns([1, 1])
                
                with col_gauge:
                    fig = create_confidence_gauge(confidence, "Prediction Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_result:
                    if prediction == 1:  # Parkinson's
                        st.markdown(f'<div class="gradient-card"><h2>⚠️ Parkinson\'s Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
                        st.warning("⚠️ **Recommendation:** Consult a neurologist for comprehensive evaluation.")
                    else:
                        st.markdown(f'<div class="success-box"><h2>✅ No Parkinson\'s Disease Detected</h2><p style="font-size:20px;">Confidence: {confidence:.2f}%</p></div>', unsafe_allow_html=True)
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
    st.header("Welcome to HealthPredict")
    st.write("""
    HealthPredict is an advanced machine learning platform designed to assist in the early detection 
    of chronic diseases. Using state-of-the-art algorithms and clinical data, our system can predict 
    the likelihood of three major chronic conditions with high accuracy.
    """)
    
    # Features
    st.header("🎯 Available Disease Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🫘 Chronic Kidney Disease</h3>
            <p><b>24 Clinical Features</b></p>
            <p><b>Extra Trees Algorithm</b></p>
            <p><b>100% Accuracy</b></p>
            <p>Detects CKD using blood tests and clinical parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🫀 Liver Disease</h3>
            <p><b>10 Clinical Features</b></p>
            <p><b>Random Forest</b></p>
            <p><b>70% Accuracy</b></p>
            <p>Predicts liver disease from liver function tests</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Parkinson's Disease</h3>
            <p><b>22 Voice Features</b></p>
            <p><b>XGBoost Algorithm</b></p>
            <p><b>92.31% Accuracy</b></p>
            <p>Detects PD from voice measurements</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.header("🔬 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="gradient-card">
            <h3>1️⃣ Input Data</h3>
            <p>Enter patient clinical parameters through our intuitive interface</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="gradient-card">
            <h3>2️⃣ Processing</h3>
            <p>Data is preprocessed, normalized and validated</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="gradient-card">
            <h3>3️⃣ Prediction</h3>
            <p>ML models analyze the data with advanced algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="gradient-card">
            <h3>4️⃣ Results</h3>
            <p>Get instant prediction with confidence score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.header("💻 Technology Stack")
    st.markdown('''
    <div class="info-box">
        <h3>Machine Learning</h3>
        <ul>
            <li>Scikit-learn</li>
            <li>XGBoost</li>
            <li>PyCaret</li>
            <li>TensorFlow/Keras</li>
        </ul>
        <h3>Visualization & Deployment</h3>
        <ul>
            <li>Streamlit</li>
            <li>Plotly</li>
            <li>Pandas & NumPy</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown("""
    <div class="info-box" style="border-color: #ff6b6b;">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This tool is for educational and research purposes only. 
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

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
