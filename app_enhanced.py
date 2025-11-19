import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="HealthPredict - AI Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #43a047;
        --danger-color: #e53935;
        --warning-color: #fb8c00;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 30px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #555;
        text-align: center;
        padding-bottom: 40px;
        font-weight: 300;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
        color: #333;
    }
    
    .feature-card h3, .feature-card h4 {
        color: #667eea;
    }
    
    .feature-card p, .feature-card ul {
        color: #333;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Disease cards */
    .disease-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-top: 4px solid #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.3);
    }
    
    .result-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        color: #333;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .info-box h3, .info-box h4 {
        color: #667eea;
        margin-top: 0;
    }
    
    .info-box p, .info-box ul {
        color: #333;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* Input labels */
    label {
        font-weight: 500 !important;
        color: #333 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation with enhanced styling
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>🏥 HealthPredict</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8);'>AI-Powered Disease Detection</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🫘 Kidney Disease", "🫀 Liver Disease", "🧠 Parkinson's Disease", "📊 About Models", "ℹ️ How to Use"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;'>
<p style='margin: 0; font-size: 0.9rem;'>
<b>HealthPredict</b> uses state-of-the-art machine learning algorithms to predict chronic diseases from clinical parameters.
</p>
</div>
""", unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Error loading {model_path}: {str(e)}")
        return None

# Helper function to create confidence gauge
def create_confidence_gauge(confidence, title="Confidence"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#4facfe'},
                {'range': [50, 75], 'color': '#fb8c00'},
                {'range': [75, 100], 'color': '#e53935'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header animated">🏥 HealthPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header animated">AI-Powered Multi-Disease Detection Platform</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    st.markdown('<div class="section-header">✨ Welcome to the Future of Healthcare</div>', unsafe_allow_html=True)
    st.write("""
    HealthPredict leverages cutting-edge machine learning algorithms to assist healthcare professionals 
    in early disease detection. Our platform combines clinical expertise with artificial intelligence 
    to provide fast, accurate preliminary assessments for three major chronic conditions.
    """)
    
    # Key Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Disease Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98%</div>
            <div class="metric-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">56</div>
            <div class="metric-label">Clinical Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10+</div>
            <div class="metric-label">ML Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    st.markdown('<div class="section-header">🎯 Disease Detection Modules</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🫘 Chronic Kidney Disease</h3>
            <p><b>Algorithm:</b> Extra Trees Classifier</p>
            <p><b>Accuracy:</b> 100% on test set</p>
            <p><b>Features:</b> 24 clinical parameters</p>
            <p><b>Dataset:</b> 400 patients</p>
            <hr>
            <p>Analyzes blood tests, urine tests, and clinical history to detect CKD in early stages.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🫀 Liver Disease</h3>
            <p><b>Algorithm:</b> Random Forest</p>
            <p><b>Accuracy:</b> 70% on test set</p>
            <p><b>Features:</b> 10 liver function tests</p>
            <p><b>Dataset:</b> 583 patients</p>
            <hr>
            <p>Evaluates liver enzyme levels and protein markers to identify liver dysfunction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Parkinson's Disease</h3>
            <p><b>Algorithm:</b> XGBoost</p>
            <p><b>Accuracy:</b> 92.31% on test set</p>
            <p><b>Features:</b> 22 voice measurements</p>
            <p><b>Dataset:</b> 195 recordings</p>
            <hr>
            <p>Uses advanced voice analysis to detect early motor symptoms of Parkinson's disease.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.markdown('<div class="section-header">🔬 How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3>1️⃣</h3>
            <h4>Input Data</h4>
            <p>Enter patient clinical parameters through intuitive forms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3>2️⃣</h3>
            <h4>Preprocessing</h4>
            <p>Data is cleaned, normalized, and prepared for analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3>3️⃣</h3>
            <h4>AI Analysis</h4>
            <p>Machine learning models process and analyze patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h3>4️⃣</h3>
            <h4>Results</h4>
            <p>Get predictions with confidence scores and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🤖 Machine Learning</h4>
            <ul>
                <li>Scikit-learn - Core ML algorithms</li>
                <li>XGBoost - Gradient boosting</li>
                <li>PyCaret - AutoML framework</li>
                <li>TensorFlow/Keras - Deep learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🛠️ Development Tools</h4>
            <ul>
                <li>Streamlit - Web interface</li>
                <li>Pandas & NumPy - Data processing</li>
                <li>Plotly - Interactive visualizations</li>
                <li>Python 3.8+ - Core language</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to action
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    st.info("👈 Select a disease module from the sidebar to begin making predictions!")
    
    # Disclaimer
    st.markdown("---")
    st.error("""
    ⚠️ **Important Medical Disclaimer**
    
    This tool is designed for **educational and research purposes only**. It should NOT be used as:
    - A replacement for professional medical diagnosis
    - The sole basis for treatment decisions
    - A substitute for laboratory tests or clinical examination
    
    **Always consult qualified healthcare professionals** for medical advice, diagnosis, and treatment.
    The predictions provided by this system are probabilistic estimates and may not reflect actual medical conditions.
    """)

# Kidney Disease Page
elif page == "🫘 Kidney Disease":
    st.markdown('<div class="main-header">🫘 Chronic Kidney Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown("### Enter patient clinical parameters for CKD assessment")
    
    # Load models
    model = load_model('models/kidney/kidney_extratrees_basic.pkl')
    scaler = load_model('models/kidney/kidney_extratrees_scaler.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_kidney_extratrees.py`")
    else:
        st.success("✅ Model loaded successfully! (Extra Trees Classifier - 100% Accuracy)")
        
        # Sample data button
        use_sample = st.checkbox("🎲 Use sample patient data for quick test")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="section-header">📋 Basic Information</div>', unsafe_allow_html=True)
            age = st.number_input("Age (years)", 1, 120, 65 if use_sample else 50, help="Patient's age in years")
            bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80 if use_sample else 80, help="Diastolic blood pressure")
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2 if use_sample else 2)
            al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5], index=0)
            su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5], index=0)
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], index=1 if use_sample else 0)
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"], index=1 if use_sample else 0)
            pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"], index=0 if use_sample else 1)
        
        with col2:
            st.markdown('<div class="section-header">🩸 Blood Tests</div>', unsafe_allow_html=True)
            ba = st.selectbox("Bacteria", ["present", "notpresent"], index=0 if use_sample else 1)
            bgr = st.number_input("Blood Glucose Random (mgs/dl)", 20, 500, 121 if use_sample else 120)
            bu = st.number_input("Blood Urea (mgs/dl)", 1, 400, 40 if use_sample else 40)
            sc = st.number_input("Serum Creatinine (mgs/dl)", 0.1, 20.0, 1.2 if use_sample else 1.2, 0.1)
            sod = st.number_input("Sodium (mEq/L)", 50, 200, 140 if use_sample else 140)
            pot = st.number_input("Potassium (mEq/L)", 1.0, 20.0, 4.5 if use_sample else 4.5, 0.1)
            hemo = st.number_input("Hemoglobin (gms)", 1.0, 20.0, 14.0 if use_sample else 14.0, 0.1)
            pcv = st.number_input("Packed Cell Volume", 10, 60, 40 if use_sample else 40)
        
        with col3:
            st.markdown('<div class="section-header">🏥 Additional Parameters</div>', unsafe_allow_html=True)
            wc = st.number_input("White Blood Cell Count (cells/cumm)", 1000, 30000, 8000 if use_sample else 8000)
            rc = st.number_input("Red Blood Cell Count (millions/cmm)", 1.0, 10.0, 4.5 if use_sample else 4.5, 0.1)
            htn = st.selectbox("Hypertension", ["yes", "no"], index=0 if use_sample else 1)
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=0 if use_sample else 1)
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"], index=0 if use_sample else 1)
            appet = st.selectbox("Appetite", ["good", "poor"], index=1 if use_sample else 0)
            pe = st.selectbox("Pedal Edema", ["yes", "no"], index=0 if use_sample else 1)
            ane = st.selectbox("Anemia", ["yes", "no"], index=0 if use_sample else 1)
        
        st.markdown("---")
        
        if st.button("🔍 Predict Kidney Disease", type="primary"):
            with st.spinner("Analyzing clinical parameters..."):
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
                
                # Display results with enhanced visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 2:  # CKD
                        st.markdown("""
                        <div class="result-positive">
                            <h1>⚠️ Chronic Kidney Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.error("""
                        **⚠️ Clinical Recommendations:**
                        - Immediate consultation with a nephrologist required
                        - Further diagnostic tests recommended (eGFR, kidney ultrasound)
                        - Monitor blood pressure and blood glucose regularly
                        - Implement kidney-friendly diet modifications
                        - Avoid nephrotoxic medications
                        """)
                    else:
                        st.markdown("""
                        <div class="result-negative">
                            <h1>✅ No Chronic Kidney Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.success("""
                        **💡 Health Maintenance Recommendations:**
                        - Continue regular health checkups
                        - Maintain healthy blood pressure (<130/80)
                        - Control blood sugar if diabetic
                        - Stay hydrated (8-10 glasses water/day)
                        - Maintain healthy weight through diet and exercise
                        """)
                
                with col2:
                    # Confidence gauge
                    fig = create_confidence_gauge(prediction_proba[prediction])
                    st.plotly_chart(fig, use_container_width=True)

# Liver Disease Page
elif page == "🫀 Liver Disease":
    st.markdown('<div class="main-header">🫀 Liver Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown("### Enter patient liver function test parameters")
    
    # Load models
    model = load_model('models/liver/liver_randomforest_basic.pkl')
    scaler = load_model('models/liver/liver_randomforest_scaler.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_liver_randomforest.py`")
    else:
        st.success("✅ Model loaded successfully! (Random Forest Classifier - 70% Accuracy)")
        
        # Sample data button
        use_sample = st.checkbox("🎲 Use sample patient data for quick test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📋 Patient Information</div>', unsafe_allow_html=True)
            age = st.number_input("Age (years)", 1, 120, 55 if use_sample else 45, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if use_sample else 0)
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.1, 100.0, 2.5 if use_sample else 1.0, 0.1, 
                                             help="Normal range: 0.3-1.2 mg/dL")
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 50.0, 0.8 if use_sample else 0.3, 0.1,
                                              help="Normal range: 0.1-0.3 mg/dL")
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", 10, 2000, 250 if use_sample else 200,
                                                  help="Normal range: 44-147 IU/L")
        
        with col2:
            st.markdown('<div class="section-header">🧪 Liver Function Tests</div>', unsafe_allow_html=True)
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase (IU/L)", 1, 2000, 60 if use_sample else 30,
                                                      help="Normal range: 7-56 IU/L")
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (IU/L)", 1, 2000, 70 if use_sample else 35,
                                                         help="Normal range: 10-40 IU/L")
            total_protiens = st.number_input("Total Proteins (g/dL)", 1.0, 15.0, 6.5 if use_sample else 7.0, 0.1,
                                            help="Normal range: 6.3-7.9 g/dL")
            albumin = st.number_input("Albumin (g/dL)", 0.5, 10.0, 3.5 if use_sample else 4.0, 0.1,
                                     help="Normal range: 3.5-5.5 g/dL")
            ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.1, 5.0, 0.9 if use_sample else 1.0, 0.1,
                                      help="Normal range: 1.0-2.5")
        
        st.markdown("---")
        
        if st.button("🔍 Predict Liver Disease", type="primary"):
            with st.spinner("Analyzing liver function parameters..."):
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
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 1:  # Liver Disease
                        st.markdown("""
                        <div class="result-positive">
                            <h1>⚠️ Liver Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.error("""
                        **⚠️ Clinical Recommendations:**
                        - Consult a hepatologist immediately
                        - Additional tests recommended (liver biopsy, imaging)
                        - Avoid alcohol and hepatotoxic medications
                        - Monitor liver enzyme levels regularly
                        - Consider dietary modifications (low-fat, high-protein)
                        """)
                    else:
                        st.markdown("""
                        <div class="result-negative">
                            <h1>✅ No Liver Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.success("""
                        **💡 Health Maintenance Recommendations:**
                        - Maintain healthy liver through balanced diet
                        - Limit alcohol consumption
                        - Exercise regularly (30 min/day)
                        - Maintain healthy weight
                        - Get regular health checkups
                        """)
                
                with col2:
                    fig = create_confidence_gauge(prediction_proba[prediction])
                    st.plotly_chart(fig, use_container_width=True)

# Parkinson's Disease Page
elif page == "🧠 Parkinson's Disease":
    st.markdown('<div class="main-header">🧠 Parkinson\'s Disease Detection</div>', unsafe_allow_html=True)
    st.markdown("### Voice measurement-based Parkinson's detection")
    
    # Load models
    model = load_model('models/parkinsons/parkinsons_xgboost_basic.pkl')
    scaler = load_model('models/parkinsons/parkinsons_xgboost_scaler.pkl')
    
    if model is None or scaler is None:
        st.error("⚠️ Model not found! Please train the model first by running `python training_scripts/train_parkinsons_xgboost.py`")
    else:
        st.success("✅ Model loaded successfully! (XGBoost Classifier - 92.31% Accuracy)")
        
        st.info("📝 **Note:** This model analyzes 22 voice measurement features to detect early signs of Parkinson's Disease.")
        
        # Sample data button
        use_sample = st.checkbox("🎲 Use sample voice data for quick test")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="section-header">🎵 Frequency Measures</div>', unsafe_allow_html=True)
            mdvp_fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 197.0 if use_sample else 150.0, 0.1, 
                                     help="Average vocal fundamental frequency")
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 206.0 if use_sample else 180.0, 0.1)
            mdvp_flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 192.0 if use_sample else 100.0, 0.1)
            mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.0068 if use_sample else 0.005, 0.0001)
            mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001, 0.00005 if use_sample else 0.00003, 0.000001)
            mdvp_rap = st.number_input("MDVP:RAP", 0.0, 0.1, 0.0037 if use_sample else 0.003, 0.0001)
            mdvp_ppq = st.number_input("MDVP:PPQ", 0.0, 0.1, 0.0035 if use_sample else 0.003, 0.0001)
            jitter_ddp = st.number_input("Jitter:DDP", 0.0, 0.1, 0.011 if use_sample else 0.008, 0.001)
        
        with col2:
            st.markdown('<div class="section-header">📊 Shimmer Measures</div>', unsafe_allow_html=True)
            mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.027 if use_sample else 0.03, 0.001)
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 2.0, 0.24 if use_sample else 0.3, 0.01)
            shimmer_apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.013 if use_sample else 0.015, 0.001)
            shimmer_apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.1, 0.015 if use_sample else 0.017, 0.001)
            mdvp_apq = st.number_input("MDVP:APQ", 0.0, 0.1, 0.020 if use_sample else 0.024, 0.001)
            shimmer_dda = st.number_input("Shimmer:DDA", 0.0, 0.1, 0.040 if use_sample else 0.045, 0.001)
        
        with col3:
            st.markdown('<div class="section-header">🔊 Vocal Measures</div>', unsafe_allow_html=True)
            nhr = st.number_input("NHR", 0.0, 1.0, 0.022 if use_sample else 0.025, 0.001,
                                 help="Noise-to-Harmonics Ratio")
            hnr = st.number_input("HNR", 5.0, 35.0, 21.5 if use_sample else 20.0, 0.1,
                                 help="Harmonics-to-Noise Ratio")
            rpde = st.number_input("RPDE", 0.2, 0.8, 0.56 if use_sample else 0.5, 0.01)
            dfa = st.number_input("DFA", 0.5, 0.9, 0.72 if use_sample else 0.7, 0.01)
            spread1 = st.number_input("spread1", -8.0, -2.0, -5.6 if use_sample else -5.0, 0.1)
            spread2 = st.number_input("spread2", 0.0, 0.5, 0.23 if use_sample else 0.2, 0.01)
            d2 = st.number_input("D2", 1.0, 4.0, 2.3 if use_sample else 2.5, 0.1)
            ppe = st.number_input("PPE", 0.0, 0.7, 0.26 if use_sample else 0.2, 0.01)
        
        st.markdown("---")
        
        if st.button("🔍 Detect Parkinson's Disease", type="primary"):
            with st.spinner("Analyzing voice measurements..."):
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
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 1:  # Parkinson's
                        st.markdown("""
                        <div class="result-positive">
                            <h1>⚠️ Parkinson's Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.error("""
                        **⚠️ Clinical Recommendations:**
                        - Consult a neurologist for comprehensive evaluation
                        - Consider DaTscan or other imaging tests
                        - Evaluate motor symptoms (tremor, rigidity, bradykinesia)
                        - Discuss treatment options (medications, therapy)
                        - Join support groups and education programs
                        """)
                    else:
                        st.markdown("""
                        <div class="result-negative">
                            <h1>✅ No Parkinson's Disease Detected</h1>
                            <h2>Confidence: {:.2f}%</h2>
                        </div>
                        """.format(prediction_proba[prediction]*100), unsafe_allow_html=True)
                        
                        st.success("""
                        **💡 Health Maintenance Recommendations:**
                        - Maintain active lifestyle with regular exercise
                        - Practice good sleep hygiene
                        - Stay mentally active with cognitive challenges
                        - Monitor for any changes in motor function
                        - Regular health checkups as recommended
                        """)
                
                with col2:
                    fig = create_confidence_gauge(prediction_proba[prediction])
                    st.plotly_chart(fig, use_container_width=True)

# About Models Page  
elif page == "📊 About Models":
    st.markdown('<div class="main-header">📊 Model Information & Performance</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kidney Model
    st.markdown('<div class="section-header">🫘 Chronic Kidney Disease Model</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Dataset Information</h4>
            <ul>
                <li><b>Total Samples:</b> 400 patients</li>
                <li><b>Features:</b> 24 clinical attributes</li>
                <li><b>Classes:</b> CKD vs Non-CKD</li>
                <li><b>Data Split:</b> 80% Training, 20% Testing</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>🎯 Model Performance</h4>
            <ul>
                <li><b>Algorithm:</b> Extra Trees Classifier</li>
                <li><b>Accuracy:</b> 100% on test set</li>
                <li><b>Precision:</b> 100%</li>
                <li><b>Recall:</b> 100%</li>
                <li><b>F1-Score:</b> 100%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">100%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);">
            <div class="metric-value">24</div>
            <div class="metric-label">Features</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #fb8c00 0%, #ffa726 100%);">
            <div class="metric-value">400</div>
            <div class="metric-label">Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📋 View All 24 Features"):
        st.write("""
        1. Age, 2. Blood Pressure, 3. Specific Gravity, 4. Albumin, 5. Sugar,
        6. Red Blood Cells, 7. Pus Cell, 8. Pus Cell Clumps, 9. Bacteria,
        10. Blood Glucose Random, 11. Blood Urea, 12. Serum Creatinine,
        13. Sodium, 14. Potassium, 15. Hemoglobin, 16. Packed Cell Volume,
        17. White Blood Cell Count, 18. Red Blood Cell Count, 19. Hypertension,
        20. Diabetes Mellitus, 21. Coronary Artery Disease, 22. Appetite,
        23. Pedal Edema, 24. Anemia
        """)
    
    st.markdown("---")
    
    # Liver Model
    st.markdown('<div class="section-header">🫀 Liver Disease Model</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Dataset Information</h4>
            <ul>
                <li><b>Total Samples:</b> 583 patients</li>
                <li><b>Liver Patients:</b> 416 (71.4%)</li>
                <li><b>Non-Liver:</b> 167 (28.6%)</li>
                <li><b>Features:</b> 10 liver function tests</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>🎯 Model Performance</h4>
            <ul>
                <li><b>Algorithm:</b> Random Forest Classifier</li>
                <li><b>Accuracy:</b> 70% on test set</li>
                <li><b>Precision:</b> 65%</li>
                <li><b>Recall:</b> 70%</li>
                <li><b>F1-Score:</b> 65%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">70%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);">
            <div class="metric-value">10</div>
            <div class="metric-label">Features</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #fb8c00 0%, #ffa726 100%);">
            <div class="metric-value">583</div>
            <div class="metric-label">Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parkinson's Model
    st.markdown('<div class="section-header">🧠 Parkinson\'s Disease Model</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Dataset Information</h4>
            <ul>
                <li><b>Total Samples:</b> 195 voice recordings</li>
                <li><b>Parkinson's:</b> 147 (75.4%)</li>
                <li><b>Healthy:</b> 48 (24.6%)</li>
                <li><b>Features:</b> 22 voice measurements</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>🎯 Model Performance</h4>
            <ul>
                <li><b>Algorithm:</b> XGBoost Classifier</li>
                <li><b>Accuracy:</b> 92.31% on test set</li>
                <li><b>Precision:</b> 92%</li>
                <li><b>Recall:</b> 92%</li>
                <li><b>F1-Score:</b> 92%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">92%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);">
            <div class="metric-value">22</div>
            <div class="metric-label">Features</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #fb8c00 0%, #ffa726 100%);">
            <div class="metric-value">195</div>
            <div class="metric-label">Recordings</div>
        </div>
        """, unsafe_allow_html=True)

# How to Use Page
elif page == "ℹ️ How to Use":
    st.markdown('<div class="main-header">ℹ️ How to Use HealthPredict</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🚀 Quick Start Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>Step 1: Select a Disease Module</h3>
        <p>Use the sidebar menu to choose from:</p>
        <ul>
            <li>🫘 Kidney Disease - For CKD prediction</li>
            <li>🫀 Liver Disease - For liver disease assessment</li>
            <li>🧠 Parkinson's Disease - For PD detection</li>
        </ul>
    </div>
    
    <div class="info-box">
        <h3>Step 2: Enter Patient Data</h3>
        <p>Fill in the clinical parameters:</p>
        <ul>
            <li>All fields are required for accurate prediction</li>
            <li>Use the 🎲 "Use sample data" checkbox for quick testing</li>
            <li>Hover over field labels for helpful tooltips</li>
        </ul>
    </div>
    
    <div class="info-box">
        <h3>Step 3: Get Prediction</h3>
        <p>Click the "Predict" button to:</p>
        <ul>
            <li>Receive instant AI-powered analysis</li>
            <li>View confidence scores and visualizations</li>
            <li>Get clinical recommendations</li>
        </ul>
    </div>
    
    <div class="info-box">
        <h3>Step 4: Interpret Results</h3>
        <p>Understanding the output:</p>
        <ul>
            <li><b>Confidence Score:</b> Model's certainty (0-100%)</li>
            <li><b>Clinical Recommendations:</b> Next steps based on results</li>
            <li><b>Important:</b> Always consult healthcare professionals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">💡 Tips for Best Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>✅ Do's</h4>
            <ul>
                <li>Use accurate clinical measurements</li>
                <li>Double-check all input values</li>
                <li>Test with sample data first</li>
                <li>Consult medical professionals</li>
                <li>Use for preliminary screening only</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>❌ Don'ts</h4>
            <ul>
                <li>Don't use for final diagnosis</li>
                <li>Don't skip medical consultation</li>
                <li>Don't use estimated values</li>
                <li>Don't ignore low confidence scores</li>
                <li>Don't self-medicate based on results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px;'>
    <p style='font-size: 0.8rem; margin: 5px 0;'><b>HealthPredict v1.0</b></p>
    <p style='font-size: 0.7rem; margin: 5px 0;'>Educational & Research Use Only</p>
    <p style='font-size: 0.7rem; margin: 5px 0;'>© 2025 HealthPredict</p>
</div>
""", unsafe_allow_html=True)
