import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Extracurricular Activity Predictor", page_icon="🔥", layout="wide")

# Gradient Background + Custom CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #d9e4f5 0%, #f5e6e8 100%) !important;
        background-attachment: fixed;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Animated Title Section
st.markdown(
    """
    <div style='text-align:center; padding:25px' class='main-card'>
        <h1 style='font-size:45px;'>🎒✨ Extracurricular Activity Predictor (v2.0) ✨🎒</h1>
        <p style='font-size:18px;'>A beautiful, modern, AI-powered app that predicts a student's extracurricular activity involvement.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
MODEL_PATH = "model/logistic_regression_model.joblib"
try:
    model = load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR WITH STICKERS + SLIDERS
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align:center;'>
    <h2>🎛️ Student Inputs</h2>
    <p>Adjust the values below</p>
</div>
""", unsafe_allow_html=True)

hours_studied = st.sidebar.slider("📘 Hours Studied", 0.0, 24.0, 5.0, 0.5)
previous_scores = st.sidebar.slider("📊 Previous Scores", 0.0, 100.0, 50.0, 1.0)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0.0, 24.0, 7.0, 0.5)
sample_papers = st.sidebar.slider("📄 Sample Papers Practiced", 0, 50, 5)
performance_index = st.sidebar.slider("📈 Performance Index", 0.0, 10.0, 5.0, 0.1)

# -------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------
def generate_features():
    # The model expects specific features in a specific order, including duplicates.
    # Expected features: 
    # ['Hours Studied Previous Scores', 'Previous Scores', 'Previous Scores', 
    #  'Hours Studied^2', 'Previous Scores Performance Index', 
    #  'Performance Index^2', 'Extracurricular_Performance_Interaction', 
    #  'Extracurricular_PreviousScores_Interaction', 'Performance Index', 
    #  'Performance Index']
    
    data_values = [
        hours_studied * previous_scores,             # Hours Studied Previous Scores
        previous_scores,                             # Previous Scores
        previous_scores,                             # Previous Scores (Duplicate)
        hours_studied ** 2,                          # Hours Studied^2
        previous_scores * performance_index,         # Previous Scores Performance Index
        performance_index ** 2,                      # Performance Index^2
        performance_index * 1,                       # Extracurricular_Performance_Interaction
        previous_scores * 1,                         # Extracurricular_PreviousScores_Interaction
        performance_index,                           # Performance Index
        performance_index                            # Performance Index (Duplicate)
    ]
    
    # Names required by the model (with duplicates)
    model_column_names = [
        'Hours Studied Previous Scores',
        'Previous Scores',
        'Previous Scores',
        'Hours Studied^2',
        'Previous Scores Performance Index',
        'Performance Index^2',
        'Extracurricular_Performance_Interaction',
        'Extracurricular_PreviousScores_Interaction',
        'Performance Index',
        'Performance Index'
    ]
    
    # Names for display (unique)
    display_column_names = [
        'Hours Studied Previous Scores',
        'Previous Scores (1)',
        'Previous Scores (2)',
        'Hours Studied^2',
        'Previous Scores Performance Index',
        'Performance Index^2',
        'Extracurricular_Performance_Interaction',
        'Extracurricular_PreviousScores_Interaction',
        'Performance Index (1)',
        'Performance Index (2)'
    ]
    
    # Create DF for Model
    # We create with unique names first to satisfy any strict constructors, then rename
    df_model = pd.DataFrame([data_values])
    df_model.columns = model_column_names
    
    # Create DF for Display
    df_display = pd.DataFrame([data_values], columns=display_column_names)
    
    return df_model, df_display

input_df_model, input_df_display = generate_features()

# Interactive preview card
st.markdown(
    """
    <div class='main-card'>
        <h2>🔍 Auto-Generated Features</h2>
        <p>Your engineered features appear below:</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(input_df_display, use_container_width=True)

# -------------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎯 Predict Now", use_container_width=True):
    try:
        pred = model.predict(input_df_model)[0]

        if pred == 1:
            st.markdown(
                """
                <div class='main-card' style='background:#e0ffe6; color: #000000;'>
                    <h2 style='text-align:center; color: #000000;'>🎉 The student is likely to participate in <b>extracurricular activities</b>!</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='main-card' style='background:#ffe6e6; color: #000000;'>
                    <h2 style='text-align:center; color: #000000;'>🚫 The student is <b>not likely</b> to participate in extracurricular activities.</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Ensure the inputs are valid.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>Made with ❤️ — Buddy</p>
    """,
    unsafe_allow_html=True,
)
