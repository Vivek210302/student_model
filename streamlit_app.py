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
        <h1 style='font-size:45px;'>🎒✨ Extracurricular Activity Predictor ✨🎒</h1>
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
    
    # Calculate base values
    # Note: sleep_hours and sample_papers seem unused by the final model features based on the list
    
    # Construct the data dictionary with list values to preserve order when creating DataFrame
    # but since we need duplicate columns, we'll create a DataFrame directly from a list of lists/dicts 
    # and then manually set columns, or just use a numpy array if the model accepts it (but likely expects DF with names).
    # Best approach for duplicate columns in pandas: create with unique names then rename, 
    # OR create dictionary with list values and use `pd.DataFrame`, but dicts can't have duplicate keys.
    
    # Strategy: Create a list of values in the correct order, then creating a DF with those columns
    
    data_values = [
        hours_studied * previous_scores,             # Hours Studied Previous Scores
        previous_scores,                             # Previous Scores
        previous_scores,                             # Previous Scores (Duplicate)
        hours_studied ** 2,                          # Hours Studied^2
        previous_scores * performance_index,         # Previous Scores Performance Index
        performance_index ** 2,                      # Performance Index^2
        performance_index * 1,                       # Extracurricular_Performance_Interaction (Assuming binary 1 for interaction base?) 
                                                     # Wait, "Interaction" usually implies feature * feature. 
                                                     # If it was categorical 'Extracurricular' * Performance, we don't have 'Extracurricular' input (that's target).
                                                     # Looking at previous code: 'Extracurricular_Performance_Interaction': performance_index * 1
                                                     # It seems it was just a copy of performance index.
        previous_scores * 1,                         # Extracurricular_PreviousScores_Interaction
        performance_index,                           # Performance Index
        performance_index                            # Performance Index (Duplicate)
    ]
    
    column_names = [
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
    
    # Create DataFrame
    # We must pass data as a list of lists (rows) to allow duplicate columns if we set columns argument
    df = pd.DataFrame([data_values], columns=column_names)
    
    return df

input_df = generate_features()

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

st.dataframe(input_df, use_container_width=True)

# -------------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎯 Predict Now", use_container_width=True):
    try:
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.markdown(
                """
                <div class='main-card' style='background:#e0ffe6;'>
                    <h2 style='text-align:center;'>🎉 The student is likely to participate in <b>extracurricular activities</b>!</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='main-card' style='background:#ffe6e6;'>
                    <h2 style='text-align:center;'>🚫 The student is <b>not likely</b> to participate in extracurricular activities.</h2>
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
