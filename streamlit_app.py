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
model = load(MODEL_PATH)

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
    base = {
        'Hours Studied': hours_studied,
        'Previous Scores': previous_scores,
        'Sleep Hours': sleep_hours,
        'Sample Question Papers Practiced': sample_papers,
        'Performance Index': performance_index,
    }

    engineered = {
        'Study_Sleep_Ratio': hours_studied / sleep_hours if sleep_hours > 0 else 0,
        'Performance_Study_Ratio': performance_index / hours_studied if hours_studied > 0 else 0,
        'Extracurricular_Performance_Interaction': performance_index * 1,
        'Extracurricular_PreviousScores_Interaction': previous_scores * 1,

        'Hours Studied': hours_studied,
        'Previous Scores': previous_scores,
        'Performance Index': performance_index,

        'Hours Studied^2': hours_studied ** 2,
        'Hours Studied Previous Scores': hours_studied * previous_scores,
        'Hours Studied Performance Index': hours_studied * performance_index,

        'Previous Scores^2': previous_scores ** 2,
        'Previous Scores Performance Index': previous_scores * performance_index,
        'Performance Index^2': performance_index ** 2,
    }

    return pd.DataFrame([{**base, **engineered}])

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

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>Made with ❤️ — Buddy</p>
    """,
    unsafe_allow_html=True,
)