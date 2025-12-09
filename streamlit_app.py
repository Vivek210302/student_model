import streamlit as st
from joblib import load
import pandas as pd

# Load model
model = load("model/logistic_regression_model.joblib")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# Title Section
st.markdown(
    """
    <div style="text-align:center; padding:20px">
        <h1>🎓 Student Performance Predictor</h1>
        <p style="font-size:18px">Predict whether a student passes or fails — now with fun stickers & a modern UI ✨</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("## 🛠️ Input Controls")

# Define input fields with stickers
age = st.sidebar.number_input("👶 Age", 10, 25, 16)
study_time = st.sidebar.slider("📚 Study Time (hours/week)", 1, 20, 5)
freetime = st.sidebar.slider("🎮 Free Time Level", 1, 5, 3)
health = st.sidebar.slider("💪 Health Level", 1, 5, 4)
G1 = st.sidebar.number_input("📝 First Period Grade (0-20)", 0, 20, 10)
G2 = st.sidebar.number_input("📝 Second Period Grade (0-20)", 0, 20, 10)

# Convert to DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'studytime': [study_time],
    'freetime': [freetime],
    'health': [health],
    'G1': [G1],
    'G2': [G2]
})

# Predict 
if st.button("🔮 Predict Performance", use_container_width=True):
    prediction = model.predict(input_data)[0]
    result = "✅ PASS" if prediction == 1 else "❌ FAIL"

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:15px; background:#f0f9ff; text-align:center; margin-top:20px">
            <h2 style="font-size:40px">{result}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px">Made with ❤️ by Buddy</p>
    """,
    unsafe_allow_html=True
)