import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

# --- Config ---
st.set_page_config(page_title="Extracurricular Activity Predictor", page_icon="🎯", layout="wide")

st.markdown("""
<div style="text-align:center; padding:18px">
  <h1>🎒 Extracurricular Activity Predictor</h1>
  <p style="font-size:16px">Enter student details below and the model will predict whether they participate in extracurricular activities. Friendly UI with stickers! ✨</p>
</div>
""", unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = "model/logistic_regression_model.joblib"
try:
    model = load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model at '{MODEL_PATH}': {e}")
    st.stop()

# --- Load dataset to infer fields (uses your uploaded CSV) ---
DATA_PATH = "data/StudentPerformance.csv"
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset at '{DATA_PATH}': {e}")
    st.stop()

TARGET = "extracurricularactivity"
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in dataset. Please check your CSV.")
    st.stop()

feature_cols = [c for c in df.columns if c != TARGET]

st.sidebar.markdown("## 🛠️ Input Controls")
st.sidebar.write("(Values default to dataset medians / most common)")

# Dynamically create input widgets based on column types
user_inputs = {}
for col in feature_cols:
    series = df[col].dropna()
    # treat numeric columns
    if pd.api.types.is_numeric_dtype(series):
        # choose sensible bounds
        col_min = float(series.min())
        col_max = float(series.max())
        col_mean = float(series.median())
        # choose int vs float
        if np.all(series.dropna().apply(float.is_integer)):
            user_inputs[col] = st.sidebar.number_input(f"🔢 {col}", value=int(col_mean), min_value=int(col_min), max_value=int(col_max), step=1)
        else:
            user_inputs[col] = st.sidebar.number_input(f"🔢 {col}", value=float(col_mean), min_value=float(col_min), max_value=float(col_max), step=0.1)
    else:
        # categorical — use the most common values as options
        unique_vals = list(series.astype(str).unique())
        if len(unique_vals) > 50:
            # too many categories — allow free text but show top 10
            top_vals = series.astype(str).value_counts().head(10).index.tolist()
            user_inputs[col] = st.sidebar.selectbox(f"🏷️ {col} (top shown)", options=top_vals)
        else:
            user_inputs[col] = st.sidebar.selectbox(f"🏷️ {col}", options=unique_vals)

# Convert to DataFrame for model
input_df = pd.DataFrame([user_inputs])

st.subheader("Input Preview")
st.dataframe(input_df)

# Prediction button
if st.button("🔮 Predict Extracurricular Activity", use_container_width=True):
    try:
        # Ensure column ordering matches training features
        # If model expects a specific set of columns (e.g., pipeline), this will work if column names match.
        X = input_df.copy()
        # If model is a pipeline that expects raw columns, feed X directly
        # If model expects numpy array, it will accept DataFrame as well in sklearn-compatible models
        pred = model.predict(X)
        pred_label = pred[0]
        # probability if available
        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                prob = model.predict_proba(X)[0]
            except Exception:
                # model may be pipeline; try model steps
                prob = None

        # Display result
        if prob is not None:
            # If binary, take class 1 probability if classes are [0,1]
            if len(prob) == 2:
                prob_val = prob[1]
                st.markdown(f"<div style='padding:18px; border-radius:12px; background:#f0f9ff; text-align:center'>\n  <h2>Result: {'✅ Participates' if pred_label==1 else '❌ Does NOT Participate'}</h2>\n  <p>Probability: <b>{prob_val:.2f}</b></p>\n</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:18px; border-radius:12px; background:#f0fff0; text-align:center'>\n  <h2>Result: {pred_label}</h2>\n  <p>Probabilities: {prob}</p>\n</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:18px; border-radius:12px; background:#fff7f0; text-align:center'>\n  <h2>Result: {'✅ Participates' if pred_label==1 else '❌ Does NOT Participate'}</h2>\n</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("""
<hr>
<p style="text-align:center; font-size:14px">Made with ❤️ — Buddy</p>
""", unsafe_allow_html=True)