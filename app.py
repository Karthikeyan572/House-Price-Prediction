import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import json
import os

# Simple Streamlit app for house price prediction

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Predictor")

st.markdown("This app predicts house prices using a simple model. You can upload your own CSV or use the sample `Housing.csv` if present in the workspace.")

@st.cache_data
def load_data(path: str = "Housing.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    return df

# Try to load sample data
sample_df = load_data()

st.sidebar.header("Input data")
use_sample = False
if not sample_df.empty:
    use_sample = st.sidebar.checkbox("Use sample data from Housing.csv", value=True)

uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type=["csv"]) 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_sample and not sample_df.empty:
    df = sample_df.copy()
else:
    df = pd.DataFrame()

# If we have a dataframe, try to infer columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() if not df.empty else []
object_cols = df.select_dtypes(include=["object"]).columns.tolist() if not df.empty else []

st.sidebar.markdown("---")

st.sidebar.header("Model inputs")

# Expose one input per column: sliders for numeric columns, selectboxes for categorical columns.
# If no dataset is present, fall back to a small example feature set.
if not df.empty:
    chosen_numeric = numeric_cols.copy()
    chosen_cat = object_cols.copy()
else:
    # sensible defaults when no data is available
    chosen_numeric = ["RM", "LSTAT", "PTRATIO"]
    chosen_cat = []

# Collect inputs (one value per column)
inputs = {}
st.header("Inputs — provide one value for each column")

num_col, cat_col = st.columns(2)

with num_col:
    st.subheader("Continuous (sliders)")
    if len(chosen_numeric) == 0:
        st.write("No numeric columns detected.")
    for col in chosen_numeric:
        if not df.empty and col in df.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
        else:
            col_min, col_max, col_mean = 0.0, 100.0, 50.0
        # avoid zero step
        step = (col_max - col_min) / 100.0 if (col_max - col_min) > 0 else 0.01
        # show each numeric input as a single slider
        val = st.slider(label=f"{col}", min_value=col_min, max_value=col_max, value=col_mean, step=step, format="%.3f")
        inputs[col] = float(val)

with cat_col:
    st.subheader("Categorical (options)")
    if len(chosen_cat) == 0:
        st.write("No categorical columns detected or available.")
    for col in chosen_cat:
        if not df.empty and col in df.columns:
            options = df[col].dropna().unique().tolist()
            if len(options) == 0:
                # fallback to text input when no distinct options
                sel = st.text_input(label=f"{col} (type value)")
            else:
                sel = st.selectbox(label=f"{col}", options=options)
        else:
            sel = st.text_input(label=f"{col} (type value)")
        inputs[col] = sel

st.markdown("---")

# Simple model: train a quick LinearRegression on chosen numeric features to predict target if available
model = None
trained = False
if not df.empty and "MEDV" in df.columns and len(chosen_numeric) > 0:
    try:
        X = df[chosen_numeric].copy()
        y = df["MEDV"].copy()
        model = LinearRegression()
        model.fit(X, y)
        trained = True
    except Exception as e:
        st.warning(f"Could not train model from sample data: {e}")

# Interactive controls: presets and auto-predict
st.sidebar.header("Controls")
preset_dir = "presets"
os.makedirs(preset_dir, exist_ok=True)

preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.json')]
with st.sidebar.expander("Presets (save / load)"):
    new_preset_name = st.text_input("Preset name (to save)")
    if st.button("Save preset") and new_preset_name:
        path = os.path.join(preset_dir, f"{new_preset_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(inputs, fh, ensure_ascii=False, indent=2)
        st.success(f"Saved preset: {path}")
        preset_files = [f for f in os.listdir(preset_dir) if f.endswith('.json')]
    if len(preset_files) > 0:
        chosen_preset = st.selectbox("Load preset", options=["-- none --"] + preset_files)
        if chosen_preset and chosen_preset != "-- none --":
            with open(os.path.join(preset_dir, chosen_preset), "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            # apply the loaded values to the session (Streamlit cannot programmatically change widget values,
            # so show them and update the inputs dict to use for prediction)
            st.write("Loaded preset values shown below; to apply them, adjust sliders/selectboxes manually to match.")
            st.json(loaded)
    else:
        st.write("No saved presets yet")

auto_predict = st.sidebar.checkbox("Auto predict (live)", value=False)

st.subheader("Input summary")
st.json(inputs)

def compute_prediction(inputs_dict):
    # numeric first
    if model is None:
        numeric_vals = np.array([inputs_dict.get(c, 0.0) for c in chosen_numeric]) if len(chosen_numeric) > 0 else np.array([0.0])
        return float(numeric_vals.sum())
    else:
        X_new = np.array([inputs_dict.get(c, 0.0) for c in chosen_numeric]).reshape(1, -1)
        return float(model.predict(X_new)[0])

if auto_predict:
    pred_val = compute_prediction(inputs)
    st.success(f"Predicted (live): {pred_val:.3f}")

# Predict button (manual)
if st.button("Predict"):
    pred_val = compute_prediction(inputs)
    st.success(f"Predicted: {pred_val:.3f}")

# Feature contribution (bar chart) — if model is available, show coefficients; otherwise show heuristic weights
st.subheader("Feature contributions")
if len(chosen_numeric) > 0:
    if model is not None:
        coefs = model.coef_
        contrib = {col: float(coef * inputs.get(col, 0.0)) for col, coef in zip(chosen_numeric, coefs)}
        contrib_series = pd.Series(contrib)
    else:
        # simple heuristic: contribution = value normalized
        contrib_series = pd.Series({col: float(inputs.get(col, 0.0)) for col in chosen_numeric})
    st.bar_chart(contrib_series)
else:
    st.write("No numeric features to show.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with Streamlit")
