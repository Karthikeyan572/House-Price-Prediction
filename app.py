import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

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

# If dataset is available, let user pick features. Otherwise provide example inputs
if not df.empty and len(numeric_cols) > 0:
    st.sidebar.write("Detected numeric columns:")
    chosen_numeric = st.sidebar.multiselect("Choose numeric features to expose as sliders (max 6)", numeric_cols, default=numeric_cols[:6])
else:
    chosen_numeric = ["RM", "LSTAT", "PTRATIO"]

# For continuous sliders use float ranges
inputs = {}
st.header("Inputs")
for col in chosen_numeric:
    if not df.empty:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
    else:
        # reasonable defaults
        col_min, col_max, col_mean = 0.0, 100.0, 50.0
    val = st.slider(label=f"{col}", min_value=col_min, max_value=col_max, value=col_mean, step=(col_max-col_min)/100 if col_max>col_min else 1.0, format="%.3f")
    inputs[col] = val

# Categorical inputs
if not df.empty and len(object_cols) > 0:
    st.sidebar.write("Detected categorical columns:")
    chosen_cat = st.sidebar.multiselect("Choose categorical features to expose as select boxes (max 4)", object_cols, default=object_cols[:4])
else:
    chosen_cat = []

for col in chosen_cat:
    options = df[col].dropna().unique().tolist() if not df.empty else ["A", "B"]
    sel = st.selectbox(label=f"{col}", options=options)
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

# Predict button
if st.button("Predict"):
    if model is None:
        st.info("No trained model available. Training a simple default model with random weights.")
        # fallback: create dummy prediction by a linear combination of numeric inputs
        numeric_vals = np.array([inputs[c] for c in chosen_numeric]) if len(chosen_numeric)>0 else np.array([0])
        pred = float(numeric_vals.sum())
        st.success(f"Predicted value (fallback): {pred:.3f}")
    else:
        # build input vector
        X_new = np.array([inputs[c] for c in chosen_numeric]).reshape(1, -1)
        pred = model.predict(X_new)[0]
        st.success(f"Predicted MEDV: {pred:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with Streamlit")
