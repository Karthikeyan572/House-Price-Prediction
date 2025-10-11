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

# Expose one input per column: sliders for numeric columns, selectboxes for categorical columns.
# If no dataset is present, fall back to a small example feature set.
if not df.empty:
    # exclude the target column `price` from inputs so the app predicts it
    chosen_numeric = [c for c in numeric_cols if c.lower() != 'price']
    chosen_cat = object_cols.copy()
else:
    # sensible defaults when no data is available
    chosen_numeric = ["area", "bedrooms", "bathrooms"]
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
        # determine min/max/value
        if col.lower() == 'area':
            col_min, col_max, col_mean = 1000.0, 15000.0, 5150.54
        elif col.lower() in ('bedrooms', 'bathrooms', 'stories', 'parking'):
            # integer inputs 0..10
            col_min, col_max = 0, 10
            if not df.empty and col in df.columns:
                try:
                    col_mean = int(df[col].median())
                except Exception:
                    col_mean = 3
            else:
                col_mean = 3
        elif not df.empty and col in df.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
        else:
            col_min, col_max, col_mean = 0.0, 100.0, 50.0

        # show the allowed range next to the input
        st.write(f"{col}")
        st.caption(f"range: {col_min} — {col_max}")

        # integer inputs for small discrete features
        if col.lower() in ('bedrooms', 'bathrooms', 'stories', 'parking'):
            val = st.number_input(label=f"", min_value=int(col_min), max_value=int(col_max), value=int(col_mean), step=1, format="%d", key=f"num_{col}")
            inputs[col] = int(val)
        else:
            # float inputs (area or other continuous)
            val = st.number_input(label=f"", min_value=float(col_min), max_value=float(col_max), value=float(col_mean), step=(float(col_max)-float(col_min))/100.0 if float(col_max)>float(col_min) else 0.01, format="%.3f", key=f"num_{col}")
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

# Simple model: train a quick LinearRegression on chosen numeric features to predict `price` if available
model = None
trained = False
if not df.empty and "price" in df.columns and len(chosen_numeric) > 0:
    try:
        X = df[chosen_numeric].copy()
        y = df["price"].copy()
        model = LinearRegression()
        model.fit(X, y)
        trained = True
    except Exception as e:
        st.warning(f"Could not train model from sample data: {e}")

# Predict button (manual)
def compute_prediction(inputs_dict):
    if model is None:
        numeric_vals = np.array([inputs_dict.get(c, 0.0) for c in chosen_numeric]) if len(chosen_numeric) > 0 else np.array([0.0])
        return float(numeric_vals.sum())
    else:
        X_new = np.array([inputs_dict.get(c, 0.0) for c in chosen_numeric]).reshape(1, -1)
        return float(model.predict(X_new)[0])

if st.button("Predict"):
    pred_val = compute_prediction(inputs)
    st.success(f"Predicted price: {pred_val:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with Streamlit")
