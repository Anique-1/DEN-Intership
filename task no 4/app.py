import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

st.title("Parkinson's Disease Prediction")

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        rf_all = joblib.load("model_rf_all_features.joblib")
    except Exception:
        rf_all = None
    try:
        rf_sel = joblib.load("best_model_rf_selected_features.joblib")
    except Exception:
        rf_sel = None
    try:
        scaler = joblib.load("scaler.joblib")
    except Exception:
        scaler = None
    try:
        with open("selected_features.pkl", "rb") as f:
            selected_features = pickle.load(f)
    except Exception:
        selected_features = None
    return rf_all, rf_sel, scaler, selected_features

rf_all, rf_sel, scaler, selected_features = load_models()

if not rf_all or not rf_sel or not scaler or not selected_features:
    st.error("One or more model files are missing. Please ensure the following files are present in the directory: "
             "'model_rf_all_features.joblib', 'best_model_rf_selected_features.joblib', 'scaler.joblib', 'selected_features.pkl'.")
    st.stop()

# Feature names for all features (from training)
all_features = scaler.feature_names_in_

# Model selection
model_option = st.radio(
    "Choose Model:",
    ("Without Optimization (All Features)", "With Optimization (Selected Features)")
)

if model_option == "Without Optimization (All Features)":
    model = rf_all
    feature_list = all_features
    st.info("Using Random Forest trained on all features.")
else:
    model = rf_sel
    feature_list = selected_features
    st.info("Using Random Forest trained on selected features.")

# User input for features
st.header("Input Feature Values")
user_input = {}
for feat in feature_list:
    user_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")

# Prepare input for prediction
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df[all_features])

if model_option == "With Optimization (Selected Features)":
    input_scaled = pd.DataFrame(input_scaled, columns=all_features)[selected_features].values

# Prediction
if st.button("Predict"):
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0]
    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{int(pred[0])}**")
    st.write("Probability Scores:")
    st.write({f"Class {i}": float(prob) for i, prob in enumerate(proba)})
