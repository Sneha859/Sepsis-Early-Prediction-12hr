import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load pipeline (Scaler + RF Model)
# -------------------------------
pipeline = joblib.load("models/rf_pipeline_with_scaler.pkl")

st.set_page_config(page_title="Sepsis 12hr Early Prediction", layout="centered")
st.title("🩺 Early Sepsis Prediction (12 Hours Ahead)")
st.write("Upload a patient data file to predict whether the patient is at risk of developing sepsis within the next 12 hours.")

# -----------------------------------------------------
# 📂 Upload Patient File
# -----------------------------------------------------
st.subheader("📁 Upload Patient Data (CSV or PSV)")
uploaded_file = st.file_uploader("Choose a patient file", type=["csv", "psv"])

if uploaded_file is not None:

    # Read data
    if uploaded_file.name.endswith(".psv"):
        df = pd.read_csv(uploaded_file, sep='|')
    else:
        df = pd.read_csv(uploaded_file)

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # Clean data (same as training)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Expected features from the pipeline
    expected_features = pipeline.feature_names_in_

    # Keep only features needed by model
    df_model = df[[c for c in expected_features if c in df.columns]].copy()

    # Convert all columns to numeric
    df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ---------------------------
    # Predict probabilities
    # ---------------------------
    probs = pipeline.predict_proba(df_model)[:, 1]
    df["Sepsis_Probability"] = probs

    # Overall risk
    max_prob = float(np.max(probs))

    if max_prob < 0.20:
        risk_level = "🟩 No Risk of Sepsis"
    elif max_prob < 0.40:
        risk_level = "🟦 Low Risk of Sepsis"
    elif max_prob < 0.75:
        risk_level = "🟧 Medium Risk of Sepsis"
    else:
        risk_level = "🟥 High Risk of Sepsis"

    st.markdown("---")
    st.subheader("🧍 Overall Sepsis Prediction Summary")
    st.markdown(f"**Final Prediction:** {risk_level}")
    st.markdown(f"**Maximum Hourly Probability:** `{max_prob:.3f}`")

    st.info(
        "ℹ️ The model evaluates each hour independently. "
        "The highest hourly probability determines the risk level."
    )

    # -----------------------------------------------------
    # 📈 Trend Chart
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Sepsis Risk Trend Over Time")

    plt.figure(figsize=(7, 4))
    plt.plot(df["Sepsis_Probability"], marker="o", linewidth=2)
    plt.axhline(0.20, color="green", linestyle="--")
    plt.axhline(0.40, color="blue", linestyle="--")
    plt.axhline(0.75, color="orange", linestyle="--")
    plt.title("Hourly Sepsis Probability Trend")
    plt.xlabel("Hour Index")
    plt.ylabel("Probability")
    plt.grid(True)
    st.pyplot(plt)

st.write("---")
st.caption("Model: Random Forest Pipeline | Prediction Window: 12 hours | Developed by Sneha")
