import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# ===========================
# Feature List
# ===========================
features = [
    "sigsd", "snr",
    "minute", "day", "month", "dayofweek",
    "sig_lag1", "sig_lag2", "snr_lag1", "snr_lag2",
    "sig_roll3", "sig_roll5",
    "sig_diff", "weak_signal", "strong_signal"
]

# ===========================
# Load Files
# ===========================
import urllib.request

def download_from_drive(url, filename):
    urllib.request.urlretrieve(url, filename)
    return filename

# Google Drive direct download URLs
model_url = "https://drive.google.com/uc?export=download&id=16CqUw9WKteMdfaK3GYrx0WXyM47Yu-X4"
df_url = "https://drive.google.com/uc?export=download&id=1gLCF6EnF_siGFpaIWLc4pT4ou0TjpW_K"
ytest_url = "https://drive.google.com/uc?export=download&id=1vFABI2SZ0CJNGSuKPAsdXe43oglnN1XH"
ypred_url = "https://drive.google.com/uc?export=download&id=1GJudVsC5oe6UqDRRYEB_u8LKXlMJVvj0"

# Download files into Streamlit working directory
model_file = download_from_drive(model_url, "rfmodel.pkl")
df_file = download_from_drive(df_url, "df_fe_sample.csv")
ytest_file = download_from_drive(ytest_url, "y_test.csv")
ypred_file = download_from_drive(ypred_url, "y_pred.csv")

# Load normally
model = joblib.load(model_file)
df_full = pd.read_csv(df_file)
df_sample = df_full[features].sample(100, random_state=42)

y_test = pd.read_csv(ytest_file)["sig"]
y_pred = pd.read_csv(ypred_file)["sig_pred"]


# ===========================
# Streamlit App UI
# ===========================
st.title("🦉 Owl Signal Strength Prediction App (BBO – MOTUS Project)")
st.write("Predict Saw-whet owl signal strength and explain results using XAI.")

st.header("🔧 Enter Detection Features")

# ===========================
# User Inputs
# ===========================
# ===========================
# Grouped & Clean User Inputs
# ===========================

st.header("🔧 Enter Detection Features")

user_input = {}

# ---------------------------
# A. Basic Signal Features
# ---------------------------
with st.expander("📘 A. Basic Signal Features", expanded=True):
    user_input["sigsd"] = st.number_input(
        "Signal Standard Deviation (sigsd)", 
        value=float(df_sample["sigsd"].mean())
    )
    user_input["snr"] = st.number_input(
        "Signal-to-Noise Ratio (snr)", 
        value=float(df_sample["snr"].mean())
    )

# ---------------------------
# B. Time-Based Features
# ---------------------------
with st.expander("🕒 B. Time-Based Features"):
    user_input["minute"] = st.slider(
        "Minute of Hour", 0, 59, int(df_sample["minute"].mean())
    )
    user_input["day"] = st.slider(
        "Day of Month", 1, 31, int(df_sample["day"].mean())
    )
    user_input["month"] = st.slider(
        "Month", 1, 12, int(df_sample["month"].mean())
    )
    user_input["dayofweek"] = st.selectbox(
        "Day of Week (0=Mon, 6=Sun)",
        options=[0,1,2,3,4,5,6],
        index=int(df_sample["dayofweek"].mean())
    )

# ---------------------------
# C. Lagged Features
# ---------------------------
with st.expander("📊 C. Previous Detection Features (Lagged Signals)"):
    user_input["sig_lag1"] = st.number_input(
        "Previous Signal (sig_lag1)",
        value=float(df_sample["sig_lag1"].mean())
    )
    user_input["sig_lag2"] = st.number_input(
        "Previous Signal 2 Steps Back (sig_lag2)",
        value=float(df_sample["sig_lag2"].mean())
    )
    user_input["snr_lag1"] = st.number_input(
        "Previous SNR (snr_lag1)",
        value=float(df_sample["snr_lag1"].mean())
    )
    user_input["snr_lag2"] = st.number_input(
        "Previous SNR 2 Steps Back (snr_lag2)",
        value=float(df_sample["snr_lag2"].mean())
    )

# ---------------------------
# D. Rolling Window Features
# ---------------------------
with st.expander("📈 D. Rolling Averages (Smoothed Trends)"):
    user_input["sig_roll3"] = st.number_input(
        "3-Point Rolling Signal Avg (sig_roll3)",
        value=float(df_sample["sig_roll3"].mean())
    )
    user_input["sig_roll5"] = st.number_input(
        "5-Point Rolling Signal Avg (sig_roll5)",
        value=float(df_sample["sig_roll5"].mean())
    )

# ---------------------------
# E. Behaviour Indicators
# ---------------------------
with st.expander("⚠️ E. Behaviour Indicators"):
    user_input["sig_diff"] = st.number_input(
        "Signal Change from Last Detection (sig_diff)",
        value=float(df_sample["sig_diff"].mean())
    )
    user_input["weak_signal"] = st.number_input(
        "Weak Signal Indicator (weak_signal)",
        value=float(df_sample["weak_signal"].mean())
    )
    user_input["strong_signal"] = st.number_input(
        "Strong Signal Indicator (strong_signal)",
        value=float(df_sample["strong_signal"].mean())
    )

# Convert to DataFrame for prediction
input_df = pd.DataFrame([user_input])


# ===========================
# Prediction
# ===========================
pred = model.predict(input_df)[0]
st.subheader(f"📡 Predicted Signal Strength: **{pred:.2f}**")

st.markdown("---")
st.header("📊 Explainable AI (XAI) Insights")

# ===========================
# 1. Feature Importance
# ===========================
st.subheader("1️⃣ Feature Importance (Global)")
importances = model.feature_importances_

fig1, ax1 = plt.subplots()
ax1.barh(features, importances)
ax1.set_title("Feature Importance")
ax1.set_xlabel("Importance")
st.pyplot(fig1)

st.markdown("---")

# ===========================
# 2. SHAP Analysis (On Demand)
# ===========================
st.subheader("2️⃣ SHAP Explainability (Click to Generate)")

if st.button("Generate SHAP Plots"):
    shap_values = explainer.shap_values(df_sample)

    # SHAP Summary Plot
    st.write("**SHAP Summary Plot (Global Impact)**")
    fig2 = plt.figure()
    shap.summary_plot(shap_values, df_sample, show=False)
    st.pyplot(fig2)

    # Force plot for user input
    st.write("**SHAP Force Plot (Local Explanation for Current Input)**")
    single_shap = explainer.shap_values(input_df)

    fig3 = shap.force_plot(
        explainer.expected_value,
        single_shap,
        input_df,
        matplotlib=True
    )
    st.pyplot(fig3)

st.markdown("---")

# ===========================
# 3. Partial Dependence Plot
# ===========================
st.subheader("3️⃣ Partial Dependence Plot (PDP)")

top_feature = "sig_lag1"  # You can change this to your top feature
fig4, ax4 = plt.subplots()
PartialDependenceDisplay.from_estimator(model, df_sample, [top_feature], ax=ax4)
st.pyplot(fig4)

st.markdown("---")

# ===========================
# 4. Residual Plot
# ===========================
st.subheader("4️⃣ Residual Plot (Actual - Predicted)")

residuals = y_test - y_pred
fig5, ax5 = plt.subplots()
ax5.scatter(y_pred, residuals, alpha=0.5)
ax5.axhline(0, color='red', linestyle='--')
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Residuals")
ax5.set_title("Residuals vs Predicted")
st.pyplot(fig5)

st.markdown("---")

st.success("App Loaded Successfully. Use the SHAP Button to Generate Explanations.")
