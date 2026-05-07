import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
xgb_path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
feature_path = os.path.join(BASE_DIR, "model", "feature_names.pkl")
data_path = os.path.join(BASE_DIR, "data", "intern_performance_final_data.csv")

# =========================
# LOAD MODELS
# =========================
with open(rf_path, "rb") as f:
    rf_model = pickle.load(f)

with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

with open(feature_path, "rb") as f:
    features = pickle.load(f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Intern Performance AI", layout="centered")

st.title("🎯 Intern Performance Prediction System")
st.write("Predict intern performance using Machine Learning (Random Forest + XGBoost)")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Enter Intern Details")

task_time = st.slider("Task Completion Time (hours)", 1, 12, 5)
feedback = st.slider("Feedback Rating (1-5)", 1.0, 5.0, 4.0)
attendance = st.slider("Attendance Rate (%)", 50, 100, 85)

# =========================
# PREDICTION SECTION
# =========================
if st.button("🔮 Predict Performance"):

    input_data = np.array([[task_time, feedback, attendance]])

    # Predictions
    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    # Probabilities
    rf_prob = rf_model.predict_proba(input_data)[0][1]
    xgb_prob = xgb_model.predict_proba(input_data)[0][1]

    # Labels
    rf_label = "Excel 🚀" if rf_pred == 1 else "Struggle ⚠️"
    xgb_label = "Excel 🚀" if xgb_pred == 1 else "Struggle ⚠️"

    # =========================
    # RESULTS DISPLAY
    # =========================
    st.markdown("## 📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Random Forest")
        st.write("Prediction:", rf_label)
        st.write("Confidence:", round(rf_prob, 2))

    with col2:
        st.subheader("🚀 XGBoost")
        st.write("Prediction:", xgb_label)
        st.write("Confidence:", round(xgb_prob, 2))

    # =========================
    # COMPARISON TABLE
    # =========================
    st.markdown("### ⚖️ Model Comparison")

    st.table({
        "Model": ["Random Forest", "XGBoost"],
        "Prediction": [rf_label, xgb_label],
        "Confidence": [round(rf_prob, 2), round(xgb_prob, 2)]
    })

st.markdown("---")

# =========================
# ANALYTICS SECTION
# =========================
st.title("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Dataset", "Model Insights", "Feature Importance"])

# =========================
# TAB 1 - DATASET
# =========================
with tab1:
    st.subheader("📁 Dataset Overview")

    df = pd.read_csv(data_path)

    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.write("Outcome Distribution")
    st.bar_chart(df["outcome"].value_counts())

# =========================
# TAB 2 - MODEL INSIGHTS
# =========================
with tab2:
    st.subheader("🧠 Key Insights")

    st.markdown("""
    ✔ Task completion time is most important factor  
    ✔ Feedback rating strongly impacts performance  
    ✔ Attendance plays moderate role  
    ✔ XGBoost slightly more stable than Random Forest  
    """)

# =========================
# TAB 3 - FEATURE IMPORTANCE
# =========================
with tab3:
    st.subheader("📈 Feature Importance")

    rf_importance = rf_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_

    # Random Forest Plot
    fig1, ax1 = plt.subplots()
    ax1.barh(features, rf_importance)
    ax1.set_title("Random Forest Feature Importance")
    st.pyplot(fig1)

    # XGBoost Plot
    fig2, ax2 = plt.subplots()
    ax2.barh(features, xgb_importance)
    ax2.set_title("XGBoost Feature Importance")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning (Random Forest + XGBoost)")