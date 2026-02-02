import streamlit as st
import requests
import json
import pandas as pd

# Backend Configuration
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PayLens AI",
    page_icon="💰",
    layout="wide",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Project Header
st.title("💰 PayLens AI")
st.subheader("Job Market Compensation Intelligence")
st.markdown("---")

# Layout with two columns for the main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # LLM Q&A Section
    st.header("🔮 Ask PayLens AI")
    user_question = st.text_input("Ask about salary trends, skills, or job market insights:", placeholder="e.g., Does Python skill increase salary?")
    
    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{BASE_URL}/ask", json={"question": user_question})
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer found.")
                        st.success("PayLens AI Says:")
                        st.write(answer)
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")
        else:
            st.warning("Please enter a question.")

    st.markdown("---")

    # Model Metrics Section
    st.header(" Model Performance Metrics")
    if st.button("Load Metrics"):
        with st.spinner("Fetching metrics..."):
            try:
                response = requests.get(f"{BASE_URL}/metrics")
                if response.status_code == 200:
                    metrics_data = response.json()
                    
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("RMSE", f"{metrics_data['metrics']['rmse']:.2f}")
                    m_col2.metric("R² Score", f"{metrics_data['metrics']['r2']:.2f}")
                    
                    with st.expander("View Full Metadata"):
                        st.json(metrics_data)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

with col2:
    # Salary Prediction Section
    st.header(" Salary Prediction")
    st.write("Input job features to estimate the average salary.")

    # Input fields based on feature_engineering.py
    # Numeric
    rating = st.slider("Company Rating", 1.0, 5.0, 3.5, 0.1)
    age = st.number_input("Company Age", 0, 200, 20)
    desc_len = st.number_input("Job Description Length", 100, 10000, 1500)
    num_comp = st.number_input("Number of Competitors", 0, 10, 0)

    # Categorical/Binary
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        job_simp = st.selectbox("Job Title", ["data scientist", "data engineer", "analyst", "mle", "manager", "director"])
        seniority = st.selectbox("Seniority", ["na", "jr", "sr"])
        python_yn = st.checkbox("Python Required", value=True)
        excel_yn = st.checkbox("Excel Required")
    
    with col_feat2:
        job_state = st.text_input("Job State (e.g., NY, CA)", value="NY")
        type_ownership = st.selectbox("Company Type", ["Private Practice / Firm", "Proprietorship", "Public", "Nonprofit Organization", "Subsidiary or Business Segment", "Government"])
        spark_yn = st.checkbox("Spark Required")
        aws_yn = st.checkbox("AWS Required")

    if st.button("Predict Salary"):
        # Construct the features dictionary
        # Note: The backend expects the exact feature names used in training/preprocessing
        features = {
            "Rating": rating,
            "age": age,
            "desc_len": desc_len,
            "num_comp": num_comp,
            "job_simp": job_simp,
            "seniority": seniority,
            "job_state": job_state,
            "Type of ownership": type_ownership,
            "python_yn": 1 if python_yn else 0,
            "excel": 1 if excel_yn else 0,
            "spark": 1 if spark_yn else 0,
            "aws": 1 if aws_yn else 0,
            # Add placeholders for other categorical fields required by preprocessor if necessary
            "Industry": "na",
            "Sector": "na",
            "Size": "na",
            "Revenue": "na",
            "R_yn": 0
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{BASE_URL}/predict", json={"features": features})
                if response.status_code == 200:
                    prediction = response.json().get("prediction", [0])[0]
                    st.success(f"Predicted Annual Salary: **${prediction:.2f}K**")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

st.markdown("---")
st.caption("PayLens AI - Demonstration Project for Salary Intelligence")
