🧠 PayLens AI — Salary Intelligence & Prediction System
📌 Overview

PayLens AI is an end-to-end machine learning system designed to predict job salary ranges and generate AI-powered insights from real-world job market data.

The project demonstrates a production-style ML architecture, integrating:
-Data preprocessing pipelines
-Feature engineering with leakage control
-Model training and evaluation
-REST API deployment
-LLM-powered analytical insights
-Containerized and scalable system design

🎯 Problem Statement

Job seekers and recruiters often struggle to estimate realistic salary expectations due to:
-Inconsistent salary data
-Lack of structured market insights
-Hidden feature relationships
-Dynamic job market conditions

PayLens AI solves this by:
-Predicting salary ranges based on job features
-Providing explainable insights
-Enabling intelligent querying via LLMs

🏗️ System Architecture

PayLens AI follows a modular ML engineering architecture:

Data → Preprocessing → Feature Engineering → Model Training → API → LLM Insights → UI


This structure ensures:
-Reproducibility
-Scalability
-Maintainability
-Production readiness

⚙️ Key Features
🔹 ML Pipeline
-Automated CSV loading & validation
-Config-driven preprocessing
-Feature leakage prevention
-Scalable training workflow

🔹 Feature Engineering
Includes:
-Numeric feature normalization
-Categorical encoding
-Binary skill flags
-Text-derived features
-Salary leakage removal

🔹 Model Training
-Linear regression model
-Configurable training parameters
-Artifact persistence
-Evaluation metrics logging

🔹 REST API (FastAPI)
Provides endpoints:
/predict → Salary prediction
/metrics → Model performance
/ask → AI insights using LLM

🔹 LLM Integration
-Uses a large language model to:
-Interpret prediction results
-Generate job market insights
-Answer user queries

🔹 Production-Style Design
Includes:
-Config-driven YAML setup
-Artifact management
-Modular architecture
-Containerization readiness

📁 Project Structure
PayLens-AI/
│
├── data_loader/       # Data ingestion & validation
├── features/          # Feature engineering logic
├── models/            # Model training & persistence
├── pipeline/          # ML pipeline orchestration
├── config/            # YAML configurations
├── artifacts/         # Models, logs, metadata
├── api.py             # FastAPI server
├── ui/                # Streamlit interface

🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python main.py

3️⃣ Run the API
uvicorn api:app --reload

4️⃣ Run the UI
streamlit run ui/app.py

📊 Model Performance

Metrics are automatically stored in:

artifacts/metadata/

Includes:
-RMSE
-R² Score
-Training parameters

🧠 Key ML Engineering Concepts Demonstrated
-Feature leakage prevention
-Config-driven pipelines
-Model artifact management

End-to-end ML lifecycle

Hybrid ML + LLM architecture
