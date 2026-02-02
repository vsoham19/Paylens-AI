# PayLens AI — Salary Prediction System

PayLens AI is a production-ready machine learning pipeline for job market salary estimation.

## ML Architecture
The system uses a Scikit-learn based pipeline with robust feature engineering to handle real-world job market data.

### Target Variable
- **Target**: `avg_salary` (Average annual salary in $K).

### Data Leakage Control
To maintain model validity and prevent synthetic performance inflation, the following columns are **explicitly excluded** from model features:
- `min_salary`: Directly encodes the lower bound of the target.
- `max_salary`: Directly encodes the upper bound of the target.
- `Salary Estimate`: Contains raw text strings from which the target is derived.

### Features
The model utilizes:
- **Numerical**: Rating, Age of Company, Description Length, Number of Competitors.
- **Categorical**: Job Simplified, Seniority, Location (State), Industry, Sector, Ownership, Company Size, Revenue.
- **Binary Skills**: Python, R, Spark, AWS, Excel.

## API Endpoints
- `GET /metrics`: Returns model performance metadata (RMSE, R²).
- `POST /ask`: LLM-powered domain insights.
- `POST /predict`: Salary estimation for a single job profile.
