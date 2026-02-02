# PayLens AI — Project Implementation Guide

This document provides a comprehensive, end-to-end guide for building and understanding the PayLens AI system. It is designed to guide a developer through the assembly of the project from scratch.

---

## 1. Project Overview
PayLens AI is a modular Machine Learning system for salary prediction. It features a robust data pipeline, a FastAPI backend for inference, and a Streamlit UI for user interaction. It also includes an LLM-powered RAG (Retrieval-Augmented Generation) assistant for domain-specific insights.

---

## 2. Project Roadmap: How to Build from Scratch

If you were to rebuild this project, follow this order to ensure dependencies are handled correctly:

### Phase 1: Foundation & Data
1.  **Create Root Directory**: `ml-data-pipeline`.
2.  **Raw Data**: Create `data/` and place your dataset (`sample.csv`).
3.  **Config & Logging**: Create `config/` (`config.yaml`) and `logging_utils/` (`logger.py`).
4.  **Utility Logic**: Create `data_loader/` to handle file reading (`loader.py`) and schema verification (`validator.py`).

### Phase 2: ML Engine
5.  **Feature Engineering**: Create `features/` (`feature_engineering.py`). This is the most critical logic for data transformation.
6.  **Model Training**: Create `models/` (`trainer.py`) to wrap Scikit-learn logic.
7.  **Orchestration**: Create `pipeline/` (`data_pipeline.py`) to connect everything from loading to saving models.
8.  **Main Entry**: Create `main.py` to trigger the full pipeline.

### Phase 3: Intelligence & API
9.  **LLM Layer**: Create `llm/` (`rag.py`) to implement the Groq-powered RAG logic.
10. **Environment**: Create `.env` for API keys.
11. **FastAPI Server**: Create `api.py` to serve endpoints for prediction, metrics, and Q&A.

### Phase 4: User Interface
12. **Streamlit UI**: Create `streamlit_app.py` for the final visual product.

---

## 3. Directory & File Breakdown

### Root Directory
- `main.py`: The "Start Button." It initializes and runs the `DataPipeline` to train the model and save artifacts.
- `api.py`: The Backend Server. It uses FastAPI to expose the ML model and RAG assistant to the web.
- `streamlit_app.py`: The Graphical User Interface. A Streamlit app that allows users to interact with the project visually.
- `ask.py`: A CLI-only script for testing the RAG assistant without running the web server.
- `test_api.py`: A utility script to verify that all API endpoints are working correctly.
- `.env`: A private file containing sensitive credentials like `GROQ_API_KEY`.
- `README.md`: High-level project summary and feature list.

### `artifacts/` (Generated)
*This folder is created automatically when you run `main.py`.*
- `logs/`: Contains `pipeline.log` for debugging and tracking runs.
- `metadata/`: Contains `run_metadata.json` (Model RMSE, R², etc.).
- `models/`: Stores `linear_regression.pkl` (the model) and `preprocessor.pkl` (feature engineering state).

### `config/`
- `config.yaml`: The central "Brain" of the project where file paths, required columns, and model parameters are defined.

### `data/`
- `sample.csv`: The raw dataset used for training the model.

### `data_loader/`
- `loader.py`: Handles reading CSV files and basic error checking.
- `validator.py`: Ensures the incoming data has the correct columns and data types before training.

### `features/`
- `feature_engineering.py`: Contains the logic to clean text, handle missing values, and convert categorical data into numbers for the model.

### `llm/`
- `rag.py`: Implements Retrieval-Augmented Generation. It reads the model's performance metadata and uses an LLM (via Groq) to explain results.

### `logging_utils/`
- `logger.py`: Provides a standardized logging system for the entire project.

### `models/`
- `trainer.py`: Defines how the model is trained, evaluated (RMSE/R²), and saved to the disk.

### `pipeline/`
- `data_pipeline.py`: The "Conductor." It coordinates between the loader, feature engineer, trainer, and metadata saver.

---

## 4. How to Run

1.  **Train the Model**:
    ```bash
    python main.py
    ```
2.  **Start the Backend**:
    ```bash
    python -m uvicorn api:app --port 8000
    ```
3.  **Start the UI** (in a new terminal):
    ```bash
    streamlit run streamlit_app.py
    ```
