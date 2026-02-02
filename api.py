from fastapi import FastAPI
from pydantic import BaseModel
import json
import joblib
from pathlib import Path
import pandas as pd

from llm.rag import RAGAssistant

from typing import Any, Dict
from features.feature_engineering import FeatureEngineer

# -------- Base directory --------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "models" / "random_forest_regressor.pkl"
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "models" / "preprocessor.pkl"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata" / "run_metadata.json"

# -------- Load artifacts safely --------
if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    raise RuntimeError(
        f"Artifacts not found. Run `python main.py` first."
    )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
rag = RAGAssistant(str(METADATA_PATH))

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Explainable ML API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Base directory --------
BASE_DIR = Path(__file__).resolve().parent

# Mount Static Files
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# -------- Request Schemas --------
class PredictRequest(BaseModel):
    features: Dict[str, Any]


class AskRequest(BaseModel):
    question: str


# -------- Endpoints --------
@app.get("/")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(request: PredictRequest):
    # Convert input dict to DataFrame for preprocessor
    df_input = pd.DataFrame([request.features])
    
    # Preprocess
    X_processed = preprocessor.transform(df_input)
    
    # Predict
    prediction = model.predict(X_processed)
    return {"prediction": prediction.tolist()}


@app.get("/metrics")
def metrics():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


@app.post("/ask")
def ask(request: AskRequest):
    try:
        answer = rag.ask(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error: Could not get a response from the LLM. Please check your API key in the .env file. Details: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
