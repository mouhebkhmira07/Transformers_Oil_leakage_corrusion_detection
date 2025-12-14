# app/main.py (The Universal Template)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()
class ModelInput(BaseModel):
    # Right now: Agriculture inputs
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    # LATER, you change this to: global_active_power, voltage, etc.

# Load model logic...
# Ensure we look for the model in the correct relative path
# Assuming this runs from project root or src/app
MODEL_PATH = os.path.join("models", "crop_recommender.pkl") 
# Fallback to absolute path if needed or check existence
if not os.path.exists(MODEL_PATH):
     # Try looking one level up if running from src/app
    MODEL_PATH = os.path.join("..", "models", "crop_recommender.pkl")

# For now, let's just use a placeholder or try to load if it exists
# model = joblib.load(MODEL_PATH) 

@app.post("/predict")
def predict(data: ModelInput):
    # Convert Pydantic -> DataFrame
    df = pd.DataFrame([data.dict()])

    return {"result": "Model not loaded yet, but API is working!", "input": data.dict()}


@app.get("/")
def root():
    """Basic liveness endpoint."""
    return {"status": "ok", "message": "API is running"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
