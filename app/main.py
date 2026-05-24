from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="Music Hit Predictor API",
    description="API for predicting whether a song has Top-10 Billboard potential based on audio and contextual features.",
    version="1.0.0"
)

# Global variables for model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[✔] Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Could not load model artifacts. Ensure train.py has been run. Error: {e}")

# Define the input schema with validation
class SongFeatures(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0, description="Danceability score from Spotify")
    energy: float = Field(..., ge=0.0, le=1.0, description="Energy score from Spotify")
    tempo: float = Field(..., ge=0.0, description="Tempo in BPM")
    acousticness: float = Field(..., ge=0.0, le=1.0, description="Acousticness score from Spotify")
    valence: float = Field(..., ge=0.0, le=1.0, description="Valence (musical positiveness) score")
    release_year: int = Field(..., ge=1950, le=2025, description="Year the track was released")
    artist_popularity_proxy: float = Field(..., ge=0.0, description="Log proxy of artist popularity/momentum")

@app.post("/predict")
def predict_hit(features: SongFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not initialized on the server.")

    try:
        # 1. Convert input to DataFrame (to preserve feature names and order)
        input_data = pd.DataFrame([features.dict()])
        
        # Ensure column order matches training phase exactly
        feature_order = [
            'danceability', 'energy', 'tempo', 'acousticness', 'valence', 
            'release_year', 'artist_popularity_proxy'
        ]
        input_data = input_data[feature_order]

        # 2. Preprocessing pipeline (Scaling)
        scaled_data = scaler.transform(input_data)

        # 3. Model Inference
        probability = float(model.predict_proba(scaled_data)[0, 1])
        
        # 4. Apply custom threshold optimized for recall (from training phase)
        optimal_threshold = 0.40
        is_hit = probability >= optimal_threshold

        return {
            "prediction": "Hit" if is_hit else "Not Hit",
            "probability": round(probability, 4),
            "threshold_used": optimal_threshold
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
