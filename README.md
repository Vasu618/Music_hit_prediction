# Music Hit Predictor (Production System)

A deployable ML system that predicts the Top-10 Billboard potential of a music track based on Spotify audio features and artist momentum.

## Project Structure
```text
music-hit-predictor/
├── app/
│   ├── main.py         # FastAPI application and endpoints
│   ├── model.pkl       # Serialized XGBoost model (generated after training)
│   ├── scaler.pkl      # Serialized StandardScaler (generated after training)
├── training/
│   └── train.py        # Feature engineering and model retraining script
├── notebooks/
│   └── analysis.ipynb  # Exploratory Data Analysis (EDA)
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## Setup & Running

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
You must run the training script first to generate `model.pkl` and `scaler.pkl`.
```bash
cd training
python train.py
```

3. **Run the API**
Start the FastAPI server.
```bash
cd ../app
uvicorn main:app --reload
```

4. **Test the API**
Send a POST request to `http://127.0.0.1:8000/predict` with JSON payload:
```json
{
  "danceability": 0.82,
  "energy": 0.65,
  "tempo": 120.5,
  "acousticness": 0.12,
  "valence": 0.70,
  "release_year": 2024,
  "artist_popularity_proxy": 5.4
}
```
