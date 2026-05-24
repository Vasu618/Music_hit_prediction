# Music Hit Predictor

## Overview

This project is an end-to-end machine learning system designed to predict the likelihood of a music track reaching the Billboard Top 10. The pipeline processes raw Spotify audio features and temporal metadata, utilizing gradient boosting to output probabilistic predictions via a RESTful API. The system is designed to support A&R scouting and release strategy.

## Problem Statement

Predicting commercial success in the music industry is inherently challenging due to shifting trends and audience behaviors. Relying exclusively on domain intuition introduces subjective bias and inconsistency. The objective of this project is to provide a quantitative, data-driven baseline for track evaluation by framing hit prediction as a binary classification problem.

## Dataset & Features

The model operates on track metadata and derived audio features.

* `danceability`: Algorithmically derived suitability for dancing (tempo, beat strength).
* `energy`: Perceptual measure of track intensity and acoustic activity.
* `tempo`: Track speed measured in beats per minute (BPM).
* `acousticness`: Confidence metric indicating whether the track is acoustic.
* `valence`: The degree of musical positiveness conveyed by the audio.
* `release_year`: Temporal anchor to prevent the model from learning static audio trends, accounting for decadal shifts in pop music.
* `artist_popularity_proxy`: A logarithmic feature capturing an artist's historical momentum to account for existing audience bias.

## Machine Learning Pipeline

The modeling approach focuses on addressing extreme class imbalance and prioritizing recall.

* **Algorithm Selection:** Extreme Gradient Boosting (XGBoost) was chosen for its ability to handle non-linear relationships among audio features and its robustness to outliers compared to linear models.
* **Class Balancing:** The dataset exhibits severe class imbalance (the vast majority of tracks do not hit the Top 10). The `scale_pos_weight` hyperparameter was utilized during training to heavily penalize false negatives, preventing the model from degenerating into a majority-class classifier.
* **Threshold Tuning:** The decision boundary was shifted from the default 0.5 to a lower threshold (e.g., 0.40). This tuning specifically optimizes the recall metric, ensuring that borderline tracks with hit potential are flagged for human review rather than discarded by a strict statistical threshold.
* **Preprocessing:** Features are standardized using `StandardScaler` to ensure consistent convergence and feature importance evaluation.

## Model Architecture

* **Data Ingestion:** Batch processing pipeline for historical track data.
* **Feature Engineering:** Derivation of temporal and momentum proxies.
* **Estimator:** XGBClassifier optimized for logarithmic loss (logloss).
* **Inference Backend:** FastAPI service for low-latency JSON predictions.

## Results

By incorporating temporal and popularity context alongside raw audio features, the model significantly outperforms baseline audio-only benchmarks.

* **Baseline Performance:** Audio-only ROC-AUC approximates 0.65.
* **Enhanced Model:** Integrates contextual features, yielding improved ROC-AUC and a stable precision-recall trade-off.
* **Validation Focus:** Metrics emphasize recall over precision, accepting a higher false-positive rate to guarantee that true potential hits are captured for secondary human review.

## Business Impact

* **Resource Allocation:** Provides a statistical baseline to help record labels allocate promotional budgets toward tracks with the highest baseline probability of charting.
* **Scouting Efficiency:** Enables A&R teams to programmatically filter large volumes of independent releases, prioritizing tracks that fit current successful audio profiles.
* **Strategy Adjustments:** Highlights the importance of specific audio features relative to the current release year, aiding producers in trend alignment.

## Limitations

* **Missing Exogenous Variables:** The model operates entirely on tabular metadata and audio features. It is blind to critical external factors such as TikTok virality, lyrical themes, and music video impact.
* **Binary Classification Constraint:** Framing chart success as a static binary target fails to capture the continuous nature of chart positioning or track longevity.
* **Proxy Limitations:** True artist popularity requires real-time social and streaming data; the current popularity proxy is limited to historical presence in the training set.

## Future Improvements

* **Natural Language Processing:** Integrate transformer-based sentiment and thematic analysis of lyrics.
* **Ranking Architecture:** Transition from binary classification to a Learning-to-Rank (LTR) approach to predict relative chart positions week-over-week.
* **Sequential Modeling:** Process sequential audio waveforms or Mel-spectrograms directly rather than relying solely on pre-aggregated numerical metrics.

## Tech Stack

* **Language:** Python 3.9+
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Deployment:** FastAPI, Uvicorn, Joblib

## Local Setup

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Vasu618/Music_hit_prediction.git
cd Music_hit_prediction
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the API server:
```bash
cd app
uvicorn main:app --reload
```

## API Usage

Start the FastAPI server and send a POST request to `/predict`.

**Endpoint:** `POST /predict`

**Example Request:**
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

**Example Response:**
```json
{
  "prediction": "Hit",
  "probability": 0.68,
  "threshold_used": 0.4
}
```

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── .gitkeep
├── training/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Author

Vasu
