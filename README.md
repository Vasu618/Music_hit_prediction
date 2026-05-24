Music Hit Predictor (Production System)

Project Overview
This project predicts the Top-10 Billboard chart potential of a music track using historical audio features and artist data. Accurate hit prediction is critical for record labels and A&R teams to optimize marketing budgets, discover emerging talent, and maximize streaming revenue.

Problem Statement
The music industry often relies heavily on intuition to identify the next breakout hit. Investing in the wrong track leads to wasted promotional budgets, while missing a viral track results in lost market share. The goal of this project is to build a highly accurate, automated classification model that can predict a track's hit potential based on intrinsic audio characteristics and external contextual momentum.

Dataset Description
The model is trained on a robust dataset of historical track metadata and Spotify audio features.

Features:
danceability: Suitability of a track for dancing based on tempo, rhythm stability, and beat strength.
energy: Perceptual measure of intensity and activity.
tempo: Overall estimated tempo of a track in beats per minute (BPM).
acousticness: Confidence measure of whether the track is acoustic.
valence: Musical positiveness conveyed by a track.
release_year: Temporal anchor to account for shifting musical trends over decades.
artist_popularity_proxy: Logarithmic proxy of an artist's momentum and historical presence.

Approach & Model Explanation
This project utilizes an Extreme Gradient Boosting (XGBoost) Classifier. Gradient boosting is a powerful ensemble learning technique that builds sequential decision trees to minimize prediction errors.

Feature Engineering: Extracted contextual and temporal features such as release_year and artist_popularity_proxy to allow the model to learn that audio trends change over time and that chart success is heavily biased by an artist's existing audience.
Scaling: Data is standardized using StandardScaler to ensure uniform feature distribution for the underlying algorithms.
Class Balancing: A dynamic scaling weight (scale_pos_weight) is applied to handle the severe class imbalance between typical tracks and Billboard Top-10 hits.
Threshold Tuning: The decision threshold is custom-tuned (e.g., 0.40) to optimize recall, ensuring that potential hits are not overlooked.

Architecture:
Data Ingestion & Preprocessing Pipeline
StandardScaler for Feature Normalization
XGBoost Classifier (Logloss Evaluation Metric)
FastAPI Backend for Real-Time Inference

Results
The model demonstrates a significant improvement over baseline audio-only predictions:

Baseline Audio-Only ROC-AUC: approx 0.65
Advanced Model (with Popularity & Temporal Context): Captures complex nonlinear relationships between audio trends and artist momentum.
The model successfully flags high-potential tracks, balancing precision with a strong recall rate to ensure A&R teams receive comprehensive recommendations.

Business Impact
Optimized Scouting: A&R teams can systematically filter thousands of tracks to find high-potential artists.
Data-Driven Marketing: Labels can allocate promotional budgets mathematically to tracks with the highest statistical probability of success.
Trend Adaptation: By understanding temporal feature shifts, producers can align new music with current algorithmic preferences.

Limitations & Future Improvements
Limitations: The current model relies on tabular metadata. It does not account for exogenous shock variables such as sudden TikTok virality, lyrical content, or music video aesthetics.
Future Improvements:
Incorporate Natural Language Processing (NLP) to analyze lyrics and themes.
Integrate real-time social media sentiment and engagement metrics.
Build a ranking-based recommendation system to rank tracks within a given week rather than isolated binary classification.

Tech Stack
Python 3.x
XGBoost (Machine Learning)
Scikit-Learn (Preprocessing)
Pandas & NumPy (Data Manipulation)
FastAPI & Uvicorn (Model Deployment)
Joblib (Model Serialization)

Author
Vasu
