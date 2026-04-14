# Padel Analytics — Web Application

A Flask web app serving ML models for padel player analytics.

## Features

| Page | Model | What it does |
|------|-------|-------------|
| **Win Prediction** | Random Forest, Gradient Boosting, XGBoost | Predicts match win probability |
| **Points Forecast** | Ridge, Lasso, XGBoost | Predicts ranking points |
| **Player Segments** | K-Means Clustering | Segments 4,319 players into Stars/Contenders/Regulars/Newcomers |
| **Equipment Rec** | Cosine Similarity | Recommends gear based on pro player profile matching |
| **Talent Scout** | Isolation Forest | Detects emerging talents who outperform their ranking |

## Setup

### 1. Install dependencies

```bash
pip install flask pandas numpy scikit-learn xgboost joblib
```

### 2. Generate the models (first time only)

Put your CSV files in `./data/` and run the training pipeline:

```bash
python padel_ml_pipeline.py
```

This creates `./models/` with 14 `.pkl` model files.

### 3. Run the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Project Structure

```
webapp/
├── app.py                  # Flask app (all routes + ML inference)
├── requirements.txt        # Python dependencies
├── README.md
├── templates/
│   ├── base.html           # Layout + nav + CSS
│   ├── index.html          # Dashboard
│   ├── predict.html        # Win prediction
│   ├── points.html         # Points forecast
│   ├── segments.html       # Player segmentation
│   ├── recommend.html      # Equipment recommendation
│   └── talent.html         # Emerging talent detection
├── models/                 # Trained ML models (.pkl)
│   ├── classifier_rf.pkl
│   ├── classifier_gb.pkl
│   ├── classifier_xgb.pkl
│   ├── regressor_ridge.pkl
│   ├── regressor_lasso.pkl
│   ├── regressor_xgb.pkl
│   ├── kmeans.pkl
│   ├── gmm.pkl
│   ├── iso_forest.pkl
│   ├── scaler_cluster.pkl
│   ├── scaler_recommender.pkl
│   ├── scaler_anomaly.pkl
│   ├── le_gender.pkl
│   └── le_country.pkl
└── data/                   # CSV data files
    ├── fact_match.csv
    ├── fact_player.csv
    ├── fact_equipement.csv
    ├── dim_player.csv
    └── ... (all CSVs)
```

## API Endpoints

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/api/predict` | POST | `{"player_name": "...", "round": "..."}` | Win probabilities from 3 models |
| `/api/points` | POST | `{"player_name": "..."}` | Predicted points from 3 models |
| `/api/segment-player` | POST | `{"player_name": "..."}` | Player's cluster/segment |
| `/api/recommend` | POST | `{"player_name": "...", "type": "racket"}` | Equipment recommendations |
| `/api/check-talent` | POST | `{"player_name": "..."}` | Anomaly detection result |
| `/api/search-players` | GET | `?q=juan` | Player name autocomplete |
