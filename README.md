# 🎾 Padel Analytics — Premier Padel Dashboard & ML Platform

> Business Intelligence dashboard + Machine Learning web application analyzing Premier Padel data across seasons 2023–2025.  
> Built with Power BI, Python (14 ML Models), Flask, and PostgreSQL.

![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Talend](https://img.shields.io/badge/Talend-FF6D70?style=for-the-badge&logo=talend&logoColor=white)

---

## 📖 About

This project provides a comprehensive analytics platform for Premier Padel stakeholders — players, sponsors, and the federation. It combines a **Power BI dashboard** for business intelligence with a **Flask web application** running 14 machine learning models for match prediction, player segmentation, equipment recommendation, talent detection, and performance forecasting.

---

## 📈 Key Results

| Objective | Best Model | Key Metric |
|-----------|-----------|------------|
| Win Prediction | Gradient Boosting | **AUC = 0.938**, Accuracy = 85.5% |
| Points Prediction | XGBoost Regressor | **R² = 0.957**, MAE = 336 |
| Player Segmentation | K-Means (k=4) | **Silhouette = 0.492** |
| Equipment Recommendation | Content-Based | **Type Hit = 100%** |
| Talent Detection | Consensus (IF + LOF) | **4 emerging talents** |
| Win Rate Forecast | ARIMA / Prophet | **MAE = 3.6%–10.9%** (quarterly) |

---

## 📊 Power BI Dashboard Pages

### 1. Players & Staff Brands
- Win ratio per player and per category (Finals, Major, P1, P2)
- Win distribution per round (Final, Quarterfinals, Round of 16/32)
- Total participants per year
- Match statistics (won / lost)

### 2. Sponsors & Equipment Brands
- Win rate by brand (Head, Siux, Bullpadel, Adidas, Babolat, Wilson, Nox)
- Equipment type distribution (Chaussures, T-shirt, Racket, etc.)
- Most used brands ranking
- Correlation: brand performance vs average price
- Brand visibility score

### 3. Federation Overview
- Monthly match participation (2023–2025)
- Most represented nationalities
- Tournament popularity (YouTube views)
- Player loyalty index
- Emerging talents ranking
- Padel growth rate (14.30%)

### 4. Home Page
- Role-based navigation (Players, Sponsors, Federation)
- Custom padel-themed UI design

---

## 🤖 Machine Learning — 14 Models Across 6 Objectives

### Data Preparation
- **14 CSV files** (star schema): 4,319 players, 1,584 matches, 53 tournaments, 141 equipment records
- **Data cleaning:** Social media text→numeric, score parsing ("7-6"→sets/games), 26 name mismatches fixed via fuzzy matching (ñ→Ã± encoding), 11 unrecoverable players dropped (99.1% data retained)
- **27 creative features:** Elo ratings (K=32), partner chemistry, clutch rate, dominance score, opponent features, difference features, ranking trajectory, round-specific win rates, partner loyalty

### Objective 1 — Performance Prediction

#### Classification (Win Prediction)
3 models · 4,132 samples · 27 features · Binary (Win/Loss)

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| **Gradient Boosting** ★ | **85.49%** | **0.854** | **0.938** |
| Random Forest | 85.13% | 0.848 | 0.937 |
| XGBoost | 82.95% | 0.830 | 0.931 |

#### Regression (Points Prediction)
3 models · 331 players · 18 features · Log-transformed target

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **XGBoost** ★ | **0.957** | **586** | **336** |
| Ridge (L2) | 0.827 | 1,173 | 547 |
| Lasso (L1) | 0.552 | 1,888 | 627 |

### Objective 2 — Player Segmentation (Clustering)
3 models · 4,319 players · 10 features · k=4

| Model | Silhouette ↑ | Davies-Bouldin ↓ |
|-------|-------------|------------------|
| **K-Means** ★ | **0.492** | **0.804** |
| GMM | 0.487 | 1.222 |
| Hierarchical (Ward) | 0.486 | 1.025 |

**Segments:** Stars (18), Contenders (137), Regulars (1,642), Newcomers (2,522)

### Objective 3 — Equipment Recommendation
1 method · 12 pro players with gear data · Content-Based Filtering

| Metric | Score |
|--------|-------|
| Brand Hit Rate | 41.7% (5/12) |
| **Type Hit Rate** | **100% (12/12)** |

Uses cosine similarity between player feature vectors to match users to similar pros' equipment.

### Objective 4 — Emerging Talent Detection (Anomaly Detection)
2 models · 205 players with 3+ matches · 9 features

| Model | Anomalies | Emerging Talents |
|-------|-----------|-----------------|
| Isolation Forest | 22 | 4 |
| LOF | 22 | 4 |
| **Consensus** | — | **4** |

**Detected talents:** Joao Maria Caiano (POR, #167, 100% WR), Julian Lacamoire (ARG, #144, 100% WR), F.M. Gil Morales (ESP, #955, 69.6% WR), Jaume Romera Barcelo (ESP, #98, 100% WR)

### Objective 5 — Win Rate Forecast (Time Series)
2 models · Quarterly aggregation · 107 eligible players

| Model | MAE (quarterly) |
|-------|----------------|
| **ARIMA** | **3.6%–10.9%** |
| Prophet | 6.2%–21.6% |

Switched from monthly (MAE ~25%) to quarterly to reduce noise. Features include tournament category weights, max round reached, dominance score, and late-round appearance rate.

---

## 🌐 Flask Web Application

All 14 models deployed in a web app with **8 pages** and **7 API endpoints**.

| Page | Models Used | Description |
|------|------------|-------------|
| Dashboard | — | Stats overview, top 10 players |
| Win Prediction | RF, GB, XGBoost | Player search + Team vs Team with H2H history |
| Points Forecast | Ridge, Lasso, XGBoost | Predict ranking points, visual comparison |
| Segments | K-Means, GMM, Hierarchical | All 3 models shown per player |
| Equipment | Cosine Similarity | Gear recommendation from similar pros |
| Talent Scout | Isolation Forest, LOF | Both models + consensus table |
| Forecast | ARIMA, Prophet | Quarterly future predictions for any player |

**Display filter:** Only 294 players with match data appear in search. All 4,319 used for training.

### Running the webapp
```bash
cd padel_webapp
pip install flask pandas numpy scikit-learn xgboost joblib prophet statsmodels
python app.py
# Open http://localhost:5000
```

### Deploying via ngrok
```bash
ngrok http 5000
# Share the generated URL
```

---

## 🏗️ Architecture

```
📁 padel-analytics/
├── 📊 Power BI
│   ├── official_dashboard.pbix
│   └── DimSecurity.csv
│
├── 🤖 ML Web Application
│   ├── app.py                       # Flask app (908 lines, 14 models)
│   ├── padel_ml_pipeline.py         # Training pipeline with per-model output
│   ├── requirements.txt
│   ├── models/                      # 19 trained .pkl files
│   │   ├── classifier_random_forest.pkl
│   │   ├── classifier_gradient_boosting.pkl
│   │   ├── classifier_xgboost.pkl
│   │   ├── regressor_ridge.pkl
│   │   ├── regressor_lasso.pkl
│   │   ├── regressor_xgboost.pkl
│   │   ├── kmeans.pkl
│   │   ├── gmm.pkl
│   │   ├── iso_forest.pkl
│   │   └── scaler_*.pkl, le_*.pkl
│   ├── data/                        # 14 CSV files (star schema)
│   ├── templates/                   # 8 HTML pages
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── predict.html
│   │   ├── points.html
│   │   ├── segments.html
│   │   ├── recommend.html
│   │   ├── talent.html
│   │   └── forecast.html
│   └── output/                      # Pipeline output (per-model folders)
│       ├── 1_classification/{Random_Forest,Gradient_Boosting,XGBoost}/
│       ├── 2_regression/{Ridge,Lasso,XGBoost}/
│       ├── 3_clustering/{KMeans,GMM,Hierarchical}/
│       ├── 4_recommendation/Content_Based/
│       ├── 5_anomaly_detection/{Isolation_Forest,LOF}/
│       └── 6_time_series/{ARIMA,Prophet}/
│
├── 📄 Presentations
│   ├── professor_presentation.html  # Objective blocks with graphs
│   └── dummies_presentation.html    # Beginner-friendly explanations
│
├── 🖼️ backgrounds/
│   ├── padel_players_bg.png
│   ├── padel_sponsors_bg.png
│   ├── padel_federation_bg.png
│   └── padel_accueil_bg.png
│
├── 📖 README.md
└── 📄 .gitignore
```

---

## 🔄 Data Pipeline

```
┌─────────────┐     ┌──────────┐     ┌────────────┐
│   Sources   │────▶│  Talend  │────▶│ PostgreSQL │
│ (Web / API) │     │  (ETL)   │     │   (DWH)    │
└─────────────┘     └──────────┘     └─────┬──────┘
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                    Power BI          Flask App         ML Pipeline
                   (Dashboard)     (14 models live)   (Training + Graphs)
```

---

## 🔐 Security

### Row-Level Security (RLS)
Role-based data filtering using DAX:

| Role | Access | DAX Filter |
|------|--------|------------|
| Joueur | Own stats only | `LOOKUPVALUE` on `dim_player` |
| Sponsor | Own brand only | `LOOKUPVALUE` on `dim_brand` |
| Federation | Full access | `TRUE()` |

---

## 🧠 Feature Engineering Highlights

The creative features that improved classification from 63% → 85.5%:

| Feature | What it captures |
|---------|-----------------|
| Elo Rating (K=32) | Dynamic team strength, updated per match |
| Partner Chemistry | Team win rate minus avg individual win rate |
| Clutch Rate | Win rate in 3-set matches (pressure situations) |
| Dominance Score | (games_won − games_lost) / total_games |
| Opponent Features | Other team's stats + gap between teams |
| Ranking Trajectory | Position slope over multiple years |
| Partner Loyalty | 1 / unique_partners (consistency) |
| Round-specific WR | Early (R64/R32), Mid (R16/QF), Late (SF/Final) |
| Category WR | Win rate in Major vs P1 vs P2 tournaments |

---

## 📊 Key KPIs

| KPI | Value |
|-----|-------|
| Total Players | 4,319 |
| Active Players (with matches) | 294 |
| Total Matches | 1,584 |
| Total Tournaments | 53 |
| Nationalities | 110 |
| Sponsored Players | 12 |
| ML Models | 14 |
| Creative Features | 27 |
| Padel Growth Rate | 14.30% |

---

## ⚙️ Setup & Installation

### Prerequisites
- Power BI Desktop
- PostgreSQL
- Talend Open Studio
- Python 3.10+ with packages:
  ```bash
  pip install flask pandas numpy scikit-learn xgboost joblib prophet statsmodels
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/khalil0852/Esprit-PI-4erpbi4-Padel-Analytics.git
   ```
2. Import Talend jobs to load data into PostgreSQL
3. Open `official_dashboard.pbix` in Power BI Desktop
4. Update the PostgreSQL connection in **Home → Transform Data → Data Source Settings**
5. For the ML webapp:
   ```bash
   cd padel_webapp
   pip install -r requirements.txt
   python app.py
   ```
6. Open `http://localhost:5000` in your browser

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| **Talend Open Studio** | ETL — Extract, Transform, Load |
| **PostgreSQL** | Data Warehouse |
| **Power BI Desktop** | Dashboard & visualizations |
| **DAX** | Measures, KPIs, RLS |
| **Python** | ML pipeline (14 models) |
| **Flask** | Web application backend |
| **scikit-learn** | Classification, Clustering, Anomaly Detection |
| **XGBoost** | Classification & Regression |
| **Prophet** | Time Series Forecasting |
| **statsmodels** | ARIMA Time Series |
| **Power Automate Desktop** | Automated refresh |
| **ngrok** | Web app deployment |
| **Git / GitHub** | Version control |



## 📝 License

This project is developed as part of an academic project at ESPRIT.
