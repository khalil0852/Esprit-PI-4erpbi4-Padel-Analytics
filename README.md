<div align="center">
🎾 Padel Analytics
An End-to-End MLOps Platform for the World's Fastest-Growing Sport
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Embedded-F2C811?logo=powerbi&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-Registered-0194E2?logo=mlflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-06D6A0)
Premier Padel · Seasons 2023–2025
Overview · Features · Quick Start · Architecture · ML Models · Team
</div>
---
📋 Overview
Padel Analytics is a complete intelligence platform for the Premier Padel professional circuit. It transforms scattered match data, equipment catalogs, social media metrics, and federation rankings into actionable insight — for three stakeholder personas:
👑 Federations	🏷️ Sponsors	⚡ Players
Sport growth, audience metrics, country-level analytics	Brand visibility, ROI, sponsored player performance	Personal stats, match history, equipment recommendations
The Problem
Padel is projected to hit 50M+ players by 2027, yet its data infrastructure is decades behind sports like tennis or football. Match results live on federation sites, equipment catalogs on e-commerce platforms, rankings on yet other portals — with no integration. We fixed that.
The Numbers
<div align="center">
14	1,584	4,319	53	23	4
ML Models	Matches	Players	Tournaments	KPIs	Docker Services
</div>
---
✨ Features
🎯 Role-Based Access (RBAC)
Page	Federation	Sponsor	Joueur	Admin
Dashboard (Power BI)	✅	✅	✅	✅
Win Prediction	✅	❌	✅	✅
Points Forecast	✅	❌	✅	✅
Player Segments	✅	✅	❌	✅
Equipment Recommender	✅	✅	✅	✅
Talent Scout	✅	❌	❌	✅
Win-Rate Forecast	✅	✅	✅	✅
Operations Panel	❌	❌	❌	✅
🔒 Two-Layer Security Model
Layer 1 — Flask RBAC: page-level access control via `@role_required` decorator
Layer 2 — Power BI RLS: data-level filtering via `USERPRINCIPALNAME()` + `DimSecurity` table
📊 Three Power BI Dashboards
Same dataset, three filtered views via Row-Level Security:
```dax
[brand] IN SELECTCOLUMNS(
    FILTER(DimSecurity,
        DimSecurity[UserEmail] = USERPRINCIPALNAME() &&
        DimSecurity[UserRole] = "Sponsor"),
    "Brand", DimSecurity[LinkedEntity]
)
```
🤖 14 Production-Ready ML Models
Every model is tracked in MLflow with champion/challenger aliases for A/B testing and rollback.
⚙️ Full MLOps Stack
Experiment tracking → orchestration → monitoring → drift detection → automated retraining. All running in Docker, deployable with a single command.
---
🛠️ Tech Stack
<div align="center">
Layer	Technology
Data Warehouse	![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
ETL	![Talend](https://img.shields.io/badge/Talend_Open_Studio-FF6D70?logo=talend&logoColor=white)
Business Intelligence	![Power BI](https://img.shields.io/badge/Power_BI-F2C811?logo=powerbi&logoColor=black)
Web Application	![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) ![Python](https://img.shields.io/badge/Python_3.13-3776AB?logo=python&logoColor=white)
Containerization	![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
Orchestration	![Airflow](https://img.shields.io/badge/Apache_Airflow-017CEE?logo=apacheairflow&logoColor=white) ![n8n](https://img.shields.io/badge/n8n-EA4B71?logo=n8n&logoColor=white)
ML / MLOps	![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-CC0000?logo=xgboost&logoColor=white)
Monitoring	![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white) ![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)
</div>
---
🏗️ Architecture
A 4-layer architecture following the GIMSI methodology:
```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PRESENTATION                                              │
│  • Flask Web App (port 5000)   • Power BI iframe   • Grafana (3000)  │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — DATA WAREHOUSE                                            │
│  PostgreSQL · Constellation Schema · 9 dimensions + 4 facts          │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — ETL / ORCHESTRATION                                       │
│  Talend (batch ETL) · Airflow (nightly DAG) · n8n (event-driven)     │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — DATA SOURCES (44 raw files)                               │
│  FIP · Premier Padel · E-commerce (5 brands) · Social media          │
└──────────────────────────────────────────────────────────────────────┘
```
Docker Stack
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   padel-app     │  │     mlflow      │  │   prometheus    │  │     grafana     │
│   port 5000     │  │   port 5001     │  │   port 9090     │  │   port 3000     │
│ Flask + ML APIs │  │ Tracking server │  │ Metrics scraper │  │  9-panel UI     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
            └───── padel-network (internal DNS resolution) ─────┘
```
---
🧠 Machine Learning Models
Six analytical objectives, 14 trained algorithms, champion selection per task.
🎯 Task	🏆 Champion	📊 Score	🥈 Challengers
Win Prediction	Gradient Boosting	AUC 0.935 · F1 0.854	Random Forest, XGBoost
Points Forecasting	XGBoost Regressor	R² 0.879 · MAE 104.2	Ridge, Lasso
Player Segmentation	K-Means (k=4)	Silhouette 0.473	GMM, Hierarchical
Equipment Recommendation	Cosine Similarity	Type hit rate 100%	—
Talent Detection	IsoForest + LOF consensus	4 high-confidence talents	—
Win-Rate Forecasting	Prophet	MAE 0.246	ARIMA
Player Segments (K-Means k=4)
Segment	Count	Description
🌟 Stars	17	Top elite, podium contenders
⚡ Contenders	133	Strong professionals, tournament regulars
🎯 Regulars	1,648	Active circuit professionals
🌱 Newcomers	2,521	Emerging or partially active players
---
🚀 Quick Start
Prerequisites
Docker Desktop (Windows / Mac / Linux) with Docker Compose v2+
Power BI Pro (or trial) — for embedded dashboard
Git — to clone the repo
8 GB RAM minimum, 20 GB disk space recommended
Installation
```powershell
# 1. Clone the repository
git clone https://github.com/your-team/padel-analytics.git
cd padel-analytics

# 2. Build & start all services
docker compose up -d --build

# 3. Wait ~2 minutes, then verify
docker compose ps
```
First Boot
After `docker compose up`, the platform exposes:
URL	Service	What it is
`http://localhost:5000`	Web App	Flask UI with login
`http://localhost:5001`	MLflow	Experiment tracking + Model Registry
`http://localhost:9090`	Prometheus	Metrics + alert evaluator
`http://localhost:3000`	Grafana	Real-time monitoring dashboard
`http://localhost:8080`	Airflow	Scheduled DAG runner (optional)
`http://localhost:5678`	n8n	Event-driven workflows (optional)
Initial Training Run
Train all 14 models and register them in MLflow:
```powershell
docker compose exec padel-app python mlflow_pipeline.py
```
Takes ~90 seconds. After completion, open MLflow at `http://localhost:5001` → Models to see all registered models with champion/challenger aliases.
---
---
📁 Project Structure
```
padel-analytics/
├── 📂 data/                          # Cleaned warehouse CSVs
│   ├── fact_match.csv                # 1,584 matches
│   ├── fact_player.csv               # 4,319 player records
│   ├── fact_equipement.csv           # Equipment usage
│   ├── dim_player.csv
│   ├── dim_tournament.csv
│   └── DimSecurity.csv               # RLS mapping
│
├── 📂 models/                        # Trained .pkl artifacts
│   ├── classifier_gradient_boosting.pkl
│   ├── classifier_xgboost.pkl
│   ├── regressor_xgboost.pkl
│   ├── kmeans.pkl
│   ├── iso_forest.pkl
│   └── scaler_*.pkl
│
├── 📂 templates/                     # Flask Jinja2 templates
│   ├── base.html                     # Layout + role-aware nav
│   ├── login.html
│   ├── index.html                    # Dashboard + Power BI iframe
│   ├── predict.html                  # Win prediction UI
│   ├── points.html                   # Points forecast UI
│   ├── segments.html                 # K-Means segments
│   ├── recommend.html                # Equipment recommender
│   ├── talent.html                   # Anomaly-based talent scout
│   ├── forecast.html                 # Time-series forecast
│   ├── operations.html               # Admin ops control panel
│   └── access_denied.html
│
├── 📂 logs/                          # Application logs
│   └── app.log
│
├── 📂 airflow/                       # Airflow DAG definitions
│   └── dags/
│       └── padel_etl_dag.py          # 9-task scheduled pipeline
│
├── 📂 n8n/                           # n8n workflow exports
│   └── padel_workflow.json
│
├── 📂 talend/                        # Talend ETL jobs
│   └── jobs/
│
├── 📂 powerbi/                       # .pbix dashboards
│   └── Padel_Analytics.pbix
│
├── 📄 app.py                         # Flask main entrypoint (~1,200 lines)
├── 📄 padel_ml_pipeline.py           # ML training pipeline
├── 📄 mlflow_pipeline.py             # MLflow registry automation
├── 📄 run_simulations.ps1            # Drift simulation script
│
├── 📄 Dockerfile                     # Flask app image
├── 📄 docker-compose.yml             # 4-service orchestration
├── 📄 prometheus.yml                 # Prometheus config
├── 📄 alerts.yml                     # 4 alert rules
├── 📄 requirements.txt
└── 📄 README.md                      # ← you are here
```
---
🗄️ Data Warehouse Schema
A constellation schema with shared dimensions across 4 fact tables.
Dimensions (9)
Table	Purpose
`dim_player`	Player profile (name, gender, nationality, birth date)
`dim_team`	Doubles team identity
`dim_country`	Country reference data
`dim_gender`	Gender lookup (homme/femme)
`dim_tournament`	Tournament metadata (name, category, prize pool)
`dim_venue`	Tournament location and surface
`dim_round`	Round of competition (R32 → Final)
`dim_date`	Year, quarter, month, week
`dim_racket_equipements`	Equipment reference
Facts (4)
Table	Grain	Rows
`fact_match`	One row per match	1,584
`fact_tournament_match`	Match within tournament context	—
`fact_player`	Annual player ranking	4,319
`fact_equipement`	Equipment usage / sponsorship	—
Why Constellation?
Star schema → too many sparse columns across our 4 business processes
Snowflake schema → over-normalized, slows Power BI Import mode
Constellation → shared dimensions + multiple fact tables (best of both)
---
🔥 MLOps Stack
MLflow — Experiment Tracking + Model Registry
```python
with mlflow.start_run(run_name="gradient_boosting_v1"):
    mlflow.log_params({"n_estimators": 300, "learning_rate": 0.05})
    mlflow.log_metrics({"roc_auc": 0.935, "f1": 0.854})
    mlflow.sklearn.log_model(clf, "model")
```
After training, `mlflow_pipeline.py` automatically:
Queries the experiment for top performers per task
Validates against quality thresholds (AUC > 0.7, R² > 0.5, silhouette > 0.3)
Registers qualifying models in the registry
Promotes champion + challenger aliases
Apache Airflow — Scheduled ETL (9-task DAG)
```
start ─▶ check_sources ─▶ extract_data ─▶ validate_quality ─▶
        transform_features ─▶ train_models ─▶ run_inference ─▶
        generate_report ─▶ end
```
Runs nightly. Retries on failure. Email notifications via SMTP.
n8n — Event-Driven Workflows (11 nodes)
Webhook-triggered retraining, drift notifications, multi-step decision chains. Complements Airflow's scheduled work with reactive automation.
Prometheus + Grafana — Monitoring
Custom metrics exposed by Flask:
Metric	Type	Purpose
`predictions_total`	Counter	Total ML predictions served
`prediction_duration_seconds`	Histogram	p50/p95/p99 latency per endpoint
`model_accuracy`	Gauge	Current accuracy per model
`data_freshness_hours`	Gauge	Hours since last data refresh
`api_errors_total`	Counter	4xx/5xx errors per endpoint
Alert rules (`alerts.yml`):
Alert	Trigger	Catches
`HighLatency`	p95 > 1s for 1 min	Performance regressions
`HighErrorRate`	error rate > 10% for 1 min	Broken deployments
`AccuracyDegradation`	accuracy < 0.80 for 1 min	Model drift
`DataDrift`	data_freshness > 24h	Stale data, ETL failures
Drift Simulation
Validate the entire stack reacts correctly to real conditions:
```powershell
.\run_simulations.ps1
```
This triggers 3 scenarios:
HIGH TRAFFIC — 10 workers × 100 requests over 60s
API ERRORS — 60 malformed requests over 60s
MODEL DRIFT — drop accuracies 25%, bump data freshness past 24h
All 4 alerts fire correctly across the 3 scenarios. ✅
---
🎛️ Operations Dashboard (Admin)
The Operations page (`/operations`) is exclusive to Admin and consolidates the full MLOps stack:
```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️  OPERATIONS · System Control Panel                       │
├─────────────────────────────────────────────────────────────┤
│  🧪 MLflow      📊 Grafana      🔥 Prometheus              │
│  🌊 Airflow     🔄 n8n          📡 Raw Metrics             │
│                                                              │
│  ─── Embedded iframes ───                                   │
│  • MLflow Tracking Server                                    │
│  • Grafana Dashboards                                        │
│  • Prometheus Query UI                                       │
└─────────────────────────────────────────────────────────────┘
```
---
📈 BI Performance Audit
Following GIMSI Phase 4 (Improvement), we conducted a formal audit:
Metric	Value
Total findings	7
Power BI events analyzed	754
Cache hit ratio	99% ✅
High-severity findings (P1)	2
Key Findings
Severity	Finding	Remediation
🔴 HIGH	K-Means Python visual: 3.9s render	Pre-compute clusters in warehouse
🔴 HIGH	0% index usage on 7 tables	Add covering indexes on join keys
🟡 MED	Dataset refresh: 12 min	Implement incremental refresh
🟡 MED	DAX measure proliferation (37)	Consolidate into calculated columns
🟢 LOW	Duplicate DAX logic (3 pairs)	Refactor into shared columns
🟢 LOW	5 unused columns imported	Drop from dataset
🟢 LOW	One WCAG contrast failure	Adjust palette
---
🧪 Running Tests & Simulations
Test the API Endpoints
```bash
# Win prediction (Federation, Joueur, Admin)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Agustin Tapia", "round": "Final"}'

# Points forecast
curl -X POST http://localhost:5000/api/points \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Agustin Tapia"}'

# Equipment recommendations
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Agustin Tapia", "type": "all"}'
```
Trigger Drift Manually
```bash
curl -X POST http://localhost:5000/api/simulate-drift \
  -H "Content-Type: application/json" \
  -d '{"accuracy": 0.65, "freshness_hours": 30}'
```
Check Grafana — both `AccuracyDegradation` and `DataDrift` alerts fire within 60 seconds.
---
🐛 Troubleshooting
Docker BuildKit snapshot error on Windows
```
failed to prepare extraction snapshot ... parent snapshot does not exist
```
Fix: Disable containerd image store in Docker Desktop → Settings → General → uncheck "Use containerd for pulling and storing images".
Or run:
```powershell
docker builder prune -af
docker system prune -af
docker compose up -d --build
```
Power BI iframe shows blank / spinning logo
Verify the `reportId` and `ctid` in `templates/index.html` match your actual workspace
Ensure `&autoAuth=true` is in the iframe src URL
Make sure your browser is signed into the correct Microsoft account
Open the report URL directly in a new tab first to confirm access
"Tenant 'abcdef12-...' not found" error
Your iframe src has placeholder GUIDs. Replace `reportId` and `ctid` with the real ones from your Power BI workspace embed URL.
---
🌱 Future Work
🔐 App Owns Data embedding — unify Flask + Power BI identity via Service Principal (requires Power BI Premium)
🎥 Live match scoring — currently batch-only, real-time integration with TV feeds
👀 Computer vision — extract additional features from broadcast footage
🌍 Expanded coverage — FIP Tour, A1 Padel, national circuits
🔌 Public API — for partner federations and sponsors
🤝 Federated learning — privacy-preserving model training across federations
---
🌐 SDG Alignment
This project supports the following UN Sustainable Development Goals:
SDG	Contribution
3 — Good Health	Sport participation, healthy lifestyle promotion
5 — Gender Equality	Men's and women's circuit analytics
8 — Decent Work	Sports economy, equipment industry insights
9 — Innovation	AI-powered sport BI
17 — Partnerships	Federation · Sponsor · Player ecosystem
---
👥 Team
This project was developed as a Final Year Project at ESPRIT — École Supérieure Privée d'Ingénierie et de Technologies by 5 engineering students:
<div align="center">
Khalil Bensouissi	Mouhamed Ali	Bechir Zarrouki	Chouikh Rayene	Nidhal Ghanmi
Engineering Student	Engineering Student	Engineering Student	Engineering Student	Engineering Student
</div>
Supervised by
Mrs. Soukeina Touiti Ben Khalifa
Mr. Ridha Berrahal
Mrs. Hela Mejri
---
📚 References
Key methodologies and libraries that made this project possible:
Fernandez, A. — GIMSI Methodology, Éditions d'Organisation, 2013
Kimball, R. & Ross, M. — The Data Warehouse Toolkit, 3rd Edition, Wiley, 2013
Chen, T. & Guestrin, C. — XGBoost: A Scalable Tree Boosting System, ACM SIGKDD, 2016
Taylor, S. & Letham, B. — Forecasting at Scale (Prophet), American Statistician, 2018
Zaharia, M. et al. — MLflow: Accelerating the ML Lifecycle, IEEE Data Engineering Bulletin, 2018
Power BI Row-Level Security Documentation
Apache Airflow Documentation
Prometheus Monitoring
Grafana Documentation
---
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
---
🙏 Acknowledgments
ESPRIT faculty for the GIMSI methodology and rigorous BI/Big Data curriculum
Premier Padel for publishing open tournament data
The open-source community — PostgreSQL, Talend, Flask, MLflow, Docker, Prometheus, Grafana, n8n, and dozens of Python libraries that made this possible
Our families and friends for surviving 6 months of "the pipeline is running"
---
<div align="center">
⭐ If you found this project useful, please consider starring the repo!
Padel Analytics — Data Wins Points 🎾
Built with ❤️ at ESPRIT · Tunis, Tunisia · 2025–2026
</div>
