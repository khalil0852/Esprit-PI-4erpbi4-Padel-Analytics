# 🎾 Padel Analytics — Premier Padel Dashboard

> Business Intelligence dashboard analyzing Premier Padel data across seasons 2023–2025.  
> Built with Power BI, Python (Machine Learning), and SQL Server.

![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Talend](https://img.shields.io/badge/Talend-FF6D70?style=for-the-badge&logo=talend&logoColor=white)

---

## 📖 About

This project provides a comprehensive analytics platform for Premier Padel stakeholders — players, sponsors, and the federation. It covers match performance, brand analysis, tournament popularity, and player growth insights across three seasons.

---

## 📊 Dashboard Pages

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
- Player loyalty index (Fidélité des Joueurs)
- Emerging talents ranking
- Padel growth rate (14.30%)

### 4. Home Page
- Role-based navigation (Players, Sponsors, Federation)
- Custom padel-themed UI design

---

## 🤖 Machine Learning

### K-Means Clustering — Player Segmentation

Players are grouped into 3 performance levels using unsupervised machine learning:

| Cluster | Label | Description |
|---------|-------|-------------|
| 🟢 | Elite | High match count + high win rate |
| 🔵 | Confirmé | Regular players, average performance |
| 🟠 | En progression | Developing players |

**Features used:**
- Matches Played
- Player Win Rate
- Points

**Tech:** Python (scikit-learn, pandas, matplotlib) integrated as a Python Visual in Power BI.

---

## 🏗️ Architecture

```
📁 padel-analytics/
├── 📊 official_dashboard.pbix      # Main Power BI dashboard
├── 📄 DimSecurity.csv              # RLS security table
├── 🐍 clustering_script.py         # K-Means clustering script
├── 🖼️ backgrounds/
│   ├── padel_players_bg.png        # Players page background
│   ├── padel_sponsors_bg.png       # Sponsors page background
│   ├── padel_federation_bg.png     # Federation page background
│   └── padel_accueil_bg.png        # Home page background
├── 📖 README.md
└── 📄 .gitignore
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

### Page Navigation
Bookmark-based navigation with role buttons on the home page. Each role sees only their designated dashboard pages.

---

## 🔄 Data Pipeline

- **Source:** Web scraping & APIs (Premier Padel data)
- **ETL:** Talend Open Studio (Extract, Transform, Load)
- **Data Warehouse:** PostgreSQL (local)
- **Visualization:** Power BI Desktop
- **Refresh:** Automated via Power Automate Desktop + Windows Task Scheduler (every hour)
- **Model:** Star schema with dimension and fact tables

```
                    ┌─────────────┐
                    │   Sources   │
                    │ (Web / API) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Talend    │
                    │  (ETL)      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ PostgreSQL  │
                    │  (DWH)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        dim_player    dim_brand    dim_date
              │            │            │
              └────────┬───┘────────────┘
                       ▼
              fact_match
              fact_tournament_match
              fact_player
                       │
                       ▼
              ┌────────────────┐
              │   Power BI     │
              │  (Dashboard)   │
              └────────────────┘
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Power BI Desktop
- PostgreSQL
- Talend Open Studio
- Python 3.x with packages:
  ```
  pip install pandas scikit-learn matplotlib
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/khalil0852/Esprit-PI-4erpbi4-Padel-Analytics.git
   ```
2. Import Talend jobs to load data into PostgreSQL
3. Open `official_dashboard.pbix` in Power BI Desktop
4. Update the PostgreSQL connection in **Home → Transform Data → Data Source Settings**
5. Configure Python path in **File → Options → Python Scripting**
6. Click **Refresh** to load the latest data

---

## 📈 Key KPIs

| KPI | Value |
|-----|-------|
| Total Matches | 1,584 |
| Total Participants | 425 |
| Total Tournaments | 47 |
| Padel Growth Rate | 14.30% |
| Total Nationalities | 110 |
| Sponsored Players | 12 |
| Average Equipment Price | 111€ |

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Talend Open Studio | ETL — Extract, Transform, Load |
| PostgreSQL | Data Warehouse |
| Power BI Desktop | Dashboard & visualizations |
| DAX | Measures, KPIs, RLS |
| Python | ML clustering (K-Means) |
| Power Automate Desktop | Automated refresh |
| Git / GitHub | Version control |

---

## 👤 Author

**Khalil Bensouissi**  
ESPRIT — School of Engineering  
📧 khalil.bensouissi@esprit.tn

---

## 📝 License

This project is developed as part of an academic project at ESPRIT.
