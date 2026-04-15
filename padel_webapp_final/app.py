"""
=============================================================================
PADEL ANALYTICS — FLASK WEB APPLICATION
=============================================================================
Serves ML models for:
  1. Win Prediction (Classification)
  2. Points Prediction (Regression)
  3. Player Segmentation (Clustering)
  4. Equipment Recommendation
  5. Emerging Talent Detection (Anomaly)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ═══════════════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════

print("[*] Loading models...")
clf_rf = joblib.load(os.path.join(MODEL_DIR, "classifier_random_forest.pkl"))
clf_gb = joblib.load(os.path.join(MODEL_DIR, "classifier_gradient_boosting.pkl"))
clf_xgb = joblib.load(os.path.join(MODEL_DIR, "classifier_xgboost.pkl"))

reg_ridge = joblib.load(os.path.join(MODEL_DIR, "regressor_ridge.pkl"))
reg_lasso = joblib.load(os.path.join(MODEL_DIR, "regressor_lasso.pkl"))
reg_xgb = joblib.load(os.path.join(MODEL_DIR, "regressor_xgboost.pkl"))

kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
gmm = joblib.load(os.path.join(MODEL_DIR, "gmm.pkl"))
iso_forest = joblib.load(os.path.join(MODEL_DIR, "iso_forest.pkl"))

scaler_cluster = joblib.load(os.path.join(MODEL_DIR, "scaler_cluster.pkl"))
scaler_rec = joblib.load(os.path.join(MODEL_DIR, "scaler_recommender.pkl"))
scaler_anomaly = joblib.load(os.path.join(MODEL_DIR, "scaler_anomaly.pkl"))

le_gender = joblib.load(os.path.join(MODEL_DIR, "le_gender.pkl"))
le_country = joblib.load(os.path.join(MODEL_DIR, "le_country.pkl"))
print("[✓] All models loaded")


# ═══════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════

print("[*] Loading data...")
fact_match = pd.read_csv(os.path.join(DATA_DIR, "fact_match.csv"))
fact_player = pd.read_csv(os.path.join(DATA_DIR, "fact_player.csv"))
fact_equip = pd.read_csv(os.path.join(DATA_DIR, "fact_equipement.csv"))
dim_player = pd.read_csv(os.path.join(DATA_DIR, "dim_player.csv"))

# ─── Name fixes (encoding mismatches between CSVs) ──────────────────
_alvaro = fact_player[fact_player['player_name'].str.contains('lvaro Melendez', na=False)].sort_values('points', ascending=False)
_alvaro_name = _alvaro.iloc[0]['player_name'] if len(_alvaro) > 0 else None

NAME_MAP = {
    "Federico Mouri\u00f1o": "Federico Mouri\u00c3\u00b1o",
    "Ignacio Vilari\u00f1o Gestoso": "Ignacio Vilari\u00c3\u00b1o Gestoso",
    "Valentino Libaak": "Valentino Gabriel Libaak",
    "Leandro Roman Augsburguer": "Leandro Augsburger",
    "Maximiliano Sanchez Aguero": "Maximiliano Sanchez",
    "Agustin Gomez Silingo": "Agustin Silingo",
    "Pablo Lij\u00f3": "Pablo Lijo",
    "Agust\u00edn Torre": "Eduardo Agustin Torre",
    "Manuel Casta\u00f1o Salguero": "Manuel Casta\u00c3\u00b1o Salguero",
    "Jes\u00fas Moya Sos": "Jesus Moya Sos",
    "Francisco Miguel Ramirez Navas": "Fran Ramirez Navas",
    "Renzo Gabriel Nu\u00f1ez": "Renzo Gabriel Nu\u00c3\u00b1ez",
    "Diego Arredondo Garc\u00eda": "Diego Arredondo Garcia",
    "Diego Garc\u00eda": "Diego Garcia",
    "Juan Manuel Arga\u00f1aras": "Juan Manuel Arga\u00c3\u00b1aras",
    "Pau Mi\u00f1ano Ortinez": "Pau Mi\u00c3\u00b1ano Ortinez",
    "Nacho Moragues Molt\u00f2": "Nacho Moragues Molt\u00c3\u00b2",
    "Pablo Marqu\u00e9s Gonzalez": "Pablo Marqu\u00c3\u00a9s Gonzalez",
    "Martin Sanchez Pi\u00f1eiro": "Mart\u00c3\u00adn S\u00c3\u00a1nchez Pi\u00c3\u00b1eiro",
    "Pyry Hyrkk\u00f6nen": "Pyry Hyrkk\u00c3\u00b6nen",
    "Sebastian Mu\u00f1oz Campos": "Sebastian Mu\u00c3\u00b1oz Campos",
    "Rayyan Al Jufairi": "Rayan Abdulla Aljufairi",
    "Khalid Saadon al-Kuwari": "Khalid Saadon Alkuwari",
    "Aimar Go\u00f1i Lacabe": "Aimar Go\u00c3\u00b1i Lacabe",
}
if _alvaro_name:
    NAME_MAP["\u00c1lvaro Melendez Amaya"] = _alvaro_name
    NAME_MAP["\u00c1lvaro M\u00e9lendez Amaya"] = _alvaro_name

DROP_PLAYERS = [
    "Abdulaziz Saadon Alkuwari", "Amr Hassan", "Andres Devletian",
    "Carlos Zarhi", "Diego Gonzalez Almedo", "Eduardo Andres Lopez Figuera",
    "Juan Martin Diaz Martinez", "Manuel Yuste Apolinar", "Mohammed Abdulla",
    "Roberto Alonso Rodriguez", "Santiago Frugoni"
]

for col in ['team_a_player1','team_a_player2','team_b_player1','team_b_player2']:
    fact_match[col] = fact_match[col].replace(NAME_MAP)
fact_match['team_a_key'] = fact_match['team_a_player1'] + ' / ' + fact_match['team_a_player2']
fact_match['team_b_key'] = fact_match['team_b_player1'] + ' / ' + fact_match['team_b_player2']
for _, row in fact_match.iterrows():
    if row['winner_team'] not in [row['team_a_key'], row['team_b_key']]:
        a_players = {row['team_a_player1'], row['team_a_player2']}
        for p in a_players:
            if p in str(row['winner_team']) or any(orig in str(row['winner_team']) for orig, new in NAME_MAP.items() if new == p):
                fact_match.at[_, 'winner_team'] = row['team_a_key']
                break
        else:
            fact_match.at[_, 'winner_team'] = row['team_b_key']

all_drop = set(DROP_PLAYERS)
mask = ~(fact_match['team_a_player1'].isin(all_drop) | fact_match['team_a_player2'].isin(all_drop) |
         fact_match['team_b_player1'].isin(all_drop) | fact_match['team_b_player2'].isin(all_drop))
fact_match = fact_match[mask].reset_index(drop=True)

# Parse social media
social_cols = ['instagram_followers', 'youtube_subscribers',
               'tiktok_followers', 'twitter_followers', 'wikipedia_views']
for col in social_cols:
    fact_player[col] = pd.to_numeric(fact_player[col], errors='coerce').fillna(0)

# Round ordering
round_order = {
    'Round of 64': 1, 'Round of 32': 2, 'Round of 16': 3,
    'Quarterfinals': 4, 'Semifinals': 5, 'Final': 6
}
fact_match['match_date'] = pd.to_datetime(fact_match['match_date'], utc=True)

# Build per-player match stats with creative features
def parse_score(s):
    if pd.isna(s): return []
    return [int(x) for x in str(s).split('-') if x.strip().isdigit()]

fact_match['sets_a'] = fact_match['score_team_a'].apply(parse_score)
fact_match['sets_b'] = fact_match['score_team_b'].apply(parse_score)
fact_match['n_sets'] = fact_match.apply(lambda r: max(len(r['sets_a']), len(r['sets_b'])), axis=1)
fact_match['is_3set'] = (fact_match['n_sets'] == 3).astype(int)
fact_match['games_a'] = fact_match['sets_a'].apply(sum)
fact_match['games_b'] = fact_match['sets_b'].apply(sum)

records = []
for _, row in fact_match.iterrows():
    for players, key, gw, gl in [
        ([row['team_a_player1'], row['team_a_player2']], row['team_a_key'], row['games_a'], row['games_b']),
        ([row['team_b_player1'], row['team_b_player2']], row['team_b_key'], row['games_b'], row['games_a'])
    ]:
        won = int(row['winner_team'] == key)
        for p in players:
            records.append({
                'player_name': p, 'tournament': row['tournament_name'],
                'match_date': row['match_date'],
                'year': row['source_year'], 'round': row['round'],
                'team_key': key, 'won': won, 'is_3set': row['is_3set'],
                'games_won': gw, 'games_lost': gl,
                'dominance': (gw - gl) / max(gw + gl, 1)
            })
player_matches = pd.DataFrame(records)

player_agg = player_matches.groupby('player_name').agg(
    total_matches=('won', 'count'), total_wins=('won', 'sum'),
    total_games_won=('games_won', 'sum'), total_games_lost=('games_lost', 'sum'),
    tournaments_played=('tournament', 'nunique'), years_active=('year', 'nunique'),
    avg_round=('round', lambda x: np.mean([round_order.get(r, 0) for r in x])),
    max_round=('round', lambda x: max([round_order.get(r, 0) for r in x])),
    avg_dominance=('dominance', 'mean'),
).reset_index()
player_agg['win_rate'] = player_agg['total_wins'] / player_agg['total_matches']
player_agg['game_diff'] = player_agg['total_games_won'] - player_agg['total_games_lost']

# Clutch rate
clutch = player_matches[player_matches['is_3set']==1].groupby('player_name')['won'].agg(
    clutch_wins='sum', clutch_total='count').reset_index()
clutch['clutch_rate'] = clutch['clutch_wins'] / clutch['clutch_total']
player_agg = player_agg.merge(clutch[['player_name','clutch_rate']], on='player_name', how='left')
player_agg['clutch_rate'] = player_agg['clutch_rate'].fillna(player_agg['win_rate'])

# Late-round win rate
late_wr = player_matches[player_matches['round'].isin(['Semifinals','Final'])].groupby('player_name')['won'].mean().reset_index()
late_wr.columns = ['player_name', 'wr_late']
player_agg = player_agg.merge(late_wr, on='player_name', how='left')
player_agg['wr_late'] = player_agg['wr_late'].fillna(player_agg['win_rate'])

# Partner loyalty
partner_map = {}
for _, row in fact_match.iterrows():
    partner_map.setdefault(row['team_a_player1'], []).append(row['team_a_player2'])
    partner_map.setdefault(row['team_a_player2'], []).append(row['team_a_player1'])
    partner_map.setdefault(row['team_b_player1'], []).append(row['team_b_player2'])
    partner_map.setdefault(row['team_b_player2'], []).append(row['team_b_player1'])
player_agg['unique_partners'] = player_agg['player_name'].map(lambda p: len(set(partner_map.get(p, [1]))))
player_agg['partner_loyalty'] = 1 / player_agg['unique_partners'].clip(1)

# Ranking trajectory
multi_year = fact_player.groupby('player_name').filter(lambda x: x['year'].nunique() >= 2)
if len(multi_year) > 0:
    traj = multi_year.sort_values(['player_name','year']).groupby('player_name').agg(
        first_pos=('position','first'), last_pos=('position','last'), n_years=('year','nunique')
    ).reset_index()
    traj['trajectory_slope'] = (traj['first_pos'] - traj['last_pos']) / traj['n_years']
    player_agg = player_agg.merge(traj[['player_name','trajectory_slope']], on='player_name', how='left')
player_agg['trajectory_slope'] = player_agg.get('trajectory_slope', pd.Series(0, index=player_agg.index)).fillna(0)

# Team chemistry + Elo
team_records = []
elo_dict = {}
K_elo = 32
for _, row in fact_match.sort_values('match_date' if 'match_date' in fact_match.columns else 'fact_match_pk').iterrows():
    ta, tb = row['team_a_key'], row['team_b_key']
    for k in [ta, tb]:
        if k not in elo_dict: elo_dict[k] = 1500
    ea = 1 / (1 + 10**((elo_dict[tb] - elo_dict[ta]) / 400))
    a_won = int(row['winner_team'] == ta)
    elo_dict[ta] += K_elo * (a_won - ea)
    elo_dict[tb] += K_elo * ((1-a_won) - (1-ea))
    for key in [ta, tb]:
        team_records.append({'team_key': key, 'won': int(row['winner_team']==key), 'is_3set': row['is_3set']})
team_df = pd.DataFrame(team_records)
team_agg = team_df.groupby('team_key').agg(
    team_matches=('won','count'), team_wins=('won','sum')).reset_index()
team_agg['team_win_rate'] = team_agg['team_wins'] / team_agg['team_matches']
team_agg['team_elo'] = team_agg['team_key'].map(elo_dict).fillna(1500)
# Team clutch
t3 = team_df[team_df['is_3set']==1].groupby('team_key')['won'].agg(t3w='sum',t3m='count').reset_index()
t3['team_clutch'] = t3['t3w'] / t3['t3m']
team_agg = team_agg.merge(t3[['team_key','team_clutch']], on='team_key', how='left')
team_agg['team_clutch'] = team_agg['team_clutch'].fillna(0.5)

# Build full player dataset
latest_ranking = fact_player.sort_values('year', ascending=False).drop_duplicates('player_name')
df_all = latest_ranking[['player_name', 'country', 'points', 'position'] + social_cols].copy()
df_all = df_all.merge(dim_player[['player_name', 'gender']].drop_duplicates('player_name'), on='player_name', how='left')
merge_cols = [c for c in player_agg.columns if c != 'player_name']
df_all = df_all.merge(player_agg, on='player_name', how='left')
for c in merge_cols:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna(0)

# Encoding
all_genders = df_all['gender'].fillna('unknown').unique()
all_countries = df_all['country'].fillna('unknown').unique()
df_all['gender_enc'] = le_gender.transform(df_all['gender'].fillna('unknown'))
df_all['country_enc'] = le_country.transform(df_all['country'].fillna('unknown'))
df_all['total_social'] = df_all[social_cols].sum(axis=1)
df_all['log_social'] = np.log1p(df_all['total_social'])
df_all['log_points'] = np.log1p(df_all['points'].fillna(0))
df_all['points'] = df_all['points'].fillna(df_all['points'].median())
df_all['position'] = df_all['position'].fillna(df_all['position'].median())

# Clustering — assign segments
cluster_features = ['points', 'position', 'total_matches', 'total_wins',
                    'win_rate', 'game_diff', 'log_social', 'log_points',
                    'clutch_rate', 'trajectory_slope']
X_cl = df_all[cluster_features].fillna(0)
X_cl_scaled = scaler_cluster.transform(X_cl)

# All 3 clustering models
df_all['cluster_kmeans'] = kmeans.predict(X_cl_scaled)
df_all['cluster_gmm'] = gmm.predict(X_cl_scaled)
from sklearn.cluster import AgglomerativeClustering
hier_model = AgglomerativeClustering(n_clusters=4)
df_all['cluster_hier'] = hier_model.fit_predict(X_cl_scaled)

# Label K-Means segments (primary)
df_all['cluster'] = df_all['cluster_kmeans']
profile = df_all.groupby('cluster')[cluster_features].mean()
cluster_rank = profile['points'].sort_values(ascending=False).index.tolist()
labels = ['Stars', 'Contenders', 'Regulars', 'Newcomers']
label_map = {}
for i, c in enumerate(cluster_rank):
    label_map[c] = labels[i] if i < len(labels) else f'Group_{i}'
df_all['segment'] = df_all['cluster'].map(label_map)

# Label GMM segments
profile_gmm = df_all.groupby('cluster_gmm')[cluster_features].mean()
gmm_rank = profile_gmm['points'].sort_values(ascending=False).index.tolist()
gmm_map = {c: labels[i] if i < len(labels) else f'Group_{i}' for i, c in enumerate(gmm_rank)}
df_all['segment_gmm'] = df_all['cluster_gmm'].map(gmm_map)

# Label Hierarchical segments
profile_hier = df_all.groupby('cluster_hier')[cluster_features].mean()
hier_rank = profile_hier['points'].sort_values(ascending=False).index.tolist()
hier_map = {c: labels[i] if i < len(labels) else f'Group_{i}' for i, c in enumerate(hier_rank)}
df_all['segment_hier'] = df_all['cluster_hier'].map(hier_map)

# Anomaly detection — both Isolation Forest AND LOF
anomaly_features = ['win_rate', 'position', 'total_matches', 'game_diff',
                    'clutch_rate', 'trajectory_slope', 'avg_dominance']
df_match_only = df_all[df_all['total_matches'] > 0].copy()
if len(df_match_only) > 0:
    df_match_only['expected_win_rate'] = 1 - (df_match_only['position'] / df_match_only['position'].max())
    df_match_only['overperformance'] = df_match_only['win_rate'] - df_match_only['expected_win_rate']
    df_match_only['points_per_match'] = df_match_only['points'] / df_match_only['total_matches'].clip(1)

    # Pre-compute both anomaly models
    anom_feat_full = ['win_rate', 'position', 'overperformance', 'points_per_match',
                      'total_matches', 'game_diff', 'clutch_rate', 'trajectory_slope', 'avg_dominance']
    X_anom = df_match_only[anom_feat_full].fillna(0)
    X_anom_scaled = scaler_anomaly.transform(X_anom)
    df_match_only['anomaly_iso'] = iso_forest.predict(X_anom_scaled)
    df_match_only['score_iso'] = iso_forest.score_samples(X_anom_scaled)

    from sklearn.neighbors import LocalOutlierFactor
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df_match_only['anomaly_lof'] = lof_model.fit_predict(X_anom_scaled)

# Time Series — prepare QUARTERLY data with rich features for all players
ts_quarterly = None
ts_eligible = []
try:
    player_matches['match_date'] = pd.to_datetime(player_matches['match_date'], utc=True)
    player_matches['quarter'] = player_matches['match_date'].dt.to_period('Q')
    player_matches['round_num'] = player_matches['round'].map(round_order).fillna(1)
    dm_tourn = pd.read_csv(os.path.join(DATA_DIR, "dim_tournament.csv"))
    cat_map = dict(zip(dm_tourn['tournament_name'], dm_tourn['category']))
    player_matches['category'] = player_matches['tournament'].map(cat_map).fillna('P2')
    cat_weight = {'Major': 3, 'P1': 2, 'P2': 1}
    player_matches['cat_w'] = player_matches['category'].map(cat_weight).fillna(1)
    player_matches['weighted_win'] = player_matches['won'] * player_matches['cat_w']
    player_matches['is_late_round'] = (player_matches['round_num'] >= 5).astype(int)
    ts_quarterly = player_matches.groupby(['player_name', 'quarter']).agg(
        win_rate=('won', 'mean'), matches=('won', 'count'),
        avg_round=('round_num', 'mean'), max_round=('round_num', 'max'),
        weighted_wins=('weighted_win', 'sum'), total_cat_w=('cat_w', 'sum'),
        avg_dominance=('dominance', 'mean'), late_round_rate=('is_late_round', 'mean'),
    ).reset_index()
    ts_quarterly['weighted_wr'] = ts_quarterly['weighted_wins'] / ts_quarterly['total_cat_w'].clip(1)
    q_counts = ts_quarterly.groupby('player_name')['quarter'].count()
    ts_eligible = sorted(q_counts[q_counts >= 6].index.tolist())
    print(f"[✓] Time series (quarterly): {len(ts_eligible)} players with 6+ quarters")
except Exception as e:
    print(f"[!] Time series skipped: {e}")


def compute_forecast(player_name, n_future=2):
    """Compute ARIMA + Prophet quarterly forecast for any player."""
    if ts_quarterly is None:
        return None
    pq = ts_quarterly[ts_quarterly['player_name'] == player_name].copy()
    if len(pq) < 4:
        return {'error': f'{player_name} has only {len(pq)} quarters (need 4+)'}
    pq['date'] = pq['quarter'].dt.to_timestamp()
    pq = pq.sort_values('date')
    target = pq.set_index('date')['win_rate']
    n_test = max(2, len(target) // 4)
    train_ts, test_ts = target.iloc[:-n_test], target.iloc[-n_test:]

    from pandas.tseries.offsets import QuarterBegin
    future_dates = pd.date_range(start=target.index[-1] + QuarterBegin(1), periods=n_future, freq='QS')
    future_labels = [f"{d.year} Q{(d.month-1)//3+1}" for d in future_dates]

    history = []
    for _, row in pq.iterrows():
        history.append({'quarter': str(row['quarter']), 'win_rate': round(float(row['win_rate'])*100,1),
            'matches': int(row['matches']), 'max_round': int(row['max_round']),
            'weighted_wr': round(float(row['weighted_wr'])*100,1),
            'avg_dominance': round(float(row['avg_dominance']),3),
            'late_round_pct': round(float(row['late_round_rate'])*100,1)})

    result = {'player': player_name, 'n_quarters': len(target), 'n_test': n_test,
              'n_future': n_future, 'last_quarter': str(pq['quarter'].iloc[-1]),
              'history': history, 'test_start': str(pq['quarter'].iloc[-n_test]), 'models': {}}

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import warnings as _w; _w.filterwarnings('ignore')

    actual_test = {str(pq['quarter'].iloc[-n_test+i]): round(float(test_ts.iloc[i])*100,1) for i in range(n_test)}

    # ─── ARIMA ───
    try:
        from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
        best_aic, best_ord, best_fit = np.inf, (0,0,0), None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        fit = ARIMAModel(train_ts.values, order=(p,d,q)).fit()
                        if fit.aic < best_aic: best_aic, best_ord, best_fit = fit.aic, (p,d,q), fit
                    except: pass
        eval_fc = best_fit.forecast(steps=n_test) if best_fit else np.full(n_test, train_ts.mean())
        eval_preds = {str(pq['quarter'].iloc[-n_test+i]): round(float(np.clip(eval_fc[i],0,1))*100,1) for i in range(n_test)}

        best_aic2, best_fit2 = np.inf, None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        fit = ARIMAModel(target.values, order=(p,d,q)).fit()
                        if fit.aic < best_aic2: best_aic2, best_fit2 = fit.aic, fit
                    except: pass
        fut_fc = best_fit2.forecast(steps=n_future) if best_fit2 else np.full(n_future, target.mean())
        fut_fc = np.clip(fut_fc, 0, 1)
        future_preds = {fl: round(float(fut_fc[i])*100,1) for i, fl in enumerate(future_labels)}

        mae = round(float(mean_absolute_error(test_ts, np.clip(eval_fc,0,1))), 4)
        rmse = round(float(np.sqrt(mean_squared_error(test_ts, np.clip(eval_fc,0,1)))), 4)
        result['models']['ARIMA'] = {'order': str(best_ord), 'eval_predictions': eval_preds,
            'actual_test': actual_test, 'future_predictions': future_preds,
            'mae': mae, 'rmse': rmse, 'mae_pct': round(mae*100,1)}
    except Exception as e:
        result['models']['ARIMA'] = {'error': str(e)}

    # ─── Prophet ───
    try:
        from prophet import Prophet as ProphetModel
        prop_df = pd.DataFrame({'ds': target.index, 'y': target.values})

        m_eval = ProphetModel(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m_eval.fit(prop_df.iloc[:-n_test])
        fc_ev = m_eval.predict(m_eval.make_future_dataframe(periods=n_test, freq='QS'))
        ev_pred = np.clip(fc_ev.tail(n_test)['yhat'].values, 0, 1)
        eval_preds_p = {str(pq['quarter'].iloc[-n_test+i]): round(float(ev_pred[i])*100,1) for i in range(n_test)}

        m_full = ProphetModel(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m_full.fit(prop_df)
        fc_fu = m_full.predict(m_full.make_future_dataframe(periods=n_future, freq='QS'))
        fu_pred = np.clip(fc_fu.tail(n_future)['yhat'].values, 0, 1)
        future_preds_p = {fl: round(float(fu_pred[i])*100,1) for i, fl in enumerate(future_labels)}

        mae = round(float(mean_absolute_error(test_ts, ev_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(test_ts, ev_pred))), 4)
        result['models']['Prophet'] = {'eval_predictions': eval_preds_p, 'actual_test': actual_test,
            'future_predictions': future_preds_p, 'mae': mae, 'rmse': rmse, 'mae_pct': round(mae*100,1)}
    except Exception as e:
        result['models']['Prophet'] = {'error': str(e)}

    return result

# Pro profiles for recommendation
equip_players = fact_equip['player_name'].unique()
pro_profiles = df_all[df_all['player_name'].isin(equip_players)].copy()
pro_equip = fact_equip.groupby('player_name').agg(
    primary_brand=('brand', lambda x: x.mode()[0]),
    n_equipment=('pk_equipement', 'count'),
    avg_price=('price', 'mean'),
).reset_index()
pro_profiles = pro_profiles.merge(pro_equip, on='player_name', how='left')

rec_features = ['points', 'position', 'total_matches', 'total_wins',
                'win_rate', 'game_diff', 'gender_enc', 'log_social', 'clutch_rate']

# Player lists for dropdowns — only players who played matches
player_list = sorted(df_all[df_all['total_matches'] > 0]['player_name'].tolist())
country_list = sorted(df_all['country'].dropna().unique().tolist())
gender_list = ['homme', 'femme']

# Display stats — only players with matches
df_display = df_all[df_all['total_matches'] > 0]
segment_stats = df_display['segment'].value_counts().to_dict()
total_players = len(df_all)
total_players_active = len(df_display)
total_matches = len(fact_match)
total_tournaments = fact_match['tournament_name'].nunique()

print(f"[✓] Data ready: {total_players} players, {total_matches} matches")


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    top_players = df_display.nlargest(10, 'points')[
        ['player_name', 'country', 'points', 'position', 'segment']
    ].to_dict('records')
    return render_template('index.html',
                           total_players=total_players,
                           total_matches=total_matches,
                           total_tournaments=total_tournaments,
                           segment_stats=segment_stats,
                           top_players=top_players)


# ─── 1. Classification: Win Prediction ──────────────────────────────
@app.route('/predict')
def predict_page():
    return render_template('predict.html', players=player_list)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    player_name = data.get('player_name')
    round_name = data.get('round', 'Round of 32')

    player = df_all[df_all['player_name'] == player_name]
    if len(player) == 0:
        return jsonify({'error': 'Player not found'}), 404

    p = player.iloc[0]

    # v2: classifier needs 27 features including opponent stats
    # For single player, use average opponent (median stats)
    avg_opp_wr = df_all[df_all['total_matches']>0]['win_rate'].median()
    avg_opp_pts = df_all['points'].median()
    avg_opp_pos = df_all['position'].median()
    avg_opp_clutch = df_all[df_all['total_matches']>0]['clutch_rate'].median()
    avg_opp_dom = df_all[df_all['total_matches']>0]['avg_dominance'].median()

    my_wr = float(p.get('win_rate', 0))
    my_pts = float(p.get('points', 0))
    my_pos = float(p.get('position', 999))
    my_clutch = float(p.get('clutch_rate', 0.5))
    my_dom = float(p.get('avg_dominance', 0))
    my_late = float(p.get('wr_late', my_wr))
    my_loyalty = float(p.get('partner_loyalty', 0.5))
    my_traj = float(p.get('trajectory_slope', 0))

    features = pd.DataFrame([{
        'my_win_rate': my_wr, 'my_points': my_pts, 'my_position': my_pos,
        'my_game_diff': float(p.get('game_diff', 0)),
        'my_clutch': my_clutch, 'my_dominance': my_dom,
        'my_late_wr': my_late, 'my_loyalty': my_loyalty, 'my_trajectory': my_traj,
        'opp_win_rate': avg_opp_wr, 'opp_points': avg_opp_pts, 'opp_position': avg_opp_pos,
        'opp_clutch': avg_opp_clutch, 'opp_dominance': avg_opp_dom,
        'diff_win_rate': my_wr - avg_opp_wr, 'diff_points': my_pts - avg_opp_pts,
        'diff_position': avg_opp_pos - my_pos, 'diff_clutch': my_clutch - avg_opp_clutch,
        'my_elo': 1500, 'opp_elo': 1500, 'elo_diff': 0,
        'my_team_chemistry': my_wr, 'opp_team_chemistry': 0.5,
        'chemistry_diff': my_wr - 0.5,
        'my_team_clutch': my_clutch, 'opp_team_clutch': 0.5,
        'round_num': round_order.get(round_name, 2)
    }])

    results = {}
    for name, model in [('Random Forest', clf_rf),
                        ('Gradient Boosting', clf_gb),
                        ('XGBoost', clf_xgb)]:
        proba = float(model.predict_proba(features)[0][1])
        results[name] = {
            'probability': round(proba, 4),
            'prediction': 'WIN' if proba > 0.5 else 'LOSS',
            'confidence': round(abs(proba - 0.5) * 200, 1)
        }

    return jsonify({
        'player': player_name,
        'round': round_name,
        'country': str(p['country']),
        'position': int(p['position']),
        'points': int(p['points']),
        'win_rate': round(float(p['win_rate']) * 100, 1),
        'segment': str(p.get('segment', 'N/A')),
        'models': results
    })


# ─── 1b. Team vs Team Prediction ────────────────────────────────────
@app.route('/api/predict-team', methods=['POST'])
def api_predict_team():
    data = request.json
    team_a = data.get('team_a', [])
    team_b = data.get('team_b', [])
    round_name = data.get('round', 'Round of 32')

    if len(team_a) != 2 or len(team_b) != 2:
        return jsonify({'error': 'Each team must have exactly 2 players'}), 400

    def get_p(name):
        row = df_all[df_all['player_name'] == name]
        if len(row) == 0: return None, None
        r = row.iloc[0]
        stats = {k: float(r.get(k, 0)) for k in ['win_rate','points','position','game_diff',
                 'clutch_rate','avg_dominance','wr_late','partner_loyalty','trajectory_slope']}
        info = {'name': name, 'country': str(r['country']), 'position': int(r['position']),
                'points': int(r['points']), 'win_rate': round(float(r['win_rate'])*100, 1),
                'segment': str(r.get('segment', 'N/A'))}
        return stats, info

    a1s, a1i = get_p(team_a[0]); a2s, a2i = get_p(team_a[1])
    b1s, b1i = get_p(team_b[0]); b2s, b2i = get_p(team_b[1])
    for s, i, n in [(a1s,a1i,team_a[0]),(a2s,a2i,team_a[1]),(b1s,b1i,team_b[0]),(b2s,b2i,team_b[1])]:
        if s is None: return jsonify({'error': f'Player not found: {n}'}), 404

    # Team keys for Elo/chemistry lookup
    ta_key = f"{team_a[0]} / {team_a[1]}"
    tb_key = f"{team_b[0]} / {team_b[1]}"
    ta_rev = f"{team_a[1]} / {team_a[0]}"
    tb_rev = f"{team_b[1]} / {team_b[0]}"

    ta_row = team_agg[(team_agg['team_key']==ta_key)|(team_agg['team_key']==ta_rev)]
    tb_row = team_agg[(team_agg['team_key']==tb_key)|(team_agg['team_key']==tb_rev)]
    ta_elo = float(ta_row['team_elo'].values[0]) if len(ta_row)>0 else 1500
    tb_elo = float(tb_row['team_elo'].values[0]) if len(tb_row)>0 else 1500
    ta_chem = float(ta_row['team_win_rate'].values[0]) if len(ta_row)>0 else 0.5
    tb_chem = float(tb_row['team_win_rate'].values[0]) if len(tb_row)>0 else 0.5
    ta_clutch_t = float(ta_row['team_clutch'].values[0]) if len(ta_row)>0 else 0.5
    tb_clutch_t = float(tb_row['team_clutch'].values[0]) if len(tb_row)>0 else 0.5

    def avg(d1, d2, key): return (d1[key] + d2[key]) / 2

    def build_features(p1, p2, o1, o2, my_elo, opp_elo, my_chem, opp_chem, my_tc, opp_tc):
        return pd.DataFrame([{
            'my_win_rate': avg(p1,p2,'win_rate'), 'my_points': avg(p1,p2,'points'),
            'my_position': avg(p1,p2,'position'), 'my_game_diff': avg(p1,p2,'game_diff'),
            'my_clutch': avg(p1,p2,'clutch_rate'), 'my_dominance': avg(p1,p2,'avg_dominance'),
            'my_late_wr': avg(p1,p2,'wr_late'), 'my_loyalty': avg(p1,p2,'partner_loyalty'),
            'my_trajectory': avg(p1,p2,'trajectory_slope'),
            'opp_win_rate': avg(o1,o2,'win_rate'), 'opp_points': avg(o1,o2,'points'),
            'opp_position': avg(o1,o2,'position'),
            'opp_clutch': avg(o1,o2,'clutch_rate'), 'opp_dominance': avg(o1,o2,'avg_dominance'),
            'diff_win_rate': avg(p1,p2,'win_rate') - avg(o1,o2,'win_rate'),
            'diff_points': avg(p1,p2,'points') - avg(o1,o2,'points'),
            'diff_position': avg(o1,o2,'position') - avg(p1,p2,'position'),
            'diff_clutch': avg(p1,p2,'clutch_rate') - avg(o1,o2,'clutch_rate'),
            'my_elo': my_elo, 'opp_elo': opp_elo, 'elo_diff': my_elo - opp_elo,
            'my_team_chemistry': my_chem, 'opp_team_chemistry': opp_chem,
            'chemistry_diff': my_chem - opp_chem,
            'my_team_clutch': my_tc, 'opp_team_clutch': opp_tc,
            'round_num': round_order.get(round_name, 2)
        }])

    feat_a = build_features(a1s, a2s, b1s, b2s, ta_elo, tb_elo, ta_chem, tb_chem, ta_clutch_t, tb_clutch_t)
    feat_b = build_features(b1s, b2s, a1s, a2s, tb_elo, ta_elo, tb_chem, ta_chem, tb_clutch_t, ta_clutch_t)

    # Predict for both teams
    results = {}
    for model_name, model in [('Random Forest', clf_rf),
                               ('Gradient Boosting', clf_gb),
                               ('XGBoost', clf_xgb)]:
        proba_a = float(model.predict_proba(feat_a)[0][1])
        proba_b = float(model.predict_proba(feat_b)[0][1])
        total = proba_a + proba_b
        norm_a = proba_a / total if total > 0 else 0.5
        norm_b = proba_b / total if total > 0 else 0.5
        results[model_name] = {
            'team_a_prob': round(norm_a, 4),
            'team_b_prob': round(norm_b, 4),
            'winner': 'Team A' if norm_a > norm_b else 'Team B',
            'margin': round(abs(norm_a - norm_b) * 100, 1)
        }

    # Head-to-head history
    h2h_matches = []
    for _, row in fact_match.iterrows():
        a_set = {row['team_a_player1'], row['team_a_player2']}
        b_set = {row['team_b_player1'], row['team_b_player2']}
        ta_set = set(team_a)
        tb_set = set(team_b)
        if (a_set == ta_set and b_set == tb_set) or (a_set == tb_set and b_set == ta_set):
            h2h_matches.append({
                'tournament': row['tournament_name'],
                'round': row['round'],
                'winner': str(row['winner_team']),
                'score_a': str(row.get('score_team_a', '')),
                'score_b': str(row.get('score_team_b', ''))
            })

    return jsonify({
        'round': round_name,
        'team_a': {'players': [a1i, a2i]},
        'team_b': {'players': [b1i, b2i]},
        'models': results,
        'head_to_head': h2h_matches,
        'h2h_count': len(h2h_matches)
    })


# ─── 2. Regression: Points Prediction ───────────────────────────────
@app.route('/points')
def points_page():
    return render_template('points.html', players=player_list)


@app.route('/api/points', methods=['POST'])
def api_points():
    data = request.json
    player_name = data.get('player_name')

    player = df_all[df_all['player_name'] == player_name]
    if len(player) == 0:
        return jsonify({'error': 'Player not found'}), 404

    p = player.iloc[0]
    reg_feat = ['total_matches', 'total_wins', 'win_rate', 'game_diff',
                'tournaments_played', 'years_active', 'avg_round', 'max_round',
                'clutch_rate', 'avg_dominance', 'wr_late', 'partner_loyalty',
                'trajectory_slope', 'gender_enc', 'log_social']
    features = pd.DataFrame([[p.get(f, 0) for f in reg_feat]], columns=reg_feat)
    features['wr_x_matches'] = features['win_rate'] * features['total_matches']
    features['wins_per_tourn'] = features['total_wins'] / max(features['tournaments_played'].iloc[0], 1)
    features['gdiff_per_match'] = features['game_diff'] / max(features['total_matches'].iloc[0], 1)
    features = features.fillna(0)

    results = {}
    for name, model in [('Ridge', reg_ridge), ('Lasso', reg_lasso), ('XGBoost', reg_xgb)]:
        pred_log = float(model.predict(features)[0])
        pred = max(float(np.expm1(pred_log)), 0)
        results[name] = round(pred, 0)

    return jsonify({
        'player': player_name,
        'actual_points': int(p['points']),
        'position': int(p['position']),
        'total_matches': int(p['total_matches']),
        'win_rate': round(float(p['win_rate']) * 100, 1),
        'predictions': results
    })


# ─── 3. Clustering: Player Segmentation ─────────────────────────────
@app.route('/segments')
def segments_page():
    segments = {}
    for seg_name in ['Stars', 'Contenders', 'Regulars', 'Newcomers']:
        seg_df = df_display[df_display['segment'] == seg_name]
        segments[seg_name] = {
            'count': len(seg_df),
            'avg_points': round(seg_df['points'].mean(), 0) if len(seg_df) > 0 else 0,
            'avg_position': round(seg_df['position'].mean(), 0) if len(seg_df) > 0 else 0,
            'avg_win_rate': round(seg_df['win_rate'].mean() * 100, 1) if len(seg_df) > 0 else 0,
            'top_players': seg_df.nlargest(5, 'points')[
                ['player_name', 'country', 'points', 'position']
            ].to_dict('records')
        }
    return render_template('segments.html', segments=segments, segment_stats=segment_stats)


@app.route('/api/segment-player', methods=['POST'])
def api_segment_player():
    data = request.json
    player_name = data.get('player_name')

    player = df_all[df_all['player_name'] == player_name]
    if len(player) == 0:
        return jsonify({'error': 'Player not found'}), 404

    p = player.iloc[0]
    return jsonify({
        'player': player_name,
        'models': {
            'K-Means': str(p.get('segment', 'N/A')),
            'GMM': str(p.get('segment_gmm', 'N/A')),
            'Hierarchical': str(p.get('segment_hier', 'N/A'))
        },
        'segment': str(p['segment']),
        'points': int(p['points']),
        'position': int(p['position']),
        'win_rate': round(float(p['win_rate']) * 100, 1),
        'country': str(p['country'])
    })


# ─── 4. Recommendation: Equipment ───────────────────────────────────
@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html', players=player_list)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    player_name = data.get('player_name')
    equip_type = data.get('type', None)

    player = df_all[df_all['player_name'] == player_name]
    if len(player) == 0:
        return jsonify({'error': 'Player not found'}), 404

    p = player.iloc[0]

    # Build player vector
    player_vector = scaler_rec.transform(
        player[rec_features].fillna(0).values
    )
    pro_vectors = scaler_rec.transform(
        pro_profiles[rec_features].fillna(0).values
    )

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(player_vector, pro_vectors)[0]

    pro_ranking = pd.DataFrame({
        'pro_name': pro_profiles['player_name'].values,
        'similarity': similarities,
        'segment': pro_profiles['segment'].values,
        'brand': pro_profiles['primary_brand'].values
    }).sort_values('similarity', ascending=False)

    top_pros = pro_ranking.head(3)['pro_name'].values
    recs = fact_equip[fact_equip['player_name'].isin(top_pros)].copy()
    if equip_type and equip_type != 'all':
        recs = recs[recs['type_produit'] == equip_type]

    recs = recs.merge(
        pro_ranking[['pro_name', 'similarity']].rename(columns={'pro_name': 'player_name'}),
        on='player_name', how='left'
    )
    recs = recs.sort_values('similarity', ascending=False).drop_duplicates('product_name')

    return jsonify({
        'player': player_name,
        'segment': str(p['segment']),
        'similar_pros': [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in r.items()} for r in pro_ranking.head(3).to_dict('records')],
        'recommendations': [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                             for k, v in r.items()} for r in recs[['product_name', 'brand', 'type_produit', 'price', 'player_name']].head(10).to_dict('records')],
        'equip_types': sorted(fact_equip['type_produit'].unique().tolist())
    })


# ─── 5. Anomaly: Emerging Talent ────────────────────────────────────
@app.route('/talent')
def talent_page():
    if len(df_match_only) == 0:
        return render_template('talent.html', talents=[], total_anomalies_iso=0, total_anomalies_lof=0)

    emerging_iso = df_match_only[
        (df_match_only['anomaly_iso'] == -1) &
        (df_match_only['overperformance'] > 0) &
        (df_match_only['total_matches'] >= 3)
    ].sort_values('overperformance', ascending=False)

    emerging_lof = df_match_only[
        (df_match_only['anomaly_lof'] == -1) &
        (df_match_only['overperformance'] > 0) &
        (df_match_only['total_matches'] >= 3)
    ].sort_values('overperformance', ascending=False)

    consensus = df_match_only[
        (df_match_only['anomaly_iso'] == -1) &
        (df_match_only['anomaly_lof'] == -1) &
        (df_match_only['overperformance'] > 0) &
        (df_match_only['total_matches'] >= 3)
    ].sort_values('overperformance', ascending=False)

    display_cols = ['player_name', 'country', 'position', 'points',
                    'win_rate', 'total_matches', 'overperformance']

    return render_template('talent.html',
        talents_iso=emerging_iso[display_cols].head(15).to_dict('records'),
        talents_lof=emerging_lof[display_cols].head(15).to_dict('records'),
        talents_consensus=consensus[display_cols].head(10).to_dict('records'),
        total_anomalies_iso=(df_match_only['anomaly_iso'] == -1).sum(),
        total_anomalies_lof=(df_match_only['anomaly_lof'] == -1).sum())


@app.route('/api/check-talent', methods=['POST'])
def api_check_talent():
    data = request.json
    player_name = data.get('player_name')

    player = df_match_only[df_match_only['player_name'] == player_name]
    if len(player) == 0:
        return jsonify({'error': 'Player not found or has no match data'}), 404

    p = player.iloc[0]
    return jsonify({
        'player': player_name,
        'models': {
            'Isolation Forest': {'is_anomaly': bool(p.get('anomaly_iso') == -1),
                                  'score': round(float(p.get('score_iso', 0)), 4)},
            'LOF': {'is_anomaly': bool(p.get('anomaly_lof') == -1)}
        },
        'is_emerging_talent': bool(p.get('anomaly_iso') == -1 and p.get('overperformance', 0) > 0),
        'consensus': bool(p.get('anomaly_iso') == -1 and p.get('anomaly_lof') == -1 and p.get('overperformance', 0) > 0),
        'overperformance': round(float(p.get('overperformance', 0)), 4),
        'win_rate': round(float(p['win_rate']) * 100, 1),
        'position': int(p['position']),
        'total_matches': int(p['total_matches'])
    })


# ─── 6. Time Series Forecast ─────────────────────────────────────────
@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html', eligible_players=ts_eligible)


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    data = request.json
    player_name = data.get('player_name')
    if not player_name:
        return jsonify({'error': 'No player specified'}), 400
    result = compute_forecast(player_name)
    if result is None:
        return jsonify({'error': 'Time series not available'}), 404
    return jsonify(result)


# ─── Search API ─────────────────────────────────────────────────────
@app.route('/api/search-players')
def search_players():
    q = request.args.get('q', '').lower()
    if len(q) < 2:
        return jsonify([])
    matches = [p for p in player_list if q in p.lower()][:15]
    return jsonify(matches)


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
