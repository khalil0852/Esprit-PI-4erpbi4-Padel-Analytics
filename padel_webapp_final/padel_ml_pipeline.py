"""
=============================================================================
PADEL ANALYTICS — ML PIPELINE V2 (CREATIVE FEATURES)
=============================================================================
New features: Elo ratings, partner chemistry, clutch rate, dominance score,
ranking trajectory, tournament momentum, schedule density, opponent features
=============================================================================
"""
import os, warnings, numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor, XGBClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

DATA_DIR = "./data"
OUTPUT_DIR = "./output"
MODEL_DIR = "./models"
DIR_CLF = os.path.join(OUTPUT_DIR, "1_classification")
DIR_REG = os.path.join(OUTPUT_DIR, "2_regression")
DIR_CLU = os.path.join(OUTPUT_DIR, "3_clustering")
DIR_REC = os.path.join(OUTPUT_DIR, "4_recommendation")
DIR_ANO = os.path.join(OUTPUT_DIR, "5_anomaly_detection")
DIR_TS  = os.path.join(OUTPUT_DIR, "6_time_series")
for d in [OUTPUT_DIR, MODEL_DIR, DIR_CLF, DIR_REG, DIR_CLU, DIR_REC, DIR_ANO, DIR_TS]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# SECTION A — DATA LOADING & CREATIVE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION A — DATA LOADING & CREATIVE FEATURE ENGINEERING")
print("=" * 70)

fact_match = pd.read_csv(os.path.join(DATA_DIR, "fact_match.csv"))
fact_player = pd.read_csv(os.path.join(DATA_DIR, "fact_player.csv"))
fact_equip = pd.read_csv(os.path.join(DATA_DIR, "fact_equipement.csv"))
dim_player = pd.read_csv(os.path.join(DATA_DIR, "dim_player.csv"))
dim_tournament = pd.read_csv(os.path.join(DATA_DIR, "dim_tournament.csv"))
print(f"Loaded: {len(fact_match)} matches, {fact_player['player_name'].nunique()} players")

social_cols = ['instagram_followers','youtube_subscribers','tiktok_followers','twitter_followers','wikipedia_views']
for c in social_cols:
    fact_player[c] = pd.to_numeric(fact_player[c], errors='coerce').fillna(0)

round_order = {'Round of 64':1,'Round of 32':2,'Round of 16':3,'Quarterfinals':4,'Semifinals':5,'Final':6}

# ─── A.1 Parse scores into detailed metrics ─────────────────────────
def parse_score(s):
    if pd.isna(s): return []
    return [int(x) for x in str(s).split('-') if x.strip().isdigit()]

fact_match['sets_a'] = fact_match['score_team_a'].apply(parse_score)
fact_match['sets_b'] = fact_match['score_team_b'].apply(parse_score)
fact_match['n_sets'] = fact_match.apply(lambda r: max(len(r['sets_a']), len(r['sets_b'])), axis=1)
fact_match['is_3set'] = (fact_match['n_sets'] == 3).astype(int)
fact_match['games_a'] = fact_match['sets_a'].apply(sum)
fact_match['games_b'] = fact_match['sets_b'].apply(sum)
fact_match['match_date'] = pd.to_datetime(fact_match['match_date'], utc=True)
fact_match['round_num'] = fact_match['round'].map(round_order).fillna(0).astype(int)
fact_match = fact_match.merge(dim_tournament[['tournament_name','category','country']].rename(columns={'country':'tourn_country'}), on='tournament_name', how='left')
print("[✓] Scores parsed + tournament metadata merged")

# ─── A.2 ELO RATING SYSTEM ──────────────────────────────────────────
print("\n>>> Building Elo ratings...")
elo = {}
K = 32
elo_history = []

for _, row in fact_match.sort_values('match_date').iterrows():
    ta_key, tb_key = row['team_a_key'], row['team_b_key']
    for key in [ta_key, tb_key]:
        if key not in elo: elo[key] = 1500
    ea = 1 / (1 + 10 ** ((elo[tb_key] - elo[ta_key]) / 400))
    eb = 1 - ea
    a_won = 1 if row['winner_team'] == ta_key else 0
    elo[ta_key] += K * (a_won - ea)
    elo[tb_key] += K * ((1 - a_won) - eb)
    elo_history.append({'match_date': row['match_date'], 'team_key': ta_key, 'elo': elo[ta_key]})
    elo_history.append({'match_date': row['match_date'], 'team_key': tb_key, 'elo': elo[tb_key]})

# Map team Elo to the match
fact_match['elo_a'] = fact_match['team_a_key'].map(elo)
fact_match['elo_b'] = fact_match['team_b_key'].map(elo)
print(f"[✓] Elo ratings computed for {len(elo)} teams")

# ─── A.3 TEAM CHEMISTRY & PARTNER STATS ─────────────────────────────
print("\n>>> Computing partner chemistry...")
team_records = []
for _, row in fact_match.iterrows():
    for key in [row['team_a_key'], row['team_b_key']]:
        won = int(row['winner_team'] == key)
        team_records.append({'team_key': key, 'won': won, 'is_3set': row['is_3set'],
                             'tournament': row['tournament_name']})
team_df = pd.DataFrame(team_records)
team_agg = team_df.groupby('team_key').agg(
    team_matches=('won','count'), team_wins=('won','sum'),
    team_3set_matches=('is_3set','sum')
).reset_index()
team_agg['team_win_rate'] = team_agg['team_wins'] / team_agg['team_matches']

# 3-set wins per team
team_3set = team_df[team_df['is_3set']==1].groupby('team_key')['won'].agg(
    team_3set_wins='sum').reset_index()
team_agg = team_agg.merge(team_3set, on='team_key', how='left')
team_agg['team_3set_wins'] = team_agg['team_3set_wins'].fillna(0)
team_agg['team_clutch_rate'] = team_agg['team_3set_wins'] / team_agg['team_3set_matches'].clip(1)
team_agg['team_elo'] = team_agg['team_key'].map(elo).fillna(1500)
print(f"[✓] Team stats for {len(team_agg)} teams")

# ─── A.4 PER-PLAYER DETAILED STATS ──────────────────────────────────
print("\n>>> Building player stats with creative features...")
p_records = []
for _, row in fact_match.iterrows():
    for players, key, g_won, g_lost in [
        ([row['team_a_player1'], row['team_a_player2']], row['team_a_key'], row['games_a'], row['games_b']),
        ([row['team_b_player1'], row['team_b_player2']], row['team_b_key'], row['games_b'], row['games_a'])
    ]:
        won = int(row['winner_team'] == key)
        for p in players:
            p_records.append({
                'player_name': p, 'tournament': row['tournament_name'],
                'match_date': row['match_date'], 'year': row['source_year'],
                'round': row['round'], 'round_num': row['round_num'],
                'team_key': key, 'won': won, 'is_3set': row['is_3set'],
                'games_won': g_won, 'games_lost': g_lost, 'category': row.get('category',''),
                'dominance': (g_won - g_lost) / max(g_won + g_lost, 1)
            })
player_matches = pd.DataFrame(p_records)

# Partner tracking
partner_map = {}
for _, row in fact_match.iterrows():
    partner_map.setdefault(row['team_a_player1'], []).append(row['team_a_player2'])
    partner_map.setdefault(row['team_a_player2'], []).append(row['team_a_player1'])
    partner_map.setdefault(row['team_b_player1'], []).append(row['team_b_player2'])
    partner_map.setdefault(row['team_b_player2'], []).append(row['team_b_player1'])

# Aggregate per player
player_agg = player_matches.groupby('player_name').agg(
    total_matches=('won','count'), total_wins=('won','sum'),
    total_games_won=('games_won','sum'), total_games_lost=('games_lost','sum'),
    tournaments_played=('tournament','nunique'), years_active=('year','nunique'),
    avg_round=('round_num','mean'), max_round=('round_num','max'),
    matches_3set=('is_3set','sum'), avg_dominance=('dominance','mean'),
).reset_index()
player_agg['win_rate'] = player_agg['total_wins'] / player_agg['total_matches']
player_agg['game_diff'] = player_agg['total_games_won'] - player_agg['total_games_lost']

# Clutch rate (3-set win rate)
clutch = player_matches[player_matches['is_3set']==1].groupby('player_name')['won'].agg(
    wins_3set='sum', matches_3set_total='count').reset_index()
clutch['clutch_rate'] = clutch['wins_3set'] / clutch['matches_3set_total']
player_agg = player_agg.merge(clutch[['player_name','clutch_rate']], on='player_name', how='left')
player_agg['clutch_rate'] = player_agg['clutch_rate'].fillna(player_agg['win_rate'])

# Round-specific win rates
for rname, rnum in [('early', [1,2]), ('mid', [3,4]), ('late', [5,6])]:
    rdf = player_matches[player_matches['round_num'].isin(rnum)].groupby('player_name')['won'].mean().reset_index()
    rdf.columns = ['player_name', f'wr_{rname}']
    player_agg = player_agg.merge(rdf, on='player_name', how='left')
    player_agg[f'wr_{rname}'] = player_agg[f'wr_{rname}'].fillna(player_agg['win_rate'])

# Category-specific win rates
for cat in ['Major','P1','P2']:
    cdf = player_matches[player_matches['category']==cat].groupby('player_name')['won'].mean().reset_index()
    cdf.columns = ['player_name', f'wr_{cat.lower()}']
    player_agg = player_agg.merge(cdf, on='player_name', how='left')
    player_agg[f'wr_{cat.lower()}'] = player_agg[f'wr_{cat.lower()}'].fillna(player_agg['win_rate'])

# Partner loyalty
player_agg['unique_partners'] = player_agg['player_name'].map(
    lambda p: len(set(partner_map.get(p, []))))
player_agg['partner_loyalty'] = 1 / player_agg['unique_partners'].clip(1)

print(f"[✓] Player stats with creative features: {player_agg.shape}")

# ─── A.5 RANKING TRAJECTORY ─────────────────────────────────────────
print("\n>>> Computing ranking trajectories...")
multi_year = fact_player.groupby('player_name').filter(lambda x: x['year'].nunique() >= 2)
trajectories = multi_year.sort_values(['player_name','year']).groupby('player_name').agg(
    first_pos=('position','first'), last_pos=('position','last'),
    first_pts=('points','first'), last_pts=('points','last'),
    n_years=('year','nunique')
).reset_index()
trajectories['pos_improvement'] = trajectories['first_pos'] - trajectories['last_pos']
trajectories['pts_growth'] = trajectories['last_pts'] - trajectories['first_pts']
trajectories['trajectory_slope'] = trajectories['pos_improvement'] / trajectories['n_years']
player_agg = player_agg.merge(trajectories[['player_name','trajectory_slope','pts_growth','pos_improvement']], on='player_name', how='left')
player_agg[['trajectory_slope','pts_growth','pos_improvement']] = player_agg[['trajectory_slope','pts_growth','pos_improvement']].fillna(0)
print(f"[✓] Trajectories for {len(trajectories)} players with multi-year data")

# ─── A.6 MERGE WITH RANKING DATA ────────────────────────────────────
latest = fact_player.sort_values('year', ascending=False).drop_duplicates('player_name')
df_all = latest[['player_name','country','points','position'] + social_cols].copy()
df_all = df_all.merge(dim_player[['player_name','gender']].drop_duplicates('player_name'), on='player_name', how='left')
df_all = df_all.merge(player_agg, on='player_name', how='left')
match_fill = [c for c in player_agg.columns if c != 'player_name']
for c in match_fill:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna(0)

# Encoding
all_g = pd.concat([df_all['gender']]).fillna('unknown').unique()
le_gender = LabelEncoder(); le_gender.fit(np.append(all_g, 'unknown'))
df_all['gender_enc'] = le_gender.transform(df_all['gender'].fillna('unknown'))
all_c = df_all['country'].fillna('unknown').unique()
le_country = LabelEncoder(); le_country.fit(np.append(all_c, 'unknown'))
df_all['country_enc'] = le_country.transform(df_all['country'].fillna('unknown'))

df_all['total_social'] = df_all[social_cols].sum(axis=1)
df_all['log_social'] = np.log1p(df_all['total_social'])
df_all['log_points'] = np.log1p(df_all['points'].fillna(0))
df_all['points'] = df_all['points'].fillna(df_all['points'].median())
df_all['position'] = df_all['position'].fillna(df_all['position'].median())

print(f"\n[✓] Full dataset: {df_all.shape[0]} players, {df_all.shape[1]} features")
print(f"Creative features added: clutch_rate, avg_dominance, wr_early/mid/late, wr_major/p1/p2,")
print(f"  partner_loyalty, trajectory_slope, pts_growth, unique_partners")


# ═══════════════════════════════════════════════════════════════════════
# SECTION C — CLASSIFICATION (with opponent + team features)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION C — CLASSIFICATION: Win Prediction (IMPROVED)")
print("=" * 70)

# Build match-level features with BOTH teams
match_rows = []
for _, row in fact_match.iterrows():
    def get_player_stats(name):
        s = df_all[df_all['player_name'] == name]
        if len(s) == 0:
            return {c: 0 for c in ['win_rate','points','position','game_diff','clutch_rate',
                                    'avg_dominance','wr_late','partner_loyalty','log_points',
                                    'tournaments_played','trajectory_slope']}
        s = s.iloc[0]
        return {c: float(s.get(c, 0)) for c in ['win_rate','points','position','game_diff','clutch_rate',
                                                   'avg_dominance','wr_late','partner_loyalty','log_points',
                                                   'tournaments_played','trajectory_slope']}

    a1, a2 = get_player_stats(row['team_a_player1']), get_player_stats(row['team_a_player2'])
    b1, b2 = get_player_stats(row['team_b_player1']), get_player_stats(row['team_b_player2'])

    ta_stats = team_agg[team_agg['team_key'] == row['team_a_key']]
    tb_stats = team_agg[team_agg['team_key'] == row['team_b_key']]
    ta_elo = ta_stats['team_elo'].values[0] if len(ta_stats) > 0 else 1500
    tb_elo = tb_stats['team_elo'].values[0] if len(tb_stats) > 0 else 1500
    ta_chem = ta_stats['team_win_rate'].values[0] if len(ta_stats) > 0 else 0.5
    tb_chem = tb_stats['team_win_rate'].values[0] if len(tb_stats) > 0 else 0.5
    ta_clutch = ta_stats['team_clutch_rate'].values[0] if len(ta_stats) > 0 else 0.5
    tb_clutch = tb_stats['team_clutch_rate'].values[0] if len(tb_stats) > 0 else 0.5

    for side, p1, p2, my_elo, opp_elo, my_chem, opp_chem, my_clutch_t, opp_clutch_t, opp1, opp2, won_flag in [
        ('A', a1, a2, ta_elo, tb_elo, ta_chem, tb_chem, ta_clutch, tb_clutch, b1, b2, int(row['winner_team']==row['team_a_key'])),
        ('B', b1, b2, tb_elo, ta_elo, tb_chem, ta_chem, tb_clutch, ta_clutch, a1, a2, int(row['winner_team']==row['team_b_key']))
    ]:
        my_wr = (p1['win_rate']+p2['win_rate'])/2
        opp_wr = (opp1['win_rate']+opp2['win_rate'])/2
        my_pts = (p1['points']+p2['points'])/2
        opp_pts = (opp1['points']+opp2['points'])/2
        my_pos = (p1['position']+p2['position'])/2
        opp_pos = (opp1['position']+opp2['position'])/2

        match_rows.append({
            'my_win_rate': my_wr, 'my_points': my_pts, 'my_position': my_pos,
            'my_game_diff': (p1['game_diff']+p2['game_diff'])/2,
            'my_clutch': (p1['clutch_rate']+p2['clutch_rate'])/2,
            'my_dominance': (p1['avg_dominance']+p2['avg_dominance'])/2,
            'my_late_wr': (p1['wr_late']+p2['wr_late'])/2,
            'my_loyalty': (p1['partner_loyalty']+p2['partner_loyalty'])/2,
            'my_trajectory': (p1['trajectory_slope']+p2['trajectory_slope'])/2,
            'opp_win_rate': opp_wr, 'opp_points': opp_pts, 'opp_position': opp_pos,
            'opp_clutch': (opp1['clutch_rate']+opp2['clutch_rate'])/2,
            'opp_dominance': (opp1['avg_dominance']+opp2['avg_dominance'])/2,
            'diff_win_rate': my_wr - opp_wr,
            'diff_points': my_pts - opp_pts,
            'diff_position': opp_pos - my_pos,
            'diff_clutch': (p1['clutch_rate']+p2['clutch_rate'])/2 - (opp1['clutch_rate']+opp2['clutch_rate'])/2,
            'my_elo': my_elo, 'opp_elo': opp_elo, 'elo_diff': my_elo - opp_elo,
            'my_team_chemistry': my_chem, 'opp_team_chemistry': opp_chem,
            'chemistry_diff': my_chem - opp_chem,
            'my_team_clutch': my_clutch_t, 'opp_team_clutch': opp_clutch_t,
            'round_num': row['round_num'],
            'won': won_flag
        })

match_df = pd.DataFrame(match_rows)
feature_cols = [c for c in match_df.columns if c != 'won']
X_clf = match_df[feature_cols].fillna(0)
y_clf = match_df['won']

print(f"Classification: {X_clf.shape[0]} samples, {X_clf.shape[1]} features")
print(f"Class balance: {y_clf.value_counts().to_dict()}")

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf_results = {}
clf_models = {}
for name, model, params in [
    ("Random Forest", RandomForestClassifier(random_state=42),
     {'clf__n_estimators':[200],'clf__max_depth':[6,10],'clf__min_samples_split':[2]}),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42),
     {'clf__n_estimators':[200],'clf__learning_rate':[0.05],'clf__max_depth':[4,6]}),
    ("XGBoost", XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
     {'clf__n_estimators':[200],'clf__learning_rate':[0.05],'clf__max_depth':[5],'clf__subsample':[0.8],'clf__colsample_bytree':[0.8]})
]:
    print(f"\n>>> {name}...")
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    grid = GridSearchCV(pipe, params, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
    grid.fit(X_tr, y_tr)
    pred = grid.predict(X_te)
    proba = grid.predict_proba(X_te)[:,1]
    m = {'Accuracy': accuracy_score(y_te, pred), 'Precision': precision_score(y_te, pred),
         'Recall': recall_score(y_te, pred), 'F1': f1_score(y_te, pred),
         'ROC-AUC': roc_auc_score(y_te, proba)}
    clf_results[name] = m
    clf_models[name] = (grid, pred, proba)
    print(f"   AUC: {m['ROC-AUC']:.4f}, Acc: {m['Accuracy']:.4f}, F1: {m['F1']:.4f}")
    print(f"   Best: {grid.best_params_}")

# Visualizations — per model folder
for name, (grid, pred, proba) in clf_models.items():
    safe = name.replace(' ', '_')
    model_dir = os.path.join(DIR_CLF, safe)
    os.makedirs(model_dir, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_te, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Loss','Win'], yticklabels=['Loss','Win'])
    ax.set_title(f'{name} — Confusion Matrix\nAcc={clf_results[name]["Accuracy"]:.3f} AUC={clf_results[name]["ROC-AUC"]:.3f}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight'); plt.show()

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax)
    ax.plot([0,1],[0,1],'k--',alpha=0.3); ax.set_title(f'{name} — ROC Curve (AUC={clf_results[name]["ROC-AUC"]:.4f})')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "roc_curve.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 10))
    imp = grid.best_estimator_.named_steps['clf'].feature_importances_
    fi = pd.Series(imp, index=feature_cols).sort_values(ascending=True)
    fi.tail(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'{name} — Top 20 Feature Importance')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Classification report to file
    report = classification_report(y_te, pred, target_names=['Loss','Win'])
    with open(os.path.join(model_dir, "classification_report.txt"), 'w') as f:
        f.write(f"{name}\nBest params: {grid.best_params_}\n\n{report}")
    print(f"   [✓] {safe}/ — confusion_matrix, roc_curve, feature_importance, report")

# Combined comparison (in parent folder)
comp_df = pd.DataFrame(clf_results).T
fig, ax = plt.subplots(figsize=(10, 6))
comp_df.plot(kind='bar', ax=ax, colormap='Set2'); ax.set_ylim(0,1); ax.set_title('Classification — All Models Comparison')
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(DIR_CLF, "all_models_comparison.png"), dpi=150, bbox_inches='tight'); plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for name, (_, _, proba) in clf_models.items():
    RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax)
ax.plot([0,1],[0,1],'k--',alpha=0.3); ax.set_title('All Models — ROC Curves')
plt.tight_layout(); plt.savefig(os.path.join(DIR_CLF, "all_roc_curves.png"), dpi=150, bbox_inches='tight'); plt.show()

comp_df.to_csv(os.path.join(DIR_CLF, 'model_comparison.csv'))
print(f"\n[✓] Classification graphs saved per model in {DIR_CLF}/")


# ═══════════════════════════════════════════════════════════════════════
# SECTION D — REGRESSION (log-transformed + interactions)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION D — REGRESSION: Points Prediction (IMPROVED)")
print("=" * 70)

df_match_players = df_all[df_all['total_matches'] > 0].copy()
reg_features = ['total_matches','total_wins','win_rate','game_diff','tournaments_played',
                'years_active','avg_round','max_round','clutch_rate','avg_dominance',
                'wr_late','partner_loyalty','trajectory_slope','gender_enc','log_social']
X_reg = df_match_players[reg_features].fillna(0).copy()
X_reg['wr_x_matches'] = X_reg['win_rate'] * X_reg['total_matches']
X_reg['wins_per_tourn'] = X_reg['total_wins'] / X_reg['tournaments_played'].clip(1)
X_reg['gdiff_per_match'] = X_reg['game_diff'] / X_reg['total_matches'].clip(1)
y_reg_log = np.log1p(df_match_players['points'].fillna(0))
y_reg_raw = df_match_players['points'].fillna(0)

X_rtr, X_rte, y_rtr, y_rte = train_test_split(X_reg, y_reg_log, test_size=0.2, random_state=42)
_, _, y_rtr_raw, y_rte_raw = train_test_split(X_reg, y_reg_raw, test_size=0.2, random_state=42)

reg_results = {}
reg_models = {}
for name, model, params in [
    ("Ridge", Ridge(), {'reg__alpha':[0.1,1,10]}),
    ("Lasso", Lasso(max_iter=10000), {'reg__alpha':[0.01,0.1]}),
    ("XGBoost", XGBRegressor(random_state=42, verbosity=0),
     {'reg__n_estimators':[200],'reg__learning_rate':[0.05],'reg__max_depth':[5]})
]:
    print(f"\n>>> {name}...")
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
    grid = GridSearchCV(pipe, params, cv=KFold(5), scoring='r2')
    grid.fit(X_rtr, y_rtr)
    pred_log = grid.predict(X_rte)
    pred = np.expm1(pred_log)
    mse = mean_squared_error(y_rte_raw, pred)
    m = {'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(y_rte_raw, pred), 'R2': r2_score(y_rte_raw, pred)}
    reg_results[name] = m
    reg_models[name] = (grid, pred)
    print(f"   R²: {m['R2']:.4f}, RMSE: {m['RMSE']:.0f}, MAE: {m['MAE']:.0f}")

for name, (grid, pred) in reg_models.items():
    safe = name.replace(' ', '_')
    model_dir = os.path.join(DIR_REG, safe)
    os.makedirs(model_dir, exist_ok=True)

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_rte_raw, pred, alpha=0.5, s=20)
    lims = [min(y_rte_raw.min(), pred.min()), max(y_rte_raw.max(), pred.max())]
    ax.plot(lims, lims, 'r--'); ax.set_title(f'{name} — Actual vs Predicted\nR²={reg_results[name]["R2"]:.4f}  MAE={reg_results[name]["MAE"]:.0f}')
    ax.set_xlabel('Actual Points'); ax.set_ylabel('Predicted Points')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "actual_vs_predicted.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Residuals
    fig, ax = plt.subplots(figsize=(7, 6))
    res = y_rte_raw.values - pred; ax.scatter(pred, res, alpha=0.5, s=20); ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f'{name} — Residual Plot'); ax.set_xlabel('Predicted'); ax.set_ylabel('Residual')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "residuals.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Feature importance / coefficients
    fig, ax = plt.subplots(figsize=(10, 8))
    reg_model = grid.best_estimator_.named_steps['reg']
    if hasattr(reg_model, 'feature_importances_'):
        fi = pd.Series(reg_model.feature_importances_, index=X_reg.columns).sort_values(ascending=True)
        fi.plot(kind='barh', ax=ax, color='coral'); ax.set_title(f'{name} — Feature Importance')
    elif hasattr(reg_model, 'coef_'):
        fi = pd.Series(reg_model.coef_, index=X_reg.columns).sort_values(ascending=True)
        colors = ['green' if c > 0 else 'red' for c in fi]
        fi.plot(kind='barh', ax=ax, color=colors); ax.set_title(f'{name} — Coefficients')
        ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Metrics to file
    with open(os.path.join(model_dir, "metrics.txt"), 'w') as f:
        for k, v in reg_results[name].items():
            f.write(f"{k}: {v}\n")
    print(f"   [✓] {safe}/ — actual_vs_predicted, residuals, feature_importance, metrics")

pd.DataFrame(reg_results).T.to_csv(os.path.join(DIR_REG, 'model_comparison.csv'))
print(f"\n[✓] Regression graphs saved per model in {DIR_REG}/")


# ═══════════════════════════════════════════════════════════════════════
# SECTION E — CLUSTERING (all 4319 players)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION E — CLUSTERING (ALL PLAYERS)")
print("=" * 70)

cluster_features = ['points','position','total_matches','total_wins','win_rate','game_diff',
                    'log_social','log_points','clutch_rate','trajectory_slope']
X_cl = df_all[cluster_features].fillna(0)
scaler_c = StandardScaler()
X_cl_scaled = scaler_c.fit_transform(X_cl)

K_FINAL = 4
km_sil, gmm_sil, hier_sil = 0, 0, 0

inertias, sils, dbs = [], [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10); lb = km.fit_predict(X_cl_scaled)
    inertias.append(km.inertia_); sils.append(silhouette_score(X_cl_scaled, lb)); dbs.append(davies_bouldin_score(X_cl_scaled, lb))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(range(2,11), inertias, 'bo-'); axes[0].set_title('Elbow'); axes[0].set_xlabel('k')
axes[1].plot(range(2,11), sils, 'go-'); axes[1].set_title('Silhouette ↑'); axes[1].set_xlabel('k')
axes[2].plot(range(2,11), dbs, 'ro-'); axes[2].set_title('Davies-Bouldin ↓'); axes[2].set_xlabel('k')
plt.tight_layout(); plt.savefig(os.path.join(DIR_CLU, "elbow_silhouette.png"), dpi=150, bbox_inches='tight'); plt.show()

cluster_models_data = {}
for name, model in [("KMeans", KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)),
                     ("GMM", GaussianMixture(n_components=K_FINAL, random_state=42)),
                     ("Hierarchical", AgglomerativeClustering(n_clusters=K_FINAL))]:
    labels = model.fit_predict(X_cl_scaled)
    s = silhouette_score(X_cl_scaled, labels); db = davies_bouldin_score(X_cl_scaled, labels)
    cluster_models_data[name] = {'labels': labels, 'silhouette': s, 'db': db}
    if name == "KMeans": df_all['cluster'] = labels; km_sil = s; km_db = db
    elif name == "GMM": df_all['cluster_gmm'] = labels; gmm_sil = s
    else: df_all['cluster_hier'] = labels; hier_sil = s
    print(f"{name}: Silhouette={s:.4f}, DB={db:.4f}")

profile = df_all.groupby('cluster')[cluster_features].mean()
cluster_rank = profile['points'].sort_values(ascending=False).index.tolist()
labels_list = ['Stars','Contenders','Regulars','Newcomers']
label_map = {c: labels_list[i] if i < len(labels_list) else f'Group_{i}' for i, c in enumerate(cluster_rank)}
df_all['segment'] = df_all['cluster'].map(label_map)

pca = PCA(n_components=2); X_pca = pca.fit_transform(X_cl_scaled)

# Per-model folders
for name, data in cluster_models_data.items():
    model_dir = os.path.join(DIR_CLU, name)
    os.makedirs(model_dir, exist_ok=True)

    # PCA 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=data['labels'], cmap='viridis', alpha=0.4, s=10)
    ax.set_title(f'{name} — PCA 2D\nSilhouette={data["silhouette"]:.4f}  DB={data["db"]:.4f}')
    plt.colorbar(sc, ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "pca_2d.png"), dpi=150, bbox_inches='tight'); plt.show()

    # Metrics file
    with open(os.path.join(model_dir, "metrics.txt"), 'w') as f:
        f.write(f"Silhouette: {data['silhouette']:.4f}\nDavies-Bouldin: {data['db']:.4f}\n")
    print(f"   [✓] {name}/ — pca_2d, metrics")

# KMeans-specific: heatmap + distribution
km_dir = os.path.join(DIR_CLU, "KMeans")
fig, ax = plt.subplots(figsize=(12, 7))
prof_labeled = profile.rename(index=label_map)
sns.heatmap(prof_labeled.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('KMeans — Cluster Profiles Heatmap')
plt.tight_layout(); plt.savefig(os.path.join(km_dir, "cluster_heatmap.png"), dpi=150, bbox_inches='tight'); plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
df_all['segment'].value_counts().plot(kind='bar', ax=ax, color='teal', edgecolor='black')
ax.set_title('KMeans — Segment Distribution')
plt.tight_layout(); plt.savefig(os.path.join(km_dir, "segment_distribution.png"), dpi=150, bbox_inches='tight'); plt.show()

# Hierarchical-specific: dendrogram
hier_dir = os.path.join(DIR_CLU, "Hierarchical")
sample_idx = np.random.RandomState(42).choice(len(X_cl_scaled), size=min(200, len(X_cl_scaled)), replace=False)
Z = linkage(X_cl_scaled[sample_idx], method='ward')
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, truncate_mode='lastp', p=30, ax=ax); ax.set_title('Hierarchical — Dendrogram')
plt.tight_layout(); plt.savefig(os.path.join(hier_dir, "dendrogram.png"), dpi=150, bbox_inches='tight'); plt.show()

df_all.to_csv(os.path.join(DIR_CLU, 'all_players_processed.csv'), index=False)
profile.to_csv(os.path.join(DIR_CLU, 'cluster_profiles.csv'))
print(f"[✓] Clustering graphs saved per model in {DIR_CLU}/")


# ═══════════════════════════════════════════════════════════════════════
# OBJECTIVE 3 — RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBJECTIVE 3 — RECOMMENDATION")
print("=" * 70)

equip_players = fact_equip['player_name'].unique()
pro_profiles = df_all[df_all['player_name'].isin(equip_players)].copy()
pro_equip = fact_equip.groupby('player_name').agg(
    primary_brand=('brand', lambda x: x.mode()[0]), n_equipment=('pk_equipement','count'), avg_price=('price','mean')
).reset_index()
pro_profiles = pro_profiles.merge(pro_equip, on='player_name', how='left')

rec_features = ['points','position','total_matches','total_wins','win_rate','game_diff','gender_enc','log_social','clutch_rate']
scaler_rec = StandardScaler()
scaler_rec.fit(pro_profiles[rec_features].fillna(0))

# Evaluation
brand_hits, type_hits, total_evals = 0, 0, 0
for leave_out in equip_players:
    actual_brand = fact_equip[fact_equip['player_name']==leave_out]['brand'].mode()[0]
    actual_types = set(fact_equip[fact_equip['player_name']==leave_out]['type_produit'])
    pro_loo = pro_profiles[pro_profiles['player_name'] != leave_out]
    fe_loo = fact_equip[fact_equip['player_name'] != leave_out]
    if leave_out in df_all['player_name'].values:
        pv = scaler_rec.transform(df_all[df_all['player_name']==leave_out][rec_features].fillna(0))
        prov = scaler_rec.transform(pro_loo[rec_features].fillna(0))
        sims = cosine_similarity(pv, prov)[0]
        top3 = pro_loo.iloc[sims.argsort()[::-1][:3]]['player_name'].values
        recs = fe_loo[fe_loo['player_name'].isin(top3)]
        if len(recs) > 0:
            if actual_brand in set(recs['brand']): brand_hits += 1
            if actual_types & set(recs['type_produit']): type_hits += 1
            total_evals += 1

print(f"Brand Hit Rate: {brand_hits}/{total_evals} = {brand_hits/max(total_evals,1):.1%}")
print(f"Type Hit Rate:  {type_hits}/{total_evals} = {type_hits/max(total_evals,1):.1%}")

pro_feature_matrix = scaler_rec.transform(pro_profiles[rec_features].fillna(0))
cb_dir = os.path.join(DIR_REC, "Content_Based")
os.makedirs(cb_dir, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 8))
pro_sim = cosine_similarity(pro_feature_matrix)
sns.heatmap(pro_sim, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=[n.split()[-1] for n in pro_profiles['player_name']], yticklabels=[n.split()[-1] for n in pro_profiles['player_name']], ax=ax)
ax.set_title('Content-Based — Pro Player Similarity Matrix')
plt.tight_layout(); plt.savefig(os.path.join(cb_dir, "similarity_matrix.png"), dpi=150, bbox_inches='tight'); plt.show()
with open(os.path.join(cb_dir, "metrics.txt"), 'w') as f:
    f.write(f"Brand Hit Rate: {brand_hits}/{total_evals} = {brand_hits/max(total_evals,1):.1%}\nType Hit Rate: {type_hits}/{total_evals} = {type_hits/max(total_evals,1):.1%}\n")
print(f"[✓] Recommendation graphs saved in {DIR_REC}/")


# ═══════════════════════════════════════════════════════════════════════
# OBJECTIVE 4 — ANOMALY DETECTION (trajectory-based)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBJECTIVE 4 — TALENT DETECTION (IMPROVED)")
print("=" * 70)

df_talent = df_all[df_all['total_matches'] >= 3].copy()
df_talent['expected_wr'] = 1 - (df_talent['position'] / df_talent['position'].max())
df_talent['overperformance'] = df_talent['win_rate'] - df_talent['expected_wr']
df_talent['points_per_match'] = df_talent['points'] / df_talent['total_matches'].clip(1)

anomaly_features = ['win_rate','position','overperformance','points_per_match','total_matches',
                    'game_diff','clutch_rate','trajectory_slope','avg_dominance']
X_a = df_talent[anomaly_features].fillna(0)
scaler_a = StandardScaler()
X_a_scaled = scaler_a.fit_transform(X_a)

iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
df_talent['anomaly_iso'] = iso.fit_predict(X_a_scaled)
df_talent['anomaly_score'] = iso.score_samples(X_a_scaled)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df_talent['anomaly_lof'] = lof.fit_predict(X_a_scaled)

emerging = df_talent[(df_talent['anomaly_iso']==-1) & (df_talent['overperformance']>0)].sort_values('overperformance', ascending=False)
emerging_both = df_talent[(df_talent['anomaly_iso']==-1) & (df_talent['anomaly_lof']==-1) & (df_talent['overperformance']>0)].sort_values('overperformance', ascending=False)
print(f"Isolation Forest anomalies: {(df_talent['anomaly_iso']==-1).sum()}")
print(f"Emerging talents (IF): {len(emerging)}")
print(f"Emerging talents (consensus): {len(emerging_both)}")
if len(emerging) > 0:
    print(emerging[['player_name','country','position','points','win_rate','total_matches','overperformance','trajectory_slope','clutch_rate']].head(10).to_string(index=False))

# Per-model anomaly folders
for anom_name, col in [('Isolation_Forest', 'anomaly_iso'), ('LOF', 'anomaly_lof')]:
    model_dir = os.path.join(DIR_ANO, anom_name)
    os.makedirs(model_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    n = df_talent[df_talent[col]==1]; a = df_talent[df_talent[col]==-1]
    ax.scatter(n['position'], n['win_rate'], c='steelblue', alpha=0.4, s=20, label='Normal')
    ax.scatter(a['position'], a['win_rate'], c='red', alpha=0.7, s=40, edgecolors='black', linewidth=0.5, label='Anomaly')
    op = df_talent[(df_talent[col]==-1)&(df_talent['overperformance']>0)]
    if len(op)>0: ax.scatter(op['position'], op['win_rate'], c='gold', s=60, marker='*', edgecolors='red', label='Emerging Talent')
    ax.set_xlabel('Rank'); ax.set_ylabel('Win Rate'); ax.set_title(f'{anom_name.replace("_"," ")} — Anomaly Detection')
    ax.legend(); ax.invert_xaxis()
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "anomaly_scatter.png"), dpi=150, bbox_inches='tight'); plt.show()

    if col == 'anomaly_iso':
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_talent['anomaly_score'], bins=30, color='steelblue', edgecolor='black')
        ax.axvline(x=df_talent[df_talent['anomaly_iso']==-1]['anomaly_score'].max(), color='red', linestyle='--', label='Threshold')
        ax.set_title('Isolation Forest — Anomaly Score Distribution'); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(model_dir, "score_distribution.png"), dpi=150, bbox_inches='tight'); plt.show()

    n_anom = (df_talent[col]==-1).sum()
    with open(os.path.join(model_dir, "metrics.txt"), 'w') as f:
        f.write(f"Anomalies detected: {n_anom}\nEmerging talents: {len(op)}\n")
    print(f"   [✓] {anom_name}/ — anomaly_scatter, metrics")

# Overperformance plot (shared)
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df_talent['position'], df_talent['overperformance'], c=df_talent['trajectory_slope'], cmap='RdYlGn', alpha=0.6, s=30)
plt.colorbar(sc, ax=ax, label='Trajectory Slope (+ = improving)')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Rank'); ax.set_ylabel('Overperformance'); ax.set_title('Overperformance + Trajectory'); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(os.path.join(DIR_ANO, "overperformance_trajectory.png"), dpi=150, bbox_inches='tight'); plt.show()
df_talent.to_csv(os.path.join(DIR_ANO, 'talent_detection_results.csv'), index=False)
print(f"[✓] Anomaly graphs saved per model in {DIR_ANO}/")


# ═══════════════════════════════════════════════════════════════════════
# SECTION F — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION F — TIME SERIES")
print("=" * 70)

player_matches['match_month'] = player_matches['match_date'].dt.to_period('M')
monthly = player_matches.groupby(['player_name','match_month']).agg(
    monthly_wr=('won','mean'), monthly_matches=('won','count')).reset_index()
eligible = monthly.groupby('player_name')['match_month'].count()
top_player = eligible[eligible >= 8].idxmax() if (eligible >= 8).any() else eligible.idxmax()
ts = monthly[monthly['player_name']==top_player].copy()
ts['date'] = ts['match_month'].dt.to_timestamp()
ts = ts.set_index('date').sort_index()
target = ts['monthly_wr'].dropna()
print(f"Player: {top_player}, {len(target)} months")

n_test = max(2, len(target)//4)
train_ts, test_ts = target[:-n_test], target[-n_test:]

ts_results = {}
try:
    best_aic, best_arima, best_order = np.inf, None, (0,0,0)
    for p in range(2):
        for d in range(2):
            for q in range(2):
                try:
                    fit = ARIMA(train_ts, order=(p,d,q)).fit()
                    if fit.aic < best_aic: best_aic, best_arima, best_order = fit.aic, fit, (p,d,q)
                except: pass
    arima_fc = best_arima.forecast(steps=n_test)
    ts_results['ARIMA'] = {'MAE': mean_absolute_error(test_ts, arima_fc), 'RMSE': np.sqrt(mean_squared_error(test_ts, arima_fc))}
    print(f"ARIMA {best_order}: MAE={ts_results['ARIMA']['MAE']:.4f}")
except Exception as e:
    print(f"ARIMA failed: {e}")

try:
    from prophet import Prophet
    pdf = target.reset_index(); pdf.columns = ['ds','y']
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pdf[:-n_test])
    future = m.make_future_dataframe(periods=n_test, freq='MS')
    fc = m.predict(future)
    prophet_pred = fc.tail(n_test)['yhat'].values
    ts_results['Prophet'] = {'MAE': mean_absolute_error(test_ts, prophet_pred), 'RMSE': np.sqrt(mean_squared_error(test_ts, prophet_pred))}
    print(f"Prophet: MAE={ts_results['Prophet']['MAE']:.4f}")
except Exception as e:
    print(f"Prophet failed: {e}")

# Per-model time series folders
if 'ARIMA' in ts_results:
    arima_dir = os.path.join(DIR_TS, "ARIMA")
    os.makedirs(arima_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(target.index, target.values, 'b-o', label='Actual', markersize=5)
    ax.plot(test_ts.index, arima_fc.values, 'r--s', label=f'ARIMA {best_order}', markersize=8)
    ax.axvline(x=test_ts.index[0], color='gray', linestyle=':', alpha=0.7)
    ax.set_title(f'ARIMA — Win Rate Forecast ({top_player})\nMAE={ts_results["ARIMA"]["MAE"]:.4f}')
    ax.legend(); ax.set_ylim(-0.1, 1.1)
    plt.tight_layout(); plt.savefig(os.path.join(arima_dir, "forecast.png"), dpi=150, bbox_inches='tight'); plt.show()
    with open(os.path.join(arima_dir, "metrics.txt"), 'w') as f:
        f.write(f"Order: {best_order}\nMAE: {ts_results['ARIMA']['MAE']:.4f}\nRMSE: {ts_results['ARIMA']['RMSE']:.4f}\n")
    print(f"   [✓] ARIMA/ — forecast, metrics")

if 'Prophet' in ts_results:
    prophet_dir = os.path.join(DIR_TS, "Prophet")
    os.makedirs(prophet_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(target.index, target.values, 'b-o', label='Actual', markersize=5)
    ax.plot(pdf.tail(n_test)['ds'].values, prophet_pred, 'g--^', label='Prophet', markersize=8)
    ax.axvline(x=test_ts.index[0], color='gray', linestyle=':', alpha=0.7)
    ax.set_title(f'Prophet — Win Rate Forecast ({top_player})\nMAE={ts_results["Prophet"]["MAE"]:.4f}')
    ax.legend(); ax.set_ylim(-0.1, 1.1)
    plt.tight_layout(); plt.savefig(os.path.join(prophet_dir, "forecast.png"), dpi=150, bbox_inches='tight'); plt.show()
    with open(os.path.join(prophet_dir, "metrics.txt"), 'w') as f:
        f.write(f"MAE: {ts_results['Prophet']['MAE']:.4f}\nRMSE: {ts_results['Prophet']['RMSE']:.4f}\n")
    print(f"   [✓] Prophet/ — forecast, metrics")

# Combined comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(target.index, target.values, 'b-o', label='Actual', markersize=5)
if 'ARIMA' in ts_results:
    ax.plot(test_ts.index, arima_fc.values, 'r--s', label='ARIMA', markersize=8)
if 'Prophet' in ts_results:
    ax.plot(pdf.tail(n_test)['ds'].values, prophet_pred, 'g--^', label='Prophet', markersize=8)
ax.axvline(x=test_ts.index[0], color='gray', linestyle=':', alpha=0.7)
ax.set_title(f'All Models — Win Rate Forecast ({top_player})'); ax.legend(); ax.set_ylim(-0.1, 1.1)
plt.tight_layout(); plt.savefig(os.path.join(DIR_TS, "all_models_forecast.png"), dpi=150, bbox_inches='tight'); plt.show()
pd.DataFrame(ts_results).T.to_csv(os.path.join(DIR_TS, 'model_comparison.csv'))
print(f"[✓] Time series graphs saved per model in {DIR_TS}/")


# ═══════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

for name, (grid, _, _) in clf_models.items():
    safe = name.lower().replace(' ','_')
    joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, f'classifier_{safe}.pkl'))
for name, (grid, _) in reg_models.items():
    safe = name.lower().replace(' ','_')
    joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, f'regressor_{safe}.pkl'))
joblib.dump(KMeans(n_clusters=K_FINAL, random_state=42, n_init=10).fit(X_cl_scaled), os.path.join(MODEL_DIR, 'kmeans.pkl'))
joblib.dump(GaussianMixture(n_components=K_FINAL, random_state=42).fit(X_cl_scaled), os.path.join(MODEL_DIR, 'gmm.pkl'))
joblib.dump(iso, os.path.join(MODEL_DIR, 'iso_forest.pkl'))
joblib.dump(scaler_c, os.path.join(MODEL_DIR, 'scaler_cluster.pkl'))
joblib.dump(scaler_rec, os.path.join(MODEL_DIR, 'scaler_recommender.pkl'))
joblib.dump(scaler_a, os.path.join(MODEL_DIR, 'scaler_anomaly.pkl'))
joblib.dump(le_gender, os.path.join(MODEL_DIR, 'le_gender.pkl'))
joblib.dump(le_country, os.path.join(MODEL_DIR, 'le_country.pkl'))

# Save the team_agg and feature list for webapp
team_agg.to_csv(os.path.join(MODEL_DIR, 'team_agg.csv'), index=False)
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'clf_feature_cols.pkl'))
print("[✓] All models saved")

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V2 PIPELINE COMPLETE — SUMMARY")
print("=" * 70)

best_clf = max(clf_results, key=lambda x: clf_results[x]['ROC-AUC'])
best_reg_name = max(reg_results, key=lambda x: reg_results[x]['R2'])
print(f"""
CLASSIFICATION (Win Prediction — with opponent + Elo + chemistry):
   Best: {best_clf}
   AUC:      {clf_results[best_clf]['ROC-AUC']:.4f}
   Accuracy: {clf_results[best_clf]['Accuracy']:.4f}
   F1:       {clf_results[best_clf]['F1']:.4f}
   Features: {len(feature_cols)} (was 9 in v1)

REGRESSION (Points — log-transformed + interactions):
   Best: {best_reg_name}
   R²:   {reg_results[best_reg_name]['R2']:.4f}
   RMSE: {reg_results[best_reg_name]['RMSE']:.0f}
   MAE:  {reg_results[best_reg_name]['MAE']:.0f}

CLUSTERING: K-Means Sil={km_sil:.4f}
SEGMENTS: {dict(df_all['segment'].value_counts())}
RECOMMENDATION: Brand Hit={brand_hits}/{total_evals}, Type Hit={type_hits}/{total_evals}
ANOMALY: {len(emerging)} emerging talents (IF), {len(emerging_both)} consensus
""")
