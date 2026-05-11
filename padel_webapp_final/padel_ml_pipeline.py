"""
=============================================================================
PADEL ANALYTICS — ML PIPELINE V2 (CREATIVE FEATURES) + MLFLOW TRACKING
=============================================================================
"""
import os, warnings, numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt, seaborn as sns, joblib
import tempfile
from sqlalchemy import create_engine, text
import sys
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

# ═══════════════════════════════════════════════════════════════════════
# MLFLOW SETUP
# ═══════════════════════════════════════════════════════════════════════
import mlflow
import mlflow.sklearn
import mlflow.xgboost

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Padel_Analytics")

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ═══════════════════════════════════════════════════════════════════════
# DATABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'dwh_padel'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'khalil')
}

if sys.platform == 'win32':
    os.environ['PGCLIENTENCODING'] = 'UTF8'

user = DB_CONFIG['user']
password = DB_CONFIG['password']
host = DB_CONFIG['host']
port = int(DB_CONFIG['port'])
database = DB_CONFIG['database']

print(f"[*] Attempting to connect to PostgreSQL at {host}:{port}...")
print(f"[*] Database: {database}, User: {user}")

engine = None
conn = None

try:
    db_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url, echo=False, pool_pre_ping=True, pool_recycle=3600)
    conn = engine.connect()
    print(f"[OK] Connected to PostgreSQL: {host}:{port}/{database}")
except Exception as e:
    print(f"[WARNING]  PostgreSQL connection failed: {type(e).__name__}: {str(e)[:100]}")
    conn = None

def read_table(table_name):
    if conn is None:
        raise RuntimeError("No database connection available. Cannot load data.")
    try:
        query = text(f"SELECT * FROM {table_name}")
        return pd.read_sql(query, conn)
    except Exception as e:
        print(f"[ERROR] Could not load {table_name}: {e}")
        raise

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

print("=" * 70)
print("SECTION A — DATA LOADING & CREATIVE FEATURE ENGINEERING")
print("=" * 70)

print("\n>>> Loading data from PostgreSQL...")
fact_match = read_table('fact_match')
fact_player = read_table('fact_player')
fact_equip = read_table('fact_equipement')
dim_player = read_table('dim_player')
dim_tournament = read_table('dim_tournament')
print(f"[OK] Loaded: {len(fact_match)} matches, {fact_player['player_name'].nunique()} players")

print("\n>>> Fixing player name mismatches...")

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

match_players = set(fact_match['team_a_player1']) | set(fact_match['team_b_player1']) | set(fact_match['team_a_player2']) | set(fact_match['team_b_player2'])
ranking_players = set(fact_player['player_name'])
still_missing = match_players - ranking_players

print(f"\n=== STILL MISSING (from PostgreSQL) ===")
print(f"Missing: {len(still_missing)} players")
for p in sorted(still_missing):
    n = len(fact_match[(fact_match['team_a_player1']==p)|(fact_match['team_a_player2']==p)|(fact_match['team_b_player1']==p)|(fact_match['team_b_player2']==p)])
    print(f"  {repr(p)} - {n} matches")

print("\n=== FACT_PLAYER NAMES (PostgreSQL) ===")
for missing_name in ['Federico Mouriño', 'Ignacio Vilariño Gestoso', 'Valentino Libaak', 
                      'Leandro Roman Augsburguer', 'Pablo Lijó', 'Agustín Torre']:
    matches = fact_player[fact_player['player_name'].str.contains(missing_name.split()[0], na=False)]
    if missing_name.split()[-1] != missing_name.split()[0]:
        matches = matches[matches['player_name'].str.contains(missing_name.split()[-1][:4], na=False)]
    for _, r in matches.iterrows():
        print(f"  {repr(r['player_name'])} - rank #{r['position']}, {r['points']} pts")

name_cols = ['team_a_player1','team_a_player2','team_b_player1','team_b_player2']
for col in name_cols:
    fact_match[col] = fact_match[col].replace(NAME_MAP)

fact_match['team_a_key'] = fact_match['team_a_player1'] + ' / ' + fact_match['team_a_player2']
fact_match['team_b_key'] = fact_match['team_b_player1'] + ' / ' + fact_match['team_b_player2']

for _, row in fact_match.iterrows():
    old_winner = row['winner_team']
    if old_winner not in [row['team_a_key'], row['team_b_key']]:
        a_players = {row['team_a_player1'], row['team_a_player2']}
        for p in a_players:
            if p in str(old_winner) or any(orig in str(old_winner) for orig, new in NAME_MAP.items() if new == p):
                fact_match.at[_, 'winner_team'] = row['team_a_key']
                break
        else:
            fact_match.at[_, 'winner_team'] = row['team_b_key']

all_drop = set(DROP_PLAYERS)
mask = ~(fact_match['team_a_player1'].isin(all_drop) | fact_match['team_a_player2'].isin(all_drop) |
         fact_match['team_b_player1'].isin(all_drop) | fact_match['team_b_player2'].isin(all_drop))
n_before = len(fact_match)
fact_match = fact_match[mask].reset_index(drop=True)
print(f"[OK] Fixed 26 name mismatches, dropped {n_before - len(fact_match)} matches")
print(f"[OK] Matches: {n_before} -> {len(fact_match)} ({len(fact_match)/n_before*100:.1f}% retained)")

social_cols = ['instagram_followers','youtube_subscribers','tiktok_followers','twitter_followers','wikipedia_views']
for c in social_cols:
    fact_player[c] = pd.to_numeric(fact_player[c], errors='coerce').fillna(0)

round_order = {'Round of 64':1,'Round of 32':2,'Round of 16':3,'Quarterfinals':4,'Semifinals':5,'Final':6}

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
print("[OK] Scores parsed + tournament metadata merged")

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

fact_match['elo_a'] = fact_match['team_a_key'].map(elo)
fact_match['elo_b'] = fact_match['team_b_key'].map(elo)
print(f"[OK] Elo ratings computed for {len(elo)} teams")

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

team_3set = team_df[team_df['is_3set']==1].groupby('team_key')['won'].agg(team_3set_wins='sum').reset_index()
team_agg = team_agg.merge(team_3set, on='team_key', how='left')
team_agg['team_3set_wins'] = team_agg['team_3set_wins'].fillna(0)
team_agg['team_clutch_rate'] = team_agg['team_3set_wins'] / team_agg['team_3set_matches'].clip(1)
team_agg['team_elo'] = team_agg['team_key'].map(elo).fillna(1500)
print(f"[OK] Team stats for {len(team_agg)} teams")

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

partner_map = {}
for _, row in fact_match.iterrows():
    partner_map.setdefault(row['team_a_player1'], []).append(row['team_a_player2'])
    partner_map.setdefault(row['team_a_player2'], []).append(row['team_a_player1'])
    partner_map.setdefault(row['team_b_player1'], []).append(row['team_b_player2'])
    partner_map.setdefault(row['team_b_player2'], []).append(row['team_b_player1'])

player_agg = player_matches.groupby('player_name').agg(
    total_matches=('won','count'), total_wins=('won','sum'),
    total_games_won=('games_won','sum'), total_games_lost=('games_lost','sum'),
    tournaments_played=('tournament','nunique'), years_active=('year','nunique'),
    avg_round=('round_num','mean'), max_round=('round_num','max'),
    matches_3set=('is_3set','sum'), avg_dominance=('dominance','mean'),
).reset_index()
player_agg['win_rate'] = player_agg['total_wins'] / player_agg['total_matches']
player_agg['game_diff'] = player_agg['total_games_won'] - player_agg['total_games_lost']

clutch = player_matches[player_matches['is_3set']==1].groupby('player_name')['won'].agg(
    wins_3set='sum', matches_3set_total='count').reset_index()
clutch['clutch_rate'] = clutch['wins_3set'] / clutch['matches_3set_total']
player_agg = player_agg.merge(clutch[['player_name','clutch_rate']], on='player_name', how='left')
player_agg['clutch_rate'] = player_agg['clutch_rate'].fillna(player_agg['win_rate'])

for rname, rnum in [('early', [1,2]), ('mid', [3,4]), ('late', [5,6])]:
    rdf = player_matches[player_matches['round_num'].isin(rnum)].groupby('player_name')['won'].mean().reset_index()
    rdf.columns = ['player_name', f'wr_{rname}']
    player_agg = player_agg.merge(rdf, on='player_name', how='left')
    player_agg[f'wr_{rname}'] = player_agg[f'wr_{rname}'].fillna(player_agg['win_rate'])

for cat in ['Major','P1','P2']:
    cdf = player_matches[player_matches['category']==cat].groupby('player_name')['won'].mean().reset_index()
    cdf.columns = ['player_name', f'wr_{cat.lower()}']
    player_agg = player_agg.merge(cdf, on='player_name', how='left')
    player_agg[f'wr_{cat.lower()}'] = player_agg[f'wr_{cat.lower()}'].fillna(player_agg['win_rate'])

player_agg['unique_partners'] = player_agg['player_name'].map(lambda p: len(set(partner_map.get(p, []))))
player_agg['partner_loyalty'] = 1 / player_agg['unique_partners'].clip(1)
print(f"[OK] Player stats with creative features: {player_agg.shape}")

print("\n>>> Computing ranking trajectories...")
multi_year = fact_player.groupby('player_name').filter(lambda x: x['year'].nunique() >= 2)
multi_year = multi_year.copy()
multi_year['position'] = pd.to_numeric(multi_year['position'], errors='coerce')
multi_year['points'] = pd.to_numeric(multi_year['points'], errors='coerce')
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
print(f"[OK] Trajectories for {len(trajectories)} players")

latest = fact_player.sort_values('year', ascending=False).drop_duplicates('player_name')
df_all = latest[['player_name','country','points','position'] + social_cols].copy()
df_all['points'] = pd.to_numeric(df_all['points'], errors='coerce')
df_all['position'] = pd.to_numeric(df_all['position'], errors='coerce')
df_all = df_all.merge(dim_player[['player_name','gender']].drop_duplicates('player_name'), on='player_name', how='left')
df_all = df_all.merge(player_agg, on='player_name', how='left')
match_fill = [c for c in player_agg.columns if c != 'player_name']
for c in match_fill:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna(0)

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

print(f"\n[OK] Full dataset: {df_all.shape[0]} players, {df_all.shape[1]} features")


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION C — CLASSIFICATION: Win Prediction")
print("=" * 70)

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
    with mlflow.start_run(run_name=f"Classifier_{name.replace(' ','_')}"):
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
        
        mlflow.log_param("model_type", name)
        mlflow.log_param("task", "classification")
        mlflow.log_param("n_features", X_clf.shape[1])
        mlflow.log_param("n_samples", X_clf.shape[0])
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        for metric_name, metric_val in m.items():
            mlflow.log_metric(metric_name.replace('-','_'), metric_val)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        
        print(f"   AUC: {m['ROC-AUC']:.4f}, Acc: {m['Accuracy']:.4f}, F1: {m['F1']:.4f}")
        print(f"   Best: {grid.best_params_}")

for name, (grid, pred, proba) in clf_models.items():
    safe = name.replace(' ', '_')
    model_dir = os.path.join(DIR_CLF, safe)
    os.makedirs(model_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_te, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Loss','Win'], yticklabels=['Loss','Win'])
    ax.set_title(f'{name} - Confusion Matrix\nAcc={clf_results[name]["Accuracy"]:.3f} AUC={clf_results[name]["ROC-AUC"]:.3f}')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax)
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "roc_curve.png"), dpi=150, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    imp = grid.best_estimator_.named_steps['clf'].feature_importances_
    fi = pd.Series(imp, index=feature_cols).sort_values(ascending=True)
    fi.tail(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'{name} - Top 20 Feature Importance')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "feature_importance.png"), dpi=150, bbox_inches='tight'); plt.close()

comp_df = pd.DataFrame(clf_results).T
comp_df.to_csv(os.path.join(DIR_CLF, 'model_comparison.csv'))
print(f"\n[OK] Classification done")


# ═══════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION D — REGRESSION: Points Prediction")
print("=" * 70)

df_match_players = df_all[df_all['total_matches'] > 0].copy()
reg_features = ['total_matches','total_wins','win_rate','game_diff','tournaments_played',
                'years_active','avg_round','max_round','clutch_rate','avg_dominance',
                'wr_late','partner_loyalty','trajectory_slope','gender_enc','log_social',
                'position']
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
     {'reg__n_estimators':[300,500],'reg__learning_rate':[0.03,0.05],'reg__max_depth':[5,7,9]})
]:
    print(f"\n>>> {name}...")
    with mlflow.start_run(run_name=f"Regressor_{name}"):
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', model)])
        grid = GridSearchCV(pipe, params, cv=KFold(5), scoring='r2')
        grid.fit(X_rtr, y_rtr)
        pred_log = grid.predict(X_rte)
        pred = np.expm1(pred_log)
        mse = mean_squared_error(y_rte_raw, pred)
        m = {'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(y_rte_raw, pred), 'R2': r2_score(y_rte_raw, pred)}
        reg_results[name] = m
        reg_models[name] = (grid, pred)
        
        mlflow.log_param("model_type", name)
        mlflow.log_param("task", "regression")
        mlflow.log_param("n_features", X_reg.shape[1])
        mlflow.log_param("n_samples", X_reg.shape[0])
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        for metric_name, metric_val in m.items():
            mlflow.log_metric(metric_name, metric_val)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        
        print(f"   R2: {m['R2']:.4f}, RMSE: {m['RMSE']:.0f}, MAE: {m['MAE']:.0f}")

for name, (grid, pred) in reg_models.items():
    safe = name.replace(' ', '_')
    model_dir = os.path.join(DIR_REG, safe)
    os.makedirs(model_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_rte_raw, pred, alpha=0.5, s=20)
    lims = [min(y_rte_raw.min(), pred.min()), max(y_rte_raw.max(), pred.max())]
    ax.plot(lims, lims, 'r--')
    ax.set_title(f'{name} - Actual vs Predicted\nR2={reg_results[name]["R2"]:.4f}')
    plt.tight_layout(); plt.savefig(os.path.join(model_dir, "actual_vs_predicted.png"), dpi=150, bbox_inches='tight'); plt.close()

pd.DataFrame(reg_results).T.to_csv(os.path.join(DIR_REG, 'model_comparison.csv'))
print(f"\n[OK] Regression done")


# ═══════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION E — CLUSTERING")
print("=" * 70)

cluster_features = ['points','position','total_matches','total_wins','win_rate','game_diff',
                    'log_social','log_points','clutch_rate','trajectory_slope']
X_cl = df_all[cluster_features].fillna(0)
scaler_c = StandardScaler()
X_cl_scaled = scaler_c.fit_transform(X_cl)

K_FINAL = 4
km_sil, gmm_sil, hier_sil = 0, 0, 0

cluster_models_data = {}
for name, model in [("KMeans", KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)),
                     ("GMM", GaussianMixture(n_components=K_FINAL, random_state=42)),
                     ("Hierarchical", AgglomerativeClustering(n_clusters=K_FINAL))]:
    with mlflow.start_run(run_name=f"Clustering_{name}"):
        labels = model.fit_predict(X_cl_scaled)
        s = silhouette_score(X_cl_scaled, labels); db = davies_bouldin_score(X_cl_scaled, labels)
        cluster_models_data[name] = {'labels': labels, 'silhouette': s, 'db': db}
        if name == "KMeans": df_all['cluster'] = labels; km_sil = s
        elif name == "GMM": df_all['cluster_gmm'] = labels; gmm_sil = s
        else: df_all['cluster_hier'] = labels; hier_sil = s
        
        mlflow.log_param("model_type", name)
        mlflow.log_param("task", "clustering")
        mlflow.log_param("n_clusters", K_FINAL)
        mlflow.log_metric("silhouette", s)
        mlflow.log_metric("davies_bouldin", db)
        # KMeans + GMM are picklable; Hierarchical (AgglomerativeClustering) needs special handling
        try:
            if name == "Hierarchical":
                # AgglomerativeClustering doesn't have predict() — log via joblib artifact
                hier_path = os.path.join(tempfile.gettempdir(), "hier_model.pkl")
                joblib.dump(model, hier_path)
                mlflow.log_artifact(hier_path, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"   [WARN] Could not log {name} model: {e}")
        
        print(f"{name}: Silhouette={s:.4f}, DB={db:.4f}")

profile = df_all.groupby('cluster')[cluster_features].mean()
cluster_rank = profile['points'].sort_values(ascending=False).index.tolist()
labels_list = ['Stars','Contenders','Regulars','Newcomers']
label_map = {c: labels_list[i] if i < len(labels_list) else f'Group_{i}' for i, c in enumerate(cluster_rank)}
df_all['segment'] = df_all['cluster'].map(label_map)

print(f"[OK] Clustering done")


# ═══════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

equip_players = fact_equip['player_name'].unique()
pro_profiles = df_all[df_all['player_name'].isin(equip_players)].copy()
fact_equip_clean = fact_equip.copy()
fact_equip_clean['price'] = pd.to_numeric(fact_equip_clean['price'], errors='coerce')

pro_equip = fact_equip_clean.groupby('player_name').agg(
    primary_brand=('brand', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'),
    n_equipment=('pk_equipement','count'),
    avg_price=('price','mean')
).reset_index()
pro_profiles = pro_profiles.merge(pro_equip, on='player_name', how='left')

rec_features = ['points','position','total_matches','total_wins','win_rate','game_diff','gender_enc','log_social','clutch_rate']
scaler_rec = StandardScaler()
scaler_rec.fit(pro_profiles[rec_features].fillna(0))

with mlflow.start_run(run_name="Recommendation_ContentBased"):
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
    
    brand_hit_rate = brand_hits / max(total_evals, 1)
    type_hit_rate = type_hits / max(total_evals, 1)
    mlflow.log_param("model_type", "ContentBased")
    mlflow.log_param("task", "recommendation")
    mlflow.log_metric("brand_hit_rate", brand_hit_rate)
    mlflow.log_metric("type_hit_rate", type_hit_rate)
    # Log the scaler as the "model" artifact (recommendation uses cosine similarity, no actual ML model)
    mlflow.sklearn.log_model(scaler_rec, "model")

print(f"Brand Hit Rate: {brand_hits}/{total_evals} = {brand_hit_rate:.1%}")
print(f"Type Hit Rate:  {type_hits}/{total_evals} = {type_hit_rate:.1%}")


# ═══════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANOMALY DETECTION")
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

with mlflow.start_run(run_name="Anomaly_IsolationForest"):
    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    df_talent['anomaly_iso'] = iso.fit_predict(X_a_scaled)
    df_talent['anomaly_score'] = iso.score_samples(X_a_scaled)
    n_anom_iso = (df_talent['anomaly_iso']==-1).sum()
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("task", "anomaly_detection")
    mlflow.log_param("contamination", 0.1)
    mlflow.log_metric("anomalies_detected", n_anom_iso)
    mlflow.sklearn.log_model(iso, "model")

with mlflow.start_run(run_name="Anomaly_LOF"):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df_talent['anomaly_lof'] = lof.fit_predict(X_a_scaled)
    n_anom_lof = (df_talent['anomaly_lof']==-1).sum()
    mlflow.log_param("model_type", "LOF")
    mlflow.log_param("task", "anomaly_detection")
    mlflow.log_param("n_neighbors", 20)
    mlflow.log_metric("anomalies_detected", n_anom_lof)
    # LOF doesn't support sklearn.log_model directly (no predict() method), use joblib artifact
    lof_path = os.path.join(tempfile.gettempdir(), "lof_model.pkl")
    joblib.dump(lof, lof_path)
    mlflow.log_artifact(lof_path, artifact_path="model")

emerging = df_talent[(df_talent['anomaly_iso']==-1) & (df_talent['overperformance']>0)].sort_values('overperformance', ascending=False)
emerging_both = df_talent[(df_talent['anomaly_iso']==-1) & (df_talent['anomaly_lof']==-1) & (df_talent['overperformance']>0)]
print(f"Emerging talents (consensus): {len(emerging_both)}")


# ═══════════════════════════════════════════════════════════════════════
# TIME SERIES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TIME SERIES")
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
n_test = max(2, len(target)//4)
train_ts, test_ts = target[:-n_test], target[-n_test:]

ts_results = {}
try:
    with mlflow.start_run(run_name="TimeSeries_ARIMA"):
        best_aic, best_arima, best_order = np.inf, None, (0,0,0)
        for p in range(2):
            for d in range(2):
                for q in range(2):
                    try:
                        fit = ARIMA(train_ts, order=(p,d,q)).fit()
                        if fit.aic < best_aic: best_aic, best_arima, best_order = fit.aic, fit, (p,d,q)
                    except: pass
        arima_fc = best_arima.forecast(steps=n_test)
        mae = mean_absolute_error(test_ts, arima_fc)
        rmse = np.sqrt(mean_squared_error(test_ts, arima_fc))
        ts_results['ARIMA'] = {'MAE': mae, 'RMSE': rmse}
        
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("task", "time_series")
        mlflow.log_param("order", str(best_order))
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        # ARIMA isn't picklable with sklearn.log_model, use joblib artifact
        arima_path = os.path.join(tempfile.gettempdir(), "arima_model.pkl")
        joblib.dump(best_arima, arima_path)
        mlflow.log_artifact(arima_path, artifact_path="model")
        print(f"ARIMA {best_order}: MAE={mae:.4f}")
except Exception as e:
    print(f"ARIMA failed: {e}")

try:
    from prophet import Prophet
    with mlflow.start_run(run_name="TimeSeries_Prophet"):
        pdf = target.reset_index(); pdf.columns = ['ds','y']
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(pdf[:-n_test])
        future = m.make_future_dataframe(periods=n_test, freq='MS')
        fc = m.predict(future)
        prophet_pred = fc.tail(n_test)['yhat'].values
        mae = mean_absolute_error(test_ts, prophet_pred)
        rmse = np.sqrt(mean_squared_error(test_ts, prophet_pred))
        ts_results['Prophet'] = {'MAE': mae, 'RMSE': rmse}
        
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("task", "time_series")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        # Prophet uses pickle for serialization
        prophet_path = os.path.join(tempfile.gettempdir(), "prophet_model.pkl")
        joblib.dump(m, prophet_path)
        mlflow.log_artifact(prophet_path, artifact_path="model")
        print(f"Prophet: MAE={mae:.4f}")
except Exception as e:
    print(f"Prophet failed: {e}")


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
team_agg.to_csv(os.path.join(MODEL_DIR, 'team_agg.csv'), index=False)
joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'clf_feature_cols.pkl'))
print("[OK] All models saved")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
best_clf = max(clf_results, key=lambda x: clf_results[x]['ROC-AUC'])
best_reg_name = max(reg_results, key=lambda x: reg_results[x]['R2'])
print(f"Best Classifier: {best_clf} (AUC={clf_results[best_clf]['ROC-AUC']:.4f})")
print(f"Best Regressor: {best_reg_name} (R2={reg_results[best_reg_name]['R2']:.4f})")
print(f"\nMLflow tracking: http://localhost:5001")