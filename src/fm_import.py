"""
fm_import.py
------------
Football Manager Data Import Pipeline.

Football Manager allows you to export squad/player data and match results
as HTML or CSV files from within the game.  This module reads those exports,
maps FM columns to our feature schema, and produces a prediction-ready
feature vector for any match.

HOW TO EXPORT FROM FOOTBALL MANAGER
-------------------------------------
1. In FM, go to your Squad view (or any stats screen)
2. Right-click any column header → "Move to View"
   Make sure these columns are visible:
     - Team name, Matches Played, Wins, Draws, Losses
     - Goals Scored, Goals Conceded, Points
3. Click the small export icon (top-right of the screen) → "Export to HTML"
   OR: use the in-game keyboard shortcut  Ctrl+P  → Export
4. Save the file to:  data/fm_exports/<filename>.html
   (or .csv if your FM version supports CSV export)
5. Run:  python src/fm_import.py

WHAT THIS PRODUCES
------------------
- Parses your FM export into a clean DataFrame
- Computes per-team stats (form, goals, wins)
- Lets you call predict_fm_match(home_team, away_team) to get outcome probabilities
"""

import os
import glob
import logging
import pickle

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FM_EXPORT_DIR = os.path.join("data", "fm_exports")
MODELS_DIR    = os.path.join("outputs", "models")
CLASS_NAMES   = ["Home Win", "Draw", "Away Win"]

# ---------------------------------------------------------------------------
# Column mapping — FM export columns → our internal names
# (FM column names vary slightly by version; we try common variants)
# ---------------------------------------------------------------------------

FM_COLUMN_MAP = {
    # Team name
    "Name":          "team",
    "Club":          "team",
    "Team":          "team",

    # Matches played
    "MP":            "played",
    "Pld":           "played",
    "Matches":       "played",

    # Results
    "W":             "wins",
    "Wins":          "wins",
    "D":             "draws",
    "Draws":         "draws",
    "L":             "losses",
    "Lost":          "losses",
    "Losses":        "losses",

    # Goals
    "GF":            "goals_scored",
    "Goals":         "goals_scored",
    "GS":            "goals_scored",
    "GA":            "goals_conceded",
    "Goals Against": "goals_conceded",

    # Points
    "Pts":           "points",
    "Points":        "points",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_fm_export(path: str) -> pd.DataFrame:
    """
    Load a single FM export file (HTML or CSV).
    Tries HTML first (pandas read_html), then CSV.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".html", ".htm"):
        tables = pd.read_html(path, flavor="lxml")
        if not tables:
            raise ValueError(f"No tables found in {path}")
        # Pick the largest table (most likely the squad/stats table)
        df = max(tables, key=len)
    elif ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .html or .csv")

    log.info("Loaded FM export: %s (%d rows, %d cols)", path, *df.shape)
    return df


def normalise_fm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FM columns to our internal schema using FM_COLUMN_MAP."""
    rename = {}
    for col in df.columns:
        col_stripped = str(col).strip()
        if col_stripped in FM_COLUMN_MAP:
            rename[col] = FM_COLUMN_MAP[col_stripped]
    df = df.rename(columns=rename)

    # Keep only columns we know how to use (de-duplicated, order preserved)
    keep = []
    for c in FM_COLUMN_MAP.values():
        if c in df.columns and c not in keep:
            keep.append(c)
    missing = [c for c in ["team", "wins", "draws", "losses", "goals_scored", "goals_conceded"]
               if c not in df.columns]
    if missing:
        log.warning("Missing columns after mapping: %s", missing)
        log.warning("Available columns: %s", df.columns.tolist())

    # Drop any duplicate column names before selecting
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df


def clean_fm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns and drop rows with missing team names."""
    df = df.dropna(subset=["team"] if "team" in df.columns else df.columns[:1])
    df["team"] = df["team"].astype(str).str.strip()

    num_cols = [c for c in df.columns if c != "team"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.reset_index(drop=True)


def load_all_fm_exports(fm_dir: str = FM_EXPORT_DIR) -> pd.DataFrame | None:
    """
    Load and concatenate all FM export files found in fm_dir.
    Returns None if no files found.
    """
    os.makedirs(fm_dir, exist_ok=True)
    files = glob.glob(os.path.join(fm_dir, "*.html")) + \
            glob.glob(os.path.join(fm_dir, "*.htm"))  + \
            glob.glob(os.path.join(fm_dir, "*.csv"))

    if not files:
        log.warning(
            "No FM export files found in '%s'.\n"
            "Export your squad/league table from Football Manager and place\n"
            "the HTML or CSV file in that folder, then re-run.", fm_dir
        )
        return None

    dfs = []
    for f in files:
        try:
            raw = load_fm_export(f)
            clean = normalise_fm_columns(raw)
            clean = clean_fm_data(clean)
            if len(clean) > 0:
                dfs.append(clean)
        except Exception as e:
            log.warning("Could not parse %s: %s", f, e)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by team name — keep last occurrence (most recent export)
    combined = combined.drop_duplicates(subset=["team"], keep="last")
    log.info("FM data: %d teams loaded", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Feature extraction from FM data
# ---------------------------------------------------------------------------

def _get_team_stats(df: pd.DataFrame, team: str) -> dict | None:
    """
    Extract per-team stats from the FM DataFrame.
    Returns None if team not found.
    """
    row = df[df["team"].str.lower() == team.lower()]
    if len(row) == 0:
        # Try partial match
        row = df[df["team"].str.lower().str.contains(team.lower(), na=False)]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    played = float(row.get("played", 1)) or 1
    wins   = float(row.get("wins",   0))
    draws  = float(row.get("draws",  0))
    losses = float(row.get("losses", 0))
    gf     = float(row.get("goals_scored",    0))
    ga     = float(row.get("goals_conceded",  0))
    pts    = float(row.get("points", wins * 3 + draws))

    return {
        "form_wins":    wins   / played,
        "form_draws":   draws  / played,
        "form_losses":  losses / played,
        "form_gf":      gf     / played,
        "form_ga":      ga     / played,
        "form_gd":      (gf - ga) / played,
        "form_points":  pts    / (played * 3),
        "form_n":       int(played),
    }


def build_fm_feature_vector(
    fm_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    feat_cols: list[str],
    norm_stats: dict,
) -> np.ndarray | None:
    """
    Build a normalised feature vector for a given FM match-up.
    Returns None if either team is not found in the FM data.
    """
    home_stats = _get_team_stats(fm_df, home_team)
    away_stats = _get_team_stats(fm_df, away_team)

    if home_stats is None:
        log.error("Team '%s' not found in FM data.", home_team)
        return None
    if away_stats is None:
        log.error("Team '%s' not found in FM data.", away_team)
        return None

    feat = {
        "league_enc":      0,
        "season_year":     2024,
        "season_progress": 0.5,
    }
    for k, v in home_stats.items():
        feat[f"home_{k}"] = v
    for k, v in away_stats.items():
        feat[f"away_{k}"] = v

    # H2H — not available from FM league table; use neutral priors
    feat["h2h_home_wins"] = 0.45
    feat["h2h_draws"]     = 0.25
    feat["h2h_away_wins"] = 0.30
    feat["h2h_n"]         = 0

    # Venue strength — approximate from overall stats
    feat["home_avg_gf"]   = home_stats["form_gf"]
    feat["home_avg_ga"]   = home_stats["form_ga"]
    feat["home_win_rate"] = home_stats["form_wins"]
    feat["away_avg_gf"]   = away_stats["form_gf"]
    feat["away_avg_ga"]   = away_stats["form_ga"]
    feat["away_win_rate"] = away_stats["form_wins"]

    # Form differentials
    feat["diff_form_points"] = home_stats["form_points"] - away_stats["form_points"]
    feat["diff_form_gd"]     = home_stats["form_gd"]     - away_stats["form_gd"]
    feat["diff_form_wins"]   = home_stats["form_wins"]   - away_stats["form_wins"]

    # Build and normalise vector
    vec = np.array([feat.get(col, 0.0) for col in feat_cols], dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in norm_stats:
            mu, std = norm_stats[col]["mean"], norm_stats[col]["std"]
            vec[i]  = (vec[i] - mu) / (std if std > 1e-8 else 1.0)

    return vec


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_fm_match(
    home_team: str,
    away_team: str,
    fm_dir: str = FM_EXPORT_DIR,
    models_dir: str = MODELS_DIR,
) -> dict | None:
    """
    High-level function: load FM data + trained model, predict a match.

    Returns dict with keys:
      prediction, confidence, probabilities, home_team, away_team, source
    """
    # Load FM data
    fm_df = load_all_fm_exports(fm_dir)
    if fm_df is None:
        return None

    # Load model artefacts
    feat_path = os.path.join(models_dir, "feature_cols.pkl")
    norm_path = os.path.join(models_dir, "norm_stats.pkl")
    mlp_path  = os.path.join(models_dir, "mlp.pt")
    hist_path = os.path.join(models_dir, "mlp_history.pkl")

    for p in [feat_path, norm_path, mlp_path, hist_path]:
        if not os.path.exists(p):
            log.error("Model artefact not found: %s — run train.py first.", p)
            return None

    with open(feat_path, "rb") as f: feat_cols  = pickle.load(f)
    with open(norm_path, "rb") as f: norm_stats = pickle.load(f)
    with open(hist_path, "rb") as f: history    = pickle.load(f)

    hp = history["hparams"]
    from model import FootballMLP
    model = FootballMLP(
        n_features=len(feat_cols),
        hidden_sizes=hp["hidden_sizes"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
    model.eval()

    # Build feature vector
    vec = build_fm_feature_vector(fm_df, home_team, away_team, feat_cols, norm_stats)
    if vec is None:
        return None

    # Predict
    x     = torch.from_numpy(vec).unsqueeze(0)
    with torch.no_grad():
        proba = torch.softmax(model(x), dim=1).numpy()[0]

    pred_idx = int(np.argmax(proba))
    return {
        "home_team":     home_team,
        "away_team":     away_team,
        "prediction":    CLASS_NAMES[pred_idx],
        "confidence":    float(proba[pred_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, proba)},
        "source":        "Football Manager",
    }


def list_fm_teams(fm_dir: str = FM_EXPORT_DIR) -> list[str]:
    """Return list of team names available in FM exports."""
    fm_df = load_all_fm_exports(fm_dir)
    if fm_df is None or "team" not in fm_df.columns:
        return []
    return sorted(fm_df["team"].tolist())


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    teams = list_fm_teams()
    if not teams:
        print("\nNo FM data found.")
        print(f"Place your Football Manager HTML/CSV exports in: {FM_EXPORT_DIR}")
    else:
        print(f"\nTeams found in FM data ({len(teams)}):")
        for t in teams:
            print(f"  - {t}")

        if len(teams) >= 2:
            result = predict_fm_match(teams[0], teams[1])
            if result:
                print(f"\nExample prediction: {result['home_team']} vs {result['away_team']}")
                print(f"  Prediction : {result['prediction']}")
                print(f"  Confidence : {result['confidence']*100:.1f}%")
                for cls, p in result["probabilities"].items():
                    print(f"  {cls:<12}: {p*100:.1f}%")
