"""
preprocessing.py
----------------
Downloads free football CSV data from football-data.co.uk (no account needed),
cleans it, encodes labels, and saves a clean Parquet file for feature engineering.

Leagues downloaded:
  E0  — English Premier League
  SP1 — Spanish La Liga
  D1  — German Bundesliga
  I1  — Italian Serie A
  F1  — French Ligue 1

Seasons: 2009/10 → 2015/16  (7 seasons, ~12,000+ matches)
"""

import os
import logging
import time

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_DIR  = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "processed", "matches_clean.parquet")

# ---------------------------------------------------------------------------
# Download config
# ---------------------------------------------------------------------------

LEAGUES = {
    "E0":  "Premier League",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
}

# Season codes used in football-data.co.uk URLs  e.g. "0910" = 2009/10
SEASONS = ["0910", "1011", "1112", "1213", "1314", "1415", "1516"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Columns we actually need from each CSV
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full-time goals + result
    "B365H", "B365D", "B365A",      # Bet365 odds (optional, used for calibration)
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_all(raw_dir: str = RAW_DIR) -> list[str]:
    """Download all league×season CSVs; skip files already on disk."""
    os.makedirs(raw_dir, exist_ok=True)
    paths = []

    for season in SEASONS:
        for league_code in LEAGUES:
            filename = f"{league_code}_{season}.csv"
            local    = os.path.join(raw_dir, filename)

            if os.path.exists(local):
                log.info("Already cached: %s", filename)
                paths.append(local)
                continue

            url = BASE_URL.format(season=season, league=league_code)
            log.info("Downloading %s …", url)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with open(local, "wb") as f:
                    f.write(resp.content)
                paths.append(local)
                time.sleep(0.3)          # be polite to the server
            except requests.HTTPError as e:
                log.warning("Skipped %s — %s", filename, e)
            except requests.RequestException as e:
                log.error("Failed to download %s: %s", filename, e)

    log.info("Downloaded / cached %d CSV files", len(paths))
    return paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _season_label(season_code: str) -> str:
    """'1516' → '2015/16'"""
    y1, y2 = "20" + season_code[:2], "20" + season_code[2:]
    return f"{y1}/{y2}"


def load_all_csvs(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Read every downloaded CSV and concatenate into one DataFrame."""
    dfs = []
    for season in SEASONS:
        for league_code, league_name in LEAGUES.items():
            path = os.path.join(raw_dir, f"{league_code}_{season}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, usecols=lambda c: c in KEEP_COLS,
                                 encoding="latin-1")
                df["league"]       = league_name
                df["league_code"]  = league_code
                df["season"]       = _season_label(season)
                df["season_year"]  = int("20" + season[:2])
                dfs.append(df)
            except Exception as e:
                log.warning("Could not read %s: %s", path, e)

    if not dfs:
        raise RuntimeError("No CSV files loaded. Check your internet connection and re-run.")

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Combined raw data: %d rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    # Parse dates — football-data uses DD/MM/YY or DD/MM/YYYY
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date"])
    df = df.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    # Rename to consistent internal names
    df = df.rename(columns={
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG":     "home_goals",
        "FTAG":     "away_goals",
        "FTR":      "ftr",
    })

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Target: 0=Home Win, 1=Draw, 2=Away Win
    result_map = {"H": 0, "D": 1, "A": 2}
    df["result"] = df["ftr"].map(result_map)
    df = df.dropna(subset=["result"])
    df["result"] = df["result"].astype(int)

    # Encode team names → integers (consistent across the whole dataset)
    le_team = LabelEncoder()
    all_teams = pd.concat([df["home_team"], df["away_team"]])
    le_team.fit(all_teams)
    df["home_team_id"] = le_team.transform(df["home_team"])
    df["away_team_id"] = le_team.transform(df["away_team"])

    # Encode league
    le_league = LabelEncoder()
    df["league_enc"] = le_league.fit_transform(df["league"])

    # Assign a stable match_id
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    log.info("Cleaned: %d → %d rows (dropped %d)", n_before, len(df), n_before - len(df))
    _print_class_dist(df)
    return df


def _print_class_dist(df: pd.DataFrame):
    counts = df["result"].value_counts().sort_index()
    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    log.info("Class distribution:")
    for k, v in counts.items():
        log.info("  %s: %d (%.1f%%)", labels[k], v, 100 * v / len(df))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_dir: str = RAW_DIR, out_path: str = OUT_PATH) -> pd.DataFrame:
    log.info("=== PREPROCESSING START ===")
    download_all(raw_dir)
    raw    = load_all_csvs(raw_dir)
    clean_ = clean(raw)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_.to_parquet(out_path, index=False)
    log.info("Saved → %s  (%d rows, %d cols)", out_path, *clean_.shape)
    log.info("=== PREPROCESSING DONE ===")
    return clean_


if __name__ == "__main__":
    run_preprocessing()
