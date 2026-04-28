"""
features.py
-----------
Builds the final feature matrix from the cleaned match DataFrame.

All features use ONLY information available BEFORE kick-off (no leakage).

Feature groups:
  1. Rolling form (last N matches)  — goals, wins, draws, losses, points
  2. Head-to-head statistics        — last N H2H meetings
  3. Home / Away venue strength     — goals scored/conceded at home vs away
  4. Season context                 — normalised stage, league encoding
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join("data", "processed", "matches_clean.parquet")
FEATURES_PATH  = os.path.join("data", "processed", "features.parquet")

FORM_WINDOW  = 5   # rolling window: last N matches for overall form
H2H_WINDOW   = 5   # last N head-to-head meetings
VENUE_WINDOW = 10  # last N home (or away) matches for venue strength


# ---------------------------------------------------------------------------
# Per-team rolling statistics (any side, before a given date)
# ---------------------------------------------------------------------------

def _team_rolling_stats(df: pd.DataFrame, team: str,
                        before_date: pd.Timestamp, n: int) -> dict:
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    recent = df[mask].sort_values("date").tail(n)

    if len(recent) == 0:
        return {
            "form_wins": 0.0, "form_draws": 0.0, "form_losses": 0.0,
            "form_gf": 0.0,   "form_ga": 0.0,    "form_gd": 0.0,
            "form_points": 0.0, "form_n": 0,
        }

    wins = draws = losses = gf_total = ga_total = 0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_goals"], row["away_goals"]
        else:
            gf, ga = row["away_goals"], row["home_goals"]
        gf_total += gf
        ga_total += ga
        if gf > ga:   wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1

    played = len(recent)
    return {
        "form_wins":    wins   / played,
        "form_draws":   draws  / played,
        "form_losses":  losses / played,
        "form_gf":      gf_total / played,
        "form_ga":      ga_total / played,
        "form_gd":      (gf_total - ga_total) / played,
        "form_points":  (wins * 3 + draws) / (played * 3),
        "form_n":       played,
    }


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def _h2h_stats(df: pd.DataFrame, home: str, away: str,
               before_date: pd.Timestamp, n: int) -> dict:
    mask = (
        (
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        )
        & (df["date"] < before_date)
    )
    meetings = df[mask].sort_values("date").tail(n)

    if len(meetings) == 0:
        return {"h2h_home_wins": 0.5, "h2h_draws": 0.0,
                "h2h_away_wins": 0.5, "h2h_n": 0}

    hw = draws = aw = 0
    for _, row in meetings.iterrows():
        if row["home_team"] == home:
            hg, ag = row["home_goals"], row["away_goals"]
        else:
            hg, ag = row["away_goals"], row["home_goals"]
        if hg > ag:    hw    += 1
        elif hg == ag: draws += 1
        else:          aw    += 1

    p = len(meetings)
    return {
        "h2h_home_wins": hw    / p,
        "h2h_draws":     draws / p,
        "h2h_away_wins": aw    / p,
        "h2h_n":         p,
    }


# ---------------------------------------------------------------------------
# Venue strength (home-only or away-only record)
# ---------------------------------------------------------------------------

def _venue_strength(df: pd.DataFrame, team: str,
                    before_date: pd.Timestamp, side: str, n: int) -> dict:
    """side must be 'home' or 'away'."""
    opp_side = "away" if side == "home" else "home"
    mask = (df[f"{side}_team"] == team) & (df["date"] < before_date)
    recent = df[mask].sort_values("date").tail(n)

    if len(recent) == 0:
        return {f"{side}_avg_gf": 0.0, f"{side}_avg_ga": 0.0,
                f"{side}_win_rate": 0.0}

    gf   = recent[f"{side}_goals"].mean()
    ga   = recent[f"{opp_side}_goals"].mean()
    wins = (recent[f"{side}_goals"] > recent[f"{opp_side}_goals"]).mean()
    return {
        f"{side}_avg_gf":   gf,
        f"{side}_avg_ga":   ga,
        f"{side}_win_rate": wins,
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building features for %d matches — this takes a few minutes…", len(df))
    df = df.sort_values("date").reset_index(drop=True)

    # Pre-compute max stage per season for season_progress
    max_stage = df.groupby("season")["season_year"].transform("count")  # fallback
    if "stage" in df.columns:
        max_stage = df.groupby("season")["stage"].transform("max")
        stage_col = df["stage"]
    else:
        # Derive a stage proxy: rank of match within season for each team
        df["stage"] = df.groupby("season").cumcount() + 1
        max_stage   = df.groupby("season")["stage"].transform("max")
        stage_col   = df["stage"]

    records = []
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            log.info("  %d / %d", idx, len(df))

        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_form  = _team_rolling_stats(df, home_team, date, FORM_WINDOW)
        away_form  = _team_rolling_stats(df, away_team, date, FORM_WINDOW)
        h2h        = _h2h_stats(df, home_team, away_team, date, H2H_WINDOW)
        home_venue = _venue_strength(df, home_team, date, "home", VENUE_WINDOW)
        away_venue = _venue_strength(df, away_team, date, "away", VENUE_WINDOW)

        sp = stage_col[idx] / max_stage[idx] if max_stage[idx] > 0 else 0.5

        feat = {
            "match_id":         row["match_id"],
            "date":             date,
            "result":           int(row["result"]),
            "league_enc":       int(row["league_enc"]),
            "season_year":      int(row["season_year"]),
            "season_progress":  float(sp),
        }
        for k, v in home_form.items():
            feat[f"home_{k}"] = v
        for k, v in away_form.items():
            feat[f"away_{k}"] = v
        feat.update(h2h)
        feat.update(home_venue)
        feat.update(away_venue)

        # Form differentials (home minus away) — capture relative strength
        feat["diff_form_points"] = home_form["form_points"] - away_form["form_points"]
        feat["diff_form_gd"]     = home_form["form_gd"]     - away_form["form_gd"]
        feat["diff_form_wins"]   = home_form["form_wins"]   - away_form["form_wins"]

        records.append(feat)

    features_df = pd.DataFrame(records)
    log.info("Feature matrix: %d rows × %d cols", *features_df.shape)
    return features_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"match_id", "date", "result"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Normalisation (fit on train only)
# ---------------------------------------------------------------------------

def normalise(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stats = {}
    train_out = train_df.copy()
    test_out  = test_df.copy()

    for col in feature_cols:
        mu  = train_df[col].mean()
        std = train_df[col].std(ddof=0)
        std = std if std > 1e-8 else 1.0
        train_out[col] = (train_df[col] - mu) / std
        test_out[col]  = (test_df[col]  - mu) / std
        stats[col] = {"mean": float(mu), "std": float(std)}

    return train_out, test_out, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_features(
    processed_path: str = PROCESSED_PATH,
    out_path: str = FEATURES_PATH,
) -> pd.DataFrame:
    log.info("=== FEATURE ENGINEERING START ===")
    df       = pd.read_parquet(processed_path)
    features = build_features(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)
    log.info("=== FEATURE ENGINEERING DONE ===")
    return features


if __name__ == "__main__":
    run_features()
