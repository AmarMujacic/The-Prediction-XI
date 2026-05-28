"""
elo.py
------
Elo rating system for football teams.

How Elo works:
  - Every team starts at 1500
  - After each match, ratings update based on expected vs actual result
  - Winning against a stronger team gains more points than beating a weak team
  - Draws shift ratings slightly toward equality

Formulas:
  Expected: E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
  Update:   R_A_new = R_A + K * (S_A - E_A)

  S_A = 1.0 (win), 0.5 (draw), 0.0 (loss)
  K   = 32 (standard football K-factor)

Outputs:
  - data/processed/elo_ratings.parquet  — Elo rating per team at each point in time
  - data/processed/elo_final.parquet    — Final Elo rating per team (latest snapshot)

Usage:
  python src/elo.py
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join("data", "processed", "matches_clean.parquet")
ELO_HISTORY    = os.path.join("data", "processed", "elo_ratings.parquet")
ELO_FINAL      = os.path.join("data", "processed", "elo_final.parquet")
MODELS_DIR     = os.path.join("outputs", "models")

INITIAL_ELO = 1500
K_FACTOR    = 32


# ---------------------------------------------------------------------------
# Core Elo math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float,
               score_a: float, k: float = K_FACTOR) -> tuple[float, float]:
    """
    Update Elo ratings for both teams after one match.

    score_a: 1.0 = A wins, 0.5 = draw, 0.0 = A loses
    Returns (new_rating_a, new_rating_b)
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - e_a)
    new_b = rating_b + k * (score_b - e_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Full history computation
# ---------------------------------------------------------------------------

def compute_elo_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Compute Elo ratings for every team across all matches in chronological order.

    Returns:
      history_df : DataFrame with columns [match_id, date, home_team, away_team,
                                           home_elo_before, away_elo_before,
                                           home_elo_after,  away_elo_after,
                                           elo_diff]
      final_elos : dict {team_name: final_elo_rating}
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Initialise all teams
    teams = set(df["home_team"]) | set(df["away_team"])
    elo   = {team: float(INITIAL_ELO) for team in teams}

    records = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = int(row["result"])  # 0=H, 1=D, 2=A

        r_home = elo[home]
        r_away = elo[away]

        score_home = 1.0 if result == 0 else (0.5 if result == 1 else 0.0)

        new_home, new_away = update_elo(r_home, r_away, score_home)

        records.append({
            "match_id":         row["match_id"],
            "date":             row["date"],
            "home_team":        home,
            "away_team":        away,
            "home_elo_before":  r_home,
            "away_elo_before":  r_away,
            "home_elo_after":   new_home,
            "away_elo_after":   new_away,
            "elo_diff":         r_home - r_away,   # positive = home stronger
        })

        elo[home] = new_home
        elo[away] = new_away

    history_df = pd.DataFrame(records)
    log.info("Elo history computed: %d rows", len(history_df))
    return history_df, elo


# ---------------------------------------------------------------------------
# Feature integration helpers
# ---------------------------------------------------------------------------

def get_elo_before_match(history_df: pd.DataFrame, match_id: int) -> dict:
    """Return home_elo_before and away_elo_before for a given match_id."""
    row = history_df[history_df["match_id"] == match_id]
    if row.empty:
        return {"home_elo": INITIAL_ELO, "away_elo": INITIAL_ELO, "elo_diff": 0.0}
    row = row.iloc[0]
    return {
        "home_elo": row["home_elo_before"],
        "away_elo": row["away_elo_before"],
        "elo_diff": row["elo_diff"],
    }


def get_team_elo(team: str, final_elos: dict) -> float:
    """Get the latest Elo for a team (case-insensitive partial match)."""
    # Exact match first
    if team in final_elos:
        return final_elos[team]
    # Case-insensitive
    lower = {k.lower(): v for k, v in final_elos.items()}
    if team.lower() in lower:
        return lower[team.lower()]
    # Partial match
    for k, v in lower.items():
        if team.lower() in k or k in team.lower():
            return v
    return float(INITIAL_ELO)


def elo_tier(rating: float) -> str:
    """Return a human-readable tier label for an Elo rating."""
    if rating >= 1700: return "Elite"
    if rating >= 1600: return "Strong"
    if rating >= 1500: return "Average"
    if rating >= 1400: return "Weak"
    return "Poor"


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_elo_history(team: str, history_df: pd.DataFrame,
                     save_path: str | None = None):
    """Plot a team's Elo rating over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    home_mask = history_df["home_team"] == team
    away_mask = history_df["away_team"] == team

    home_df = history_df[home_mask][["date", "home_elo_after"]].rename(
        columns={"home_elo_after": "elo"})
    away_df = history_df[away_mask][["date", "away_elo_after"]].rename(
        columns={"away_elo_after": "elo"})

    team_hist = pd.concat([home_df, away_df]).sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(team_hist["date"], team_hist["elo"],
            color="#1565C0", linewidth=2)
    ax.axhline(INITIAL_ELO, color="grey", linewidth=1,
               linestyle="--", alpha=0.6, label="Average (1500)")
    ax.fill_between(team_hist["date"], team_hist["elo"], INITIAL_ELO,
                    where=team_hist["elo"] > INITIAL_ELO,
                    alpha=0.15, color="#2E7D32")
    ax.fill_between(team_hist["date"], team_hist["elo"], INITIAL_ELO,
                    where=team_hist["elo"] < INITIAL_ELO,
                    alpha=0.15, color="#C62828")
    ax.set_title(f"Elo Rating History — {team}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Elo Rating")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        log.info("Saved Elo chart → %s", save_path)
    return fig


def plot_top_teams(final_elos: dict, top_n: int = 20, save_path: str | None = None):
    """Bar chart of highest-rated teams at end of dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = sorted(final_elos.items(), key=lambda x: x[1], reverse=True)[:top_n]
    teams, ratings = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1565C0" if r >= 1600 else "#42A5F5" for r in ratings]
    bars = ax.barh(list(teams)[::-1], list(ratings)[::-1],
                   color=list(colors)[::-1], edgecolor="white")
    ax.axvline(INITIAL_ELO, color="grey", linewidth=1,
               linestyle="--", alpha=0.7, label="Average (1500)")
    for bar, rating in zip(bars, list(ratings)[::-1]):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{rating:.0f}", va="center", fontsize=9)
    ax.set_title(f"Top {top_n} Teams by Final Elo Rating", fontsize=13)
    ax.set_xlabel("Elo Rating")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        log.info("Saved top teams chart → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_elo(processed_path: str = PROCESSED_PATH) -> tuple[pd.DataFrame, dict]:
    log.info("=== ELO COMPUTATION START ===")
    df = pd.read_parquet(processed_path)

    history_df, final_elos = compute_elo_history(df)

    # Save history
    history_df.to_parquet(ELO_HISTORY, index=False)
    log.info("Elo history saved → %s", ELO_HISTORY)

    # Save final ratings as DataFrame
    final_df = pd.DataFrame(
        list(final_elos.items()), columns=["team", "elo"]
    ).sort_values("elo", ascending=False).reset_index(drop=True)
    final_df.to_parquet(ELO_FINAL, index=False)
    log.info("Final Elo saved → %s", ELO_FINAL)

    # Save to models dir for app access
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "elo_final.pkl"), "wb") as f:
        pickle.dump(final_elos, f)

    # Print top 10
    log.info("Top 10 teams by final Elo:")
    for team, rating in sorted(final_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        log.info("  %-30s %.0f  [%s]", team, rating, elo_tier(rating))

    # Save plots
    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_top_teams(final_elos, top_n=20,
                   save_path=os.path.join(plots_dir, "elo_top_teams.png"))

    log.info("=== ELO DONE ===")
    return history_df, final_elos


if __name__ == "__main__":
    run_elo()
