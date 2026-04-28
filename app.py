"""
app.py — Streamlit Prediction App (Phase 9 Bonus)
--------------------------------------------------
Interactive web app for predicting football match outcomes.

Users select a home team and away team, and the app computes all engineered
features from historical match data, then runs the trained MLP to output a
probability bar chart for Home Win / Draw / Away Win.

Run with:
  streamlit run app.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.model import FootballMLP
from src.features import (
    _team_rolling_stats,
    _h2h_stats,
    _venue_strength,
    FORM_WINDOW,
    H2H_WINDOW,
    VENUE_WINDOW,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="centered",
)

MODELS_DIR  = os.path.join("outputs", "models")
CLASS_NAMES  = ["Home Win", "Draw", "Away Win"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_meta():
    for fname in ["feature_cols.pkl", "norm_stats.pkl", "mlp.pt", "mlp_history.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            return None, None, None, f"Missing: outputs/models/{fname} — run the training pipeline first."

    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "norm_stats.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
        history = pickle.load(f)

    hp    = history["hparams"]
    model = FootballMLP(
        n_features=len(feat_cols),
        hidden_sizes=hp["hidden_sizes"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mlp.pt"), map_location="cpu"))
    model.eval()
    return model, feat_cols, norm_stats, None


@st.cache_data
def load_match_history():
    path = os.path.join("data", "processed", "matches_clean.parquet")
    if not os.path.exists(path):
        return None, "Run preprocessing.py first to generate match history."
    df = pd.read_parquet(path)
    return df, None


# ---------------------------------------------------------------------------
# Feature computation for one hypothetical match
# ---------------------------------------------------------------------------

def compute_features(matches: pd.DataFrame, home: str, away: str,
                     date: pd.Timestamp, league_enc: int, season_year: int) -> dict:
    home_form  = _team_rolling_stats(matches, home, date, FORM_WINDOW)
    away_form  = _team_rolling_stats(matches, away, date, FORM_WINDOW)
    h2h        = _h2h_stats(matches, home, away, date, H2H_WINDOW)
    home_venue = _venue_strength(matches, home, date, "home", VENUE_WINDOW)
    away_venue = _venue_strength(matches, away, date, "away", VENUE_WINDOW)

    feat = {
        "league_enc":      league_enc,
        "season_year":     season_year,
        "season_progress": 0.5,
    }
    for k, v in home_form.items():
        feat[f"home_{k}"] = v
    for k, v in away_form.items():
        feat[f"away_{k}"] = v
    feat.update(h2h)
    feat.update(home_venue)
    feat.update(away_venue)
    feat["diff_form_points"] = home_form["form_points"] - away_form["form_points"]
    feat["diff_form_gd"]     = home_form["form_gd"]     - away_form["form_gd"]
    feat["diff_form_wins"]   = home_form["form_wins"]   - away_form["form_wins"]
    return feat


def normalise_vec(feat_dict: dict, feat_cols: list, norm_stats: dict) -> np.ndarray:
    vec = np.array([feat_dict.get(col, 0.0) for col in feat_cols], dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in norm_stats:
            mu, std = norm_stats[col]["mean"], norm_stats[col]["std"]
            vec[i]  = (vec[i] - mu) / (std if std > 1e-8 else 1.0)
    return vec


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("⚽ Football Match Outcome Predictor")
    st.markdown(
        "Uses a trained **Deep MLP** to predict Home Win / Draw / Away Win "
        "probabilities from historical team statistics."
    )

    model, feat_cols, norm_stats, model_err = load_model_and_meta()
    matches, data_err = load_match_history()

    if model_err:
        st.error(model_err)
        st.code(
            "cd \"Football-prediction PAAI project\"\n"
            "python src/preprocessing.py\n"
            "python src/features.py\n"
            "python src/train.py --skip-optuna"
        )
        return

    if data_err:
        st.error(data_err)
        return

    all_teams   = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    all_leagues = sorted(matches["league"].unique())
    league_enc_map = (
        matches[["league", "league_enc"]].drop_duplicates()
        .set_index("league")["league_enc"].to_dict()
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", all_teams, index=0)
    with col2:
        away_default = min(1, len(all_teams) - 1)
        away_team = st.selectbox("✈️ Away Team", all_teams, index=away_default)

    league = st.selectbox("🌍 League", all_leagues)
    pred_date = st.date_input(
        "📅 Match Date (determines historical lookback)",
        value=pd.Timestamp("2016-04-01").date(),
    )

    st.markdown("---")
    if st.button("🔮 Predict Outcome", use_container_width=True):
        if home_team == away_team:
            st.warning("Please select two different teams.")
            return

        date       = pd.Timestamp(pred_date)
        league_enc = league_enc_map.get(league, 0)
        season_yr  = date.year if date.month >= 7 else date.year - 1

        with st.spinner("Computing features…"):
            feat_dict = compute_features(
                matches, home_team, away_team, date, league_enc, season_yr
            )
            feat_vec  = normalise_vec(feat_dict, feat_cols, norm_stats)

        x     = torch.from_numpy(feat_vec).unsqueeze(0)
        proba = model.predict_proba(x)[0]

        pred_cls   = int(np.argmax(proba))
        pred_label = CLASS_NAMES[pred_cls]
        confidence = proba[pred_cls] * 100

        color_map = {"Home Win": "blue", "Draw": "orange", "Away Win": "red"}
        st.markdown(f"### Prediction: **:{color_map[pred_label]}[{pred_label}]**")
        st.markdown(f"Confidence: **{confidence:.1f}%**")

        # Probability bar chart
        st.markdown("#### Outcome Probabilities")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        bars = ax.barh(CLASS_NAMES, [float(p) for p in proba],
                       color=CLASS_COLORS, edgecolor="white")
        for bar, prob in zip(bars, proba):
            ax.text(
                min(bar.get_width() + 0.02, 1.05),
                bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=12, fontweight="bold",
            )
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Probability")
        ax.set_title(f"{home_team}  vs  {away_team}", fontsize=13)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Recent form summary
        st.markdown("#### Recent Form (last 5 matches)")
        col_h, col_a = st.columns(2)
        with col_h:
            st.metric("Home team points/match",
                      f"{feat_dict['home_form_points']:.2f}")
            st.metric("Home team avg goal diff",
                      f"{feat_dict['home_form_gd']:+.2f}")
        with col_a:
            st.metric("Away team points/match",
                      f"{feat_dict['away_form_points']:.2f}")
            st.metric("Away team avg goal diff",
                      f"{feat_dict['away_form_gd']:+.2f}")

        with st.expander("Full feature values"):
            st.json({k: round(float(v), 4) for k, v in feat_dict.items()})

    # Sidebar
    with st.sidebar:
        st.header("Model Info")
        if model is not None:
            with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
                h = pickle.load(f)
            hp = h["hparams"]
            st.write("**Architecture:**", " → ".join(str(x) for x in hp["hidden_sizes"]))
            st.write("**Dropout:**", hp["dropout"])
            st.write("**Learning rate:**", hp["lr"])
            st.write("**Epochs trained:**", len(h["train_loss"]))
            st.write("**Best val loss:**", f"{min(h['val_loss']):.4f}")
        st.markdown("---")
        st.markdown("**Data:** football-data.co.uk")
        st.markdown("**Leagues:** PL, La Liga, Bundesliga, Serie A, Ligue 1")
        st.markdown("**Seasons:** 2009/10 – 2015/16")
        st.markdown("**Test set:** 2015/2016 season")


if __name__ == "__main__":
    main()
