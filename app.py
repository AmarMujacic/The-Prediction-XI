"""
app.py — Streamlit Prediction App
----------------------------------
Two tabs:
  1. Historical Data  — predict from football-data.co.uk historical stats
  2. Football Manager — predict from your FM export files

Models available: Deep MLP, TabPFN, Random Forest

Run with:
  streamlit run app.py
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, "src")

import shap

from model import FootballMLP
from features import (
    _team_rolling_stats,
    _h2h_stats,
    _venue_strength,
    FORM_WINDOW,
    H2H_WINDOW,
    VENUE_WINDOW,
)
from fm_import import (
    load_all_fm_exports,
    build_fm_feature_vector,
    list_fm_teams,
    FM_EXPORT_DIR,
)
from shap_explain import shap_for_prediction
from elo import get_team_elo, elo_tier, INITIAL_ELO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Prediction XI",
    page_icon="⚽",
    layout="centered",
)

MODELS_DIR   = os.path.join("outputs", "models")
CLASS_NAMES  = ["Home Win", "Draw", "Away Win"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]
LOG_PATH     = os.path.join("outputs", "reports", "prediction_log.csv")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_mlp_and_meta():
    for fname in ["feature_cols.pkl", "norm_stats.pkl", "mlp.pt", "mlp_history.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            return None, None, None, f"Missing: outputs/models/{fname}"
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
        feat_cols = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "norm_stats.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
        history = pickle.load(f)
    hp = history["hparams"]
    model = FootballMLP(n_features=len(feat_cols),
                        hidden_sizes=hp["hidden_sizes"], dropout=hp["dropout"])
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mlp.pt"), map_location="cpu"))
    model.eval()
    return model, feat_cols, norm_stats, None


@st.cache_resource
def load_tabpfn():
    path = os.path.join(MODELS_DIR, "tabpfn.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_baseline():
    path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_match_history():
    path = os.path.join("data", "processed", "matches_clean.parquet")
    if not os.path.exists(path):
        return None, "Run preprocessing.py first."
    return pd.read_parquet(path), None


@st.cache_data
def load_elo_ratings() -> dict:
    path = os.path.join(MODELS_DIR, "elo_final.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------

LOG_COLUMNS = [
    "timestamp", "home_team", "away_team", "model",
    "prediction", "p_home_win", "p_draw", "p_away_win",
    "confidence", "source", "actual_result",
]


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_prediction(home_team: str, away_team: str, model: str,
                    proba: np.ndarray, source: str = "Historical"):
    """Append one prediction row to the CSV log."""
    pred_cls = int(np.argmax(proba))
    row = {
        "timestamp":   pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "home_team":   home_team,
        "away_team":   away_team,
        "model":       model,
        "prediction":  CLASS_NAMES[pred_cls],
        "p_home_win":  round(float(proba[0]), 4),
        "p_draw":      round(float(proba[1]), 4),
        "p_away_win":  round(float(proba[2]), 4),
        "confidence":  round(float(proba[pred_cls]), 4),
        "source":      source,
        "actual_result": "",       # user can fill in later
    }
    df = load_log()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def show_prediction_log():
    """Render the full prediction history tab."""
    st.subheader("Prediction History Log")
    df = load_log()

    if df.empty:
        st.info("No predictions made yet. Go to the Historical Data or Football Manager tab and make a prediction.")
        return

    # ---- Summary stats ----
    total = len(df)
    filled = df[df["actual_result"].notna() & (df["actual_result"] != "")]
    accuracy = None
    if len(filled) > 0:
        correct = (filled["prediction"] == filled["actual_result"]).sum()
        accuracy = correct / len(filled) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", total)
    col2.metric("With Actual Result", len(filled))
    col3.metric("Accuracy", f"{accuracy:.1f}%" if accuracy is not None else "N/A")
    col4.metric("Most Used Model", df["model"].mode()[0] if not df.empty else "—")

    st.markdown("---")

    # ---- Result distribution pie ----
    if total > 0:
        pred_counts = df["prediction"].value_counts()
        col_a, col_b = st.columns([1, 2])

        with col_a:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            colors_pie = [CLASS_COLORS[CLASS_NAMES.index(c)]
                          if c in CLASS_NAMES else "#999"
                          for c in pred_counts.index]
            ax.pie(pred_counts.values, labels=pred_counts.index,
                   colors=colors_pie, autopct="%1.0f%%",
                   startangle=90, textprops={"fontsize": 10})
            ax.set_title("Predicted outcomes", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            # Avg confidence per predicted outcome
            avg_conf = df.groupby("prediction")["confidence"].mean().reset_index()
            avg_conf.columns = ["Outcome", "Avg Confidence"]
            avg_conf["Avg Confidence"] = (avg_conf["Avg Confidence"] * 100).round(1)
            avg_conf = avg_conf.sort_values("Avg Confidence", ascending=False)

            fig2, ax2 = plt.subplots(figsize=(4.5, 3))
            bar_colors = [CLASS_COLORS[CLASS_NAMES.index(c)]
                          if c in CLASS_NAMES else "#999"
                          for c in avg_conf["Outcome"]]
            ax2.bar(avg_conf["Outcome"], avg_conf["Avg Confidence"],
                    color=bar_colors, edgecolor="white")
            ax2.set_ylabel("Avg Confidence (%)")
            ax2.set_title("Avg confidence per outcome", fontsize=11)
            ax2.set_ylim(0, 100)
            for i, v in enumerate(avg_conf["Avg Confidence"]):
                ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.markdown("---")

    # ---- Editable table — user can enter actual results ----
    st.markdown("**Full Log** — enter actual results in the last column to track accuracy:")

    result_options = ["", "Home Win", "Draw", "Away Win"]

    display_df = df.copy()
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"

    # Colour-code prediction column
    def _color_pred(val):
        colors_map = {"Home Win": "background-color:#E3F2FD",
                      "Draw": "background-color:#FFF8E1",
                      "Away Win": "background-color:#FFEBEE"}
        return colors_map.get(val, "")

    styled = display_df.style.applymap(_color_pred, subset=["prediction"])
    st.dataframe(styled, use_container_width=True, height=350)

    # Allow updating actual results
    with st.expander("Enter actual result for a prediction"):
        if len(df) > 0:
            row_idx = st.selectbox(
                "Select prediction",
                df.index,
                format_func=lambda i: (
                    f"{df.loc[i,'timestamp']} | "
                    f"{df.loc[i,'home_team']} vs {df.loc[i,'away_team']} "
                    f"→ predicted {df.loc[i,'prediction']}"
                ),
            )
            actual = st.selectbox("Actual result", result_options[1:], key="actual_sel")
            if st.button("Save actual result", key="save_actual"):
                df.loc[row_idx, "actual_result"] = actual
                df.to_csv(LOG_PATH, index=False)
                st.success("Saved!")
                st.rerun()

    # Download button
    st.download_button(
        label="Download log as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_log.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Odds comparison
# ---------------------------------------------------------------------------

def get_bet365_odds(matches: pd.DataFrame, home_team: str,
                    away_team: str, date: pd.Timestamp) -> dict | None:
    """
    Look up Bet365 odds for the closest matching fixture to the given date.
    Returns implied probabilities (normalised to sum to 1) or None if not found.
    """
    mask = (
        (matches["home_team"] == home_team) &
        (matches["away_team"] == away_team) &
        matches["B365H"].notna() &
        matches["B365D"].notna() &
        matches["B365A"].notna()
    )
    candidates = matches[mask].copy()
    if candidates.empty:
        return None

    # Pick the row closest in date to the requested date
    candidates["date_diff"] = (candidates["date"] - date).abs()
    row = candidates.sort_values("date_diff").iloc[0]

    raw_h = float(row["B365H"])
    raw_d = float(row["B365D"])
    raw_a = float(row["B365A"])

    # Convert decimal odds → raw implied probs
    imp_h = 1.0 / raw_h
    imp_d = 1.0 / raw_d
    imp_a = 1.0 / raw_a
    total = imp_h + imp_d + imp_a   # > 1.0 due to bookmaker margin (overround)
    margin = (total - 1.0) * 100

    return {
        "odds":     {"Home Win": raw_h, "Draw": raw_d, "Away Win": raw_a},
        "implied":  {
            "Home Win": imp_h / total,
            "Draw":     imp_d / total,
            "Away Win": imp_a / total,
        },
        "margin_pct": margin,
        "match_date": pd.Timestamp(row["date"]).strftime("%d %b %Y"),
    }


def show_odds_comparison(proba: np.ndarray, odds_data: dict,
                         home_team: str, away_team: str):
    """
    Side-by-side bar chart: our model probabilities vs Bet365 implied probabilities.
    Highlights outcomes where our model disagrees significantly (potential value).
    """
    implied = [
        odds_data["implied"]["Home Win"],
        odds_data["implied"]["Draw"],
        odds_data["implied"]["Away Win"],
    ]
    model_proba = [float(p) for p in proba]
    diffs       = [m - b for m, b in zip(model_proba, implied)]

    st.markdown("#### Our Model vs Bet365 Market")
    st.caption(
        f"Odds from closest historical fixture ({odds_data['match_date']}) · "
        f"Bookmaker margin: {odds_data['margin_pct']:.1f}%"
    )

    # Grouped bar chart
    x      = np.arange(3)
    width  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - width/2, [p*100 for p in model_proba],
                   width, label="Our Model", color=CLASS_COLORS, alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width/2, [p*100 for p in implied],
                   width, label="Bet365 Implied", color=["#90CAF9","#FFCC80","#EF9A9A"],
                   edgecolor="white")

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{home_team} vs {away_team}", fontsize=12)
    ax.legend()
    ax.set_ylim(0, max(max(model_proba), max(implied)) * 120)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Value gap table
    st.markdown("**Where our model disagrees with the market:**")
    cols = st.columns(3)
    for i, (cls, diff, odds_val) in enumerate(zip(
        CLASS_NAMES, diffs,
        [odds_data["odds"]["Home Win"],
         odds_data["odds"]["Draw"],
         odds_data["odds"]["Away Win"]]
    )):
        with cols[i]:
            arrow = "🟢 Overpriced" if diff > 0.05 else ("🔴 Underpriced" if diff < -0.05 else "🟡 Fair")
            delta_str = f"{diff*100:+.1f}%"
            st.metric(
                label=f"{cls}",
                value=f"Odds: {odds_val:.2f}",
                delta=delta_str,
            )
            st.caption(arrow)

    st.caption(
        "🟢 Our model gives higher probability than the market suggests — potential value.\n"
        "🔴 Market more confident than our model.\n"
        "🟡 Model and market broadly agree (gap < 5%)."
    )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def compute_historical_features(matches, home, away, date, league_enc, season_year):
    home_form  = _team_rolling_stats(matches, home, date, FORM_WINDOW)
    away_form  = _team_rolling_stats(matches, away, date, FORM_WINDOW)
    h2h        = _h2h_stats(matches, home, away, date, H2H_WINDOW)
    home_venue = _venue_strength(matches, home, date, "home", VENUE_WINDOW)
    away_venue = _venue_strength(matches, away, date, "away", VENUE_WINDOW)

    feat = {"league_enc": league_enc, "season_year": season_year, "season_progress": 0.5}
    for k, v in home_form.items():  feat[f"home_{k}"] = v
    for k, v in away_form.items():  feat[f"away_{k}"] = v
    feat.update(h2h)
    feat.update(home_venue)
    feat.update(away_venue)
    feat["diff_form_points"] = home_form["form_points"] - away_form["form_points"]
    feat["diff_form_gd"]     = home_form["form_gd"]     - away_form["form_gd"]
    feat["diff_form_wins"]   = home_form["form_wins"]   - away_form["form_wins"]
    return feat


def normalise_vec(feat_dict, feat_cols, norm_stats):
    vec = np.array([feat_dict.get(col, 0.0) for col in feat_cols], dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in norm_stats:
            mu, std = norm_stats[col]["mean"], norm_stats[col]["std"]
            vec[i]  = (vec[i] - mu) / (std if std > 1e-8 else 1.0)
    return vec


def run_model(model_name, vec, mlp, tabpfn, baseline):
    if model_name == "Deep MLP" and mlp is not None:
        x = torch.from_numpy(vec).unsqueeze(0)
        with torch.no_grad():
            proba = torch.softmax(mlp(x), dim=1).numpy()[0]
    elif model_name == "TabPFN" and tabpfn is not None:
        proba = tabpfn.predict_proba(vec.reshape(1, -1))[0]
    elif model_name == "Random Forest" and baseline is not None:
        proba = baseline.predict_proba(vec.reshape(1, -1))[0]
    else:
        return None
    return proba


# ---------------------------------------------------------------------------
# SHAP explanation chart
# ---------------------------------------------------------------------------

def show_shap_explanation(vec: np.ndarray, feat_cols: list, pred_cls: int):
    """
    Show a horizontal bar chart of the top SHAP feature contributions
    for the predicted class. Green = pushed toward this outcome,
    Red = pushed away from this outcome.
    """
    try:
        shap_dict = shap_for_prediction(vec, feat_cols, top_n=10)
        class_name = CLASS_NAMES[pred_cls]
        features   = shap_dict[class_name]

        names  = [f for f, _ in features]
        values = [v for _, v in features]
        colors = ["#2E7D32" if v > 0 else "#C62828" for v in values]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Why {class_name}? — Top feature contributions", fontsize=12)
        ax.set_xlabel("SHAP value (impact on prediction)")
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "🟢 Green = feature pushed model toward this outcome  "
            "🔴 Red = feature pushed model away"
        )
    except Exception as e:
        st.caption(f"SHAP explanation unavailable: {e}")


# ---------------------------------------------------------------------------
# Head-to-Head history
# ---------------------------------------------------------------------------

def show_h2h_history(matches: pd.DataFrame, home_team: str, away_team: str, n: int = 10):
    """
    Display the last N head-to-head meetings between home_team and away_team
    as a visual timeline with scores and result colour coding.
    """
    mask = (
        ((matches["home_team"] == home_team) & (matches["away_team"] == away_team))
        | ((matches["home_team"] == away_team) & (matches["away_team"] == home_team))
    )
    h2h = matches[mask].sort_values("date", ascending=False).head(n)

    if h2h.empty:
        st.info("No historical H2H meetings found between these two teams.")
        return

    st.markdown(f"#### Last {len(h2h)} Meetings — {home_team} vs {away_team}")

    # Summary stats
    home_wins = draws = away_wins = 0
    home_gf = home_ga = 0

    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
        else:
            hg, ag = int(row["away_goals"]), int(row["home_goals"])
        home_gf += hg
        home_ga += ag
        if hg > ag:   home_wins += 1
        elif hg == ag: draws += 1
        else:          away_wins += 1

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"{home_team} wins", home_wins)
    col2.metric("Draws", draws)
    col3.metric(f"{away_team} wins", away_wins)
    col4.metric(f"Avg goals ({home_team})", f"{home_gf/len(h2h):.1f}")
    col5.metric(f"Avg goals ({away_team})", f"{home_ga/len(h2h):.1f}")

    st.markdown("---")

    # Visual timeline — one row per match
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            h_name = home_team
            a_name = away_team
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
        else:
            h_name = home_team
            a_name = away_team
            hg = int(row["away_goals"])
            ag = int(row["home_goals"])

        # Determine result from home_team perspective
        if hg > ag:
            bg_color = "#E3F2FD"   # blue — home win
            icon     = "🟦"
            label    = f"{home_team} won"
        elif hg == ag:
            bg_color = "#F5F5F5"   # grey — draw
            icon     = "🟨"
            label    = "Draw"
        else:
            bg_color = "#FFEBEE"   # red — away win
            icon     = "🟥"
            label    = f"{away_team} won"

        date_str = pd.Timestamp(row["date"]).strftime("%d %b %Y")

        st.markdown(
            f"""
            <div style="background:{bg_color};border-radius:10px;
                        padding:10px 16px;margin-bottom:6px;
                        display:flex;align-items:center;justify-content:space-between;">
                <span style="font-size:13px;color:#555">{date_str}</span>
                <span style="font-size:15px;font-weight:bold">
                    {h_name} &nbsp;
                    <span style="font-size:22px;color:#1565C0">{hg}</span>
                    &nbsp;–&nbsp;
                    <span style="font-size:22px;color:#C62828">{ag}</span>
                    &nbsp; {a_name}
                </span>
                <span style="font-size:13px">{icon} {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mini bar chart of results
    fig, ax = plt.subplots(figsize=(5, 1.2))
    total = len(h2h)
    ax.barh([0], [home_wins / total], color="#2196F3", height=0.5)
    ax.barh([0], [draws / total], left=[home_wins / total], color="#B0BEC5", height=0.5)
    ax.barh([0], [away_wins / total], left=[(home_wins + draws) / total],
            color="#F44336", height=0.5)
    ax.set_xlim(0, 1)
    ax.axis("off")
    # Labels
    if home_wins / total > 0.12:
        ax.text(home_wins / total / 2, 0, f"{home_wins}W",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    if draws / total > 0.12:
        ax.text(home_wins / total + draws / total / 2, 0, f"{draws}D",
                ha="center", va="center", fontsize=10, color="black", fontweight="bold")
    if away_wins / total > 0.12:
        ax.text((home_wins + draws) / total + away_wins / total / 2, 0, f"{away_wins}W",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"🟦 {home_team}  🟨 Draw  🟥 {away_team}")


# ---------------------------------------------------------------------------
# Elo rating display
# ---------------------------------------------------------------------------

def show_elo_ratings(home_team: str, away_team: str, elo_ratings: dict):
    """Display Elo ratings and tier badges for both teams."""
    if not elo_ratings:
        return

    home_elo  = get_team_elo(home_team, elo_ratings)
    away_elo  = get_team_elo(away_team, elo_ratings)
    home_tier = elo_tier(home_elo)
    away_tier = elo_tier(away_elo)

    tier_colors = {
        "Elite": "#FFD700", "Strong": "#2196F3",
        "Average": "#78909C", "Weak": "#FF9800", "Poor": "#F44336"
    }

    st.markdown("#### Elo Strength Ratings")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.metric(f"🏠 {home_team}", f"{home_elo:.0f}",
                  delta=f"{home_elo - INITIAL_ELO:+.0f} vs avg")
        color = tier_colors.get(home_tier, "#78909C")
        st.markdown(
            f"<span style='background:{color};color:black;padding:3px 10px;"
            f"border-radius:12px;font-weight:bold;font-size:13px'>{home_tier}</span>",
            unsafe_allow_html=True,
        )

    with col2:
        diff = home_elo - away_elo
        arrow = "🔴" if diff < -30 else ("🟢" if diff > 30 else "🟡")
        st.markdown(
            f"<div style='text-align:center;margin-top:20px'>"
            f"<b style='font-size:18px'>{arrow}</b><br>"
            f"<span style='font-size:11px;color:grey'>{abs(diff):.0f} pt gap</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.metric(f"✈️ {away_team}", f"{away_elo:.0f}",
                  delta=f"{away_elo - INITIAL_ELO:+.0f} vs avg")
        color = tier_colors.get(away_tier, "#78909C")
        st.markdown(
            f"<span style='background:{color};color:black;padding:3px 10px;"
            f"border-radius:12px;font-weight:bold;font-size:13px'>{away_tier}</span>",
            unsafe_allow_html=True,
        )

    # Elo bar comparison
    fig, ax = plt.subplots(figsize=(6, 1.2))
    total = home_elo + away_elo
    home_pct = home_elo / total
    ax.barh([0], [home_pct], color="#2196F3", height=0.5, label=home_team)
    ax.barh([0], [1 - home_pct], left=[home_pct], color="#F44336",
            height=0.5, label=away_team)
    ax.axvline(0.5, color="white", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared probability chart
# ---------------------------------------------------------------------------

def show_probability_chart(proba, home_team, away_team):
    pred_cls   = int(np.argmax(proba))
    pred_label = CLASS_NAMES[pred_cls]
    color_map  = {"Home Win": "blue", "Draw": "orange", "Away Win": "red"}

    st.markdown(f"### Prediction: **:{color_map[pred_label]}[{pred_label}]** "
                f"— {proba[pred_cls]*100:.1f}% confidence")

    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(CLASS_NAMES, [float(p) for p in proba],
                   color=CLASS_COLORS, edgecolor="white")
    for bar, prob in zip(bars, proba):
        ax.text(min(bar.get_width() + 0.02, 1.08),
                bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probability")
    ax.set_title(f"{home_team}  vs  {away_team}", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(mlp, tabpfn, baseline):
    with st.sidebar:
        st.header("The Prediction XI")
        st.markdown("Football match outcome predictor")
        st.markdown("---")

        # Model selector
        available = []
        if mlp      is not None: available.append("Deep MLP")
        if tabpfn   is not None: available.append("TabPFN")
        if baseline is not None: available.append("Random Forest")
        if not available:
            available = ["Deep MLP"]

        model_choice = st.selectbox("Model", available)

        st.markdown("---")
        st.markdown("**Data source:** football-data.co.uk")
        st.markdown("**Leagues:** PL, La Liga, Bundesliga, Serie A, Ligue 1")
        st.markdown("**Seasons:** 2009/10 – 2015/16")

        if mlp is not None:
            with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
                h = pickle.load(f)
            st.markdown("---")
            st.markdown("**MLP Info**")
            hp = h["hparams"]
            st.write("Architecture:", " → ".join(str(x) for x in hp["hidden_sizes"]))
            st.write("Best val loss:", f"{min(h['val_loss']):.4f}")
            st.write("Epochs:", len(h["train_loss"]))

    return model_choice


# ---------------------------------------------------------------------------
# Tab 1 — Historical Data
# ---------------------------------------------------------------------------

def tab_historical(mlp, feat_cols, norm_stats, tabpfn, baseline, model_choice):
    st.subheader("Predict from Historical Match Data")
    st.caption("Features computed from 2009–2015 match history")

    matches, data_err = load_match_history()
    if data_err:
        st.error(data_err)
        return

    all_teams   = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    all_leagues = sorted(matches["league"].unique())
    league_enc_map = (matches[["league", "league_enc"]].drop_duplicates()
                      .set_index("league")["league_enc"].to_dict())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", all_teams, key="hist_home")
    with col2:
        away_team = st.selectbox("Away Team", all_teams, index=1, key="hist_away")

    league    = st.selectbox("League", all_leagues, key="hist_league")
    pred_date = st.date_input("Match Date", value=pd.Timestamp("2016-04-01").date(),
                              key="hist_date")

    # Show Elo ratings as soon as teams are selected
    elo_ratings = load_elo_ratings()
    if elo_ratings:
        show_elo_ratings(home_team, away_team, elo_ratings)

    # Show H2H history
    if home_team != away_team:
        with st.expander("Head-to-Head History", expanded=True):
            show_h2h_history(matches, home_team, away_team)

    if st.button("Predict", use_container_width=True, key="hist_btn"):
        if home_team == away_team:
            st.warning("Select two different teams.")
            return
        date       = pd.Timestamp(pred_date)
        league_enc = league_enc_map.get(league, 0)
        season_yr  = date.year if date.month >= 7 else date.year - 1

        with st.spinner("Computing features…"):
            feat_dict = compute_historical_features(
                matches, home_team, away_team, date, league_enc, season_yr)
            vec = normalise_vec(feat_dict, feat_cols, norm_stats)

        proba = run_model(model_choice, vec, mlp, tabpfn, baseline)
        if proba is None:
            st.error(f"{model_choice} model not loaded.")
            return

        show_probability_chart(proba, home_team, away_team)
        save_prediction(home_team, away_team, model_choice, proba, source="Historical")
        st.caption("✅ Prediction saved to log")

        # Odds comparison
        odds_data = get_bet365_odds(matches, home_team, away_team, date)
        if odds_data:
            show_odds_comparison(proba, odds_data, home_team, away_team)
        else:
            st.caption("No Bet365 odds available for this fixture in the dataset.")

        col_h, col_a = st.columns(2)
        with col_h:
            st.metric("Home form (pts/game)", f"{feat_dict['home_form_points']:.2f}")
            st.metric("Home goal diff", f"{feat_dict['home_form_gd']:+.2f}")
        with col_a:
            st.metric("Away form (pts/game)", f"{feat_dict['away_form_points']:.2f}")
            st.metric("Away goal diff", f"{feat_dict['away_form_gd']:+.2f}")

        st.markdown("#### Why this prediction?")
        show_shap_explanation(vec, feat_cols, int(np.argmax(proba)))

        with st.expander("All features"):
            st.json({k: round(float(v), 4) for k, v in feat_dict.items()})


# ---------------------------------------------------------------------------
# Tab 2 — Football Manager
# ---------------------------------------------------------------------------

def tab_football_manager(mlp, feat_cols, norm_stats, tabpfn, baseline, model_choice):
    st.subheader("Predict Using Football Manager Data")
    st.caption("Import your FM export files to predict outcomes from your save game")

    st.info(
        "**How to export from Football Manager:**\n"
        "1. Go to your league table or squad view in FM\n"
        "2. Press **Ctrl+P** → Export as **HTML**\n"
        "3. Save the file to: `data/fm_exports/`\n"
        "4. Reload this page"
    )

    # Also allow direct file upload
    uploaded = st.file_uploader(
        "Or upload your FM export directly",
        type=["html", "htm", "csv"],
        key="fm_upload",
    )

    fm_df = None

    if uploaded is not None:
        import tempfile
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            from fm_import import load_fm_export, normalise_fm_columns, clean_fm_data
            raw  = load_fm_export(tmp_path)
            norm = normalise_fm_columns(raw)
            fm_df = clean_fm_data(norm)
            st.success(f"Loaded {len(fm_df)} teams from uploaded file.")
        except Exception as e:
            st.error(f"Could not parse file: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        fm_df = load_all_fm_exports(FM_EXPORT_DIR)

    if fm_df is None or len(fm_df) == 0:
        st.warning("No FM data found. Export from Football Manager and upload above, "
                   "or place files in `data/fm_exports/`.")
        return

    fm_teams = sorted(fm_df["team"].tolist())
    st.success(f"{len(fm_teams)} teams loaded from Football Manager data.")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", fm_teams, key="fm_home")
    with col2:
        away_team = st.selectbox("Away Team", fm_teams, index=min(1, len(fm_teams)-1),
                                 key="fm_away")

    # Show Elo ratings live as teams are selected
    elo_ratings = load_elo_ratings()
    if elo_ratings:
        show_elo_ratings(home_team, away_team, elo_ratings)

    if st.button("Predict (FM Data)", use_container_width=True, key="fm_btn"):
        if home_team == away_team:
            st.warning("Select two different teams.")
            return

        vec = build_fm_feature_vector(fm_df, home_team, away_team, feat_cols, norm_stats)
        if vec is None:
            st.error("Could not build features — check team names.")
            return

        proba = run_model(model_choice, vec, mlp, tabpfn, baseline)
        if proba is None:
            st.error(f"{model_choice} model not loaded.")
            return

        show_probability_chart(proba, home_team, away_team)
        save_prediction(home_team, away_team, model_choice, proba, source="Football Manager")
        st.caption("✅ Prediction saved to log")

        st.markdown("#### Why this prediction?")
        show_shap_explanation(vec, feat_cols, int(np.argmax(proba)))

        # Show FM raw stats
        with st.expander("FM team stats"):
            home_row = fm_df[fm_df["team"].str.lower() == home_team.lower()]
            away_row = fm_df[fm_df["team"].str.lower() == away_team.lower()]
            if not home_row.empty and not away_row.empty:
                combined = pd.concat([home_row, away_row]).set_index("team")
                st.dataframe(combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("⚽ The Prediction XI")
    st.markdown("Football match outcome prediction — Home Win / Draw / Away Win")

    mlp, feat_cols, norm_stats, model_err = load_mlp_and_meta()
    tabpfn  = load_tabpfn()
    baseline = load_baseline()

    if model_err:
        st.error(model_err)
        st.code("python src/preprocessing.py\n"
                "python src/features.py\n"
                "python src/train.py --skip-optuna")
        return

    model_choice = render_sidebar(mlp, tabpfn, baseline)

    tab1, tab2, tab3 = st.tabs(["Historical Data", "Football Manager", "Prediction Log"])

    with tab1:
        tab_historical(mlp, feat_cols, norm_stats, tabpfn, baseline, model_choice)

    with tab2:
        tab_football_manager(mlp, feat_cols, norm_stats, tabpfn, baseline, model_choice)

    with tab3:
        show_prediction_log()


if __name__ == "__main__":
    main()
