"""
visualize.py
------------
Generates all plots for the project and saves them to outputs/plots/.

Plots produced:
  1. Training & validation loss curves (MLP, LSTM if available)
  2. Confusion matrices (raw + normalised) for each model
  3. Feature importance (Random Forest)
  4. Prediction probability distributions
  5. Calibration curves
  6. Class distribution bar chart
  7. Model comparison (grouped bar chart)

Usage:
  python src/visualize.py
"""

import json
import logging
import os
import pickle

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
import torch

from model import FootballMLP, load_mlp
from evaluate import _load_baseline, _load_mlp, CLASS_NAMES
from train import load_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PLOTS_DIR  = os.path.join("outputs", "plots")
MODELS_DIR = os.path.join("outputs", "models")
REPORTS_DIR = os.path.join("outputs", "reports")

PALETTE    = {"Home Win": "#2196F3", "Draw": "#FF9800", "Away Win": "#F44336"}
STYLE      = "seaborn-v0_8-whitegrid"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})


def _save(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# 1. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves():
    hist_path = os.path.join(MODELS_DIR, "mlp_history.pkl")
    if not os.path.exists(hist_path):
        log.warning("MLP history not found, skipping loss curve plot.")
        return

    with open(hist_path, "rb") as f:
        history = pickle.load(f)

    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    epochs     = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1565C0", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   color="#E53935", linewidth=2, linestyle="--")
    ax.set_title("MLP Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    best_ep = int(np.argmin(val_loss)) + 1
    ax.axvline(best_ep, color="grey", linestyle=":", alpha=0.7,
               label=f"Best epoch ({best_ep})")
    ax.legend()
    _save(fig, "loss_curves_mlp.png")

    # LSTM (if exists)
    lstm_path = os.path.join(MODELS_DIR, "lstm_history.pkl")
    if os.path.exists(lstm_path):
        with open(lstm_path, "rb") as f:
            lstm_hist = pickle.load(f)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ep2 = range(1, len(lstm_hist["train_loss"]) + 1)
        ax2.plot(ep2, lstm_hist["train_loss"], label="Train Loss", color="#1565C0", linewidth=2)
        ax2.plot(ep2, lstm_hist["val_loss"],   label="Val Loss",   color="#E53935", linewidth=2, linestyle="--")
        ax2.set_title("LSTM Training & Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cross-Entropy Loss")
        ax2.legend()
        _save(fig2, "loss_curves_lstm.png")


# ---------------------------------------------------------------------------
# 2. Confusion matrices
# ---------------------------------------------------------------------------

def _plot_cm(cm: np.ndarray, title: str, filename: str, normalise: bool = False):
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="lightgrey",
        vmin=0, vmax=vmax, ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    _save(fig, filename)


def plot_confusion_matrices():
    report_path = os.path.join(REPORTS_DIR, "metrics_report.json")
    if not os.path.exists(report_path):
        log.warning("Metrics report not found; run evaluate.py first.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    for key in ("random_forest", "mlp", "lstm", "naive"):
        if key not in report:
            continue
        cm = np.array(report[key]["confusion_matrix"])
        label = {"random_forest": "Random Forest", "mlp": "Deep MLP",
                 "lstm": "LSTM", "naive": "Naive Baseline"}.get(key, key)
        _plot_cm(cm, f"Confusion Matrix — {label}",
                 f"cm_{key}.png", normalise=False)
        _plot_cm(cm, f"Normalised Confusion Matrix — {label}",
                 f"cm_{key}_norm.png", normalise=True)


# ---------------------------------------------------------------------------
# 3. Feature importance (Random Forest)
# ---------------------------------------------------------------------------

def plot_feature_importance(top_n: int = 20):
    X_train, y_train, X_test, y_test, feat_cols, _ = load_split()

    baseline = _load_baseline()
    clf = baseline.pipeline.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        log.warning("Baseline model has no feature_importances_; skipping.")
        return

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    ax.barh(
        [feat_cols[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors,
    )
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance (Gini)")
    ax.invert_yaxis()
    _save(fig, "feature_importance_rf.png")


# ---------------------------------------------------------------------------
# 4. Prediction probability distributions
# ---------------------------------------------------------------------------

def plot_probability_distributions():
    X_train, y_train, X_test, y_test, feat_cols, _ = load_split()
    n_features = len(feat_cols)

    mlp = _load_mlp(n_features)
    X_test_t = torch.from_numpy(X_test)
    proba = mlp.predict_proba(X_test_t)  # (n, 3)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for i, (cls, ax) in enumerate(zip(CLASS_NAMES, axes)):
        ax.hist(proba[:, i], bins=40, color=list(PALETTE.values())[i],
                edgecolor="white", alpha=0.85)
        ax.set_title(f"P({cls})")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count" if i == 0 else "")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("MLP — Predicted Probability Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "probability_distributions.png")


# ---------------------------------------------------------------------------
# 5. Calibration curves
# ---------------------------------------------------------------------------

def plot_calibration_curves():
    report_path = os.path.join(REPORTS_DIR, "metrics_report.json")
    if not os.path.exists(report_path):
        log.warning("Run evaluate.py first to generate calibration data.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    cal_data = report.get("calibration", {})
    if not cal_data:
        return

    models_to_plot = [m for m in ("random_forest", "mlp") if m in cal_data]
    labels_map = {"random_forest": "Random Forest", "mlp": "Deep MLP"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, cls in enumerate(CLASS_NAMES):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        for model_key in models_to_plot:
            cal = cal_data[model_key].get(cls, {})
            if cal:
                ax.plot(
                    cal["mean_pred"], cal["frac_pos"],
                    marker="o", markersize=4,
                    label=labels_map[model_key],
                )
        ax.set_title(f"Calibration — {cls}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives" if i == 0 else "")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration Curves (One-vs-Rest)", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "calibration_curves.png")


# ---------------------------------------------------------------------------
# 6. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution():
    features_path = os.path.join("data", "processed", "features.parquet")
    if not os.path.exists(features_path):
        log.warning("features.parquet not found; skipping class distribution plot.")
        return

    df = pd.read_parquet(features_path)
    counts = df["result"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        CLASS_NAMES,
        [counts.get(i, 0) for i in range(3)],
        color=list(PALETTE.values()),
        edgecolor="white",
        width=0.6,
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 50,
            f"{height:,}\n({100*height/len(df):.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of Matches")
    ax.set_ylim(0, max(counts.values) * 1.18)
    _save(fig, "class_distribution.png")


# ---------------------------------------------------------------------------
# 7. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison():
    comparison_path = os.path.join(REPORTS_DIR, "comparison_table.csv")
    if not os.path.exists(comparison_path):
        log.warning("comparison_table.csv not found; run evaluate.py first.")
        return

    df = pd.read_csv(comparison_path)
    # Keep only the main metrics for the chart
    metric_cols = ["F1 Home Win", "F1 Draw", "F1 Away Win", "Macro F1"]
    df_float = df[metric_cols].astype(float)

    x = np.arange(len(metric_cols))
    width = 0.8 / len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - len(df) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_float.iloc[i], width * 0.9,
                      label=row["Model"], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_plots():
    log.info("=== GENERATING ALL PLOTS ===")
    with plt.style.context(STYLE):
        plot_loss_curves()
        plot_class_distribution()
        plot_confusion_matrices()
        plot_feature_importance()
        plot_probability_distributions()
        plot_calibration_curves()
        plot_model_comparison()
    log.info("=== ALL PLOTS SAVED TO %s ===", PLOTS_DIR)


if __name__ == "__main__":
    run_all_plots()
