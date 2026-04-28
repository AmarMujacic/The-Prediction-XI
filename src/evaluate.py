"""
evaluate.py
-----------
Loads trained models and the test set, then produces:

  - Per-class precision, recall, F1
  - Macro and weighted averages
  - Confusion matrices (raw + normalised)
  - Calibration curves (are predicted probabilities reliable?)
  - Side-by-side comparison: Baseline vs MLP (vs LSTM if available)
  - Saved metrics JSON report

Usage:
  python src/evaluate.py
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from model import FootballMLP, FootballLSTM, load_mlp
from features import get_feature_columns
from train import load_split, build_lstm_sequences, LSTM_SEQ_LEN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR  = os.path.join("outputs", "models")
REPORTS_DIR = os.path.join("outputs", "reports")
CLASS_NAMES = ["Home Win", "Draw", "Away Win"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_baseline():
    path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_mlp(n_features: int) -> FootballMLP:
    path = os.path.join(MODELS_DIR, "mlp.pt")
    with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
        history = pickle.load(f)
    hp = history["hparams"]
    return load_mlp(path, n_features, hp["hidden_sizes"], hp["dropout"])


def _load_lstm(n_features: int) -> FootballLSTM | None:
    path = os.path.join(MODELS_DIR, "lstm.pt")
    if not os.path.exists(path):
        return None
    model = FootballLSTM(n_features=n_features)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute all metrics for a single model and return a structured dict."""
    acc   = accuracy_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    log.info("\n=== %s ===", name)
    log.info("Accuracy: %.4f", acc)
    log.info("\n%s", report_str)

    return {
        "name":        name,
        "accuracy":    float(acc),
        "confusion_matrix": cm.tolist(),
        "report":      report_dict,
        "report_str":  report_str,
        "y_true":      y_true,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }


def naive_baseline_metrics(y_true: np.ndarray) -> dict:
    """
    Naive 'always predict Home Win' baseline — useful to sanity-check that
    our models beat the trivially predictable majority class.
    """
    n = len(y_true)
    y_pred = np.zeros(n, dtype=np.int64)  # always Home Win
    proba  = np.zeros((n, 3), dtype=np.float32)
    proba[:, 0] = 1.0
    return evaluate_model("Naive (Always Home Win)", y_true, y_pred, proba)


def calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute calibration data for each class (one-vs-rest).
    Returns dict of {class_name: (fraction_of_positives, mean_predicted_value)}.
    """
    cal = {}
    for i, cls in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        frac, mean_pred = calibration_curve(
            binary_true, y_proba[:, i], n_bins=n_bins, strategy="quantile"
        )
        cal[cls] = {"frac_pos": frac.tolist(), "mean_pred": mean_pred.tolist()}
    return cal


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test, feat_cols, _ = load_split()
    n_features = len(feat_cols)
    device = torch.device("cpu")   # evaluation always on CPU for portability

    results = {}

    # ---- 1. Naive baseline ----
    results["naive"] = naive_baseline_metrics(y_test)

    # ---- 2. Random Forest baseline ----
    log.info("Evaluating Random Forest baseline…")
    baseline = _load_baseline()
    rf_pred  = baseline.predict(X_test)
    rf_proba = baseline.predict_proba(X_test)
    results["random_forest"] = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

    # ---- 3. MLP ----
    log.info("Evaluating MLP…")
    mlp      = _load_mlp(n_features)
    X_test_t = torch.from_numpy(X_test)
    mlp_proba = mlp.predict_proba(X_test_t)
    mlp_pred  = np.argmax(mlp_proba, axis=1)
    results["mlp"] = evaluate_model("Deep MLP", y_test, mlp_pred, mlp_proba)

    # ---- 4. LSTM (if available) ----
    lstm = _load_lstm(n_features)
    if lstm is not None:
        log.info("Evaluating LSTM…")
        # Build sequences from full test set; use all of X_train as "history"
        X_full  = np.concatenate([X_train, X_test], axis=0)
        y_full  = np.concatenate([y_train, y_test], axis=0)
        Xs_all, ys_all = build_lstm_sequences(X_full, y_full, LSTM_SEQ_LEN)
        # Take only the test portion (last len(y_test) - LSTM_SEQ_LEN rows)
        n_test_seq = len(y_test) - LSTM_SEQ_LEN
        Xs_test_seq = Xs_all[-n_test_seq:]
        ys_test_seq = ys_all[-n_test_seq:]

        Xs_t    = torch.from_numpy(Xs_test_seq)
        lstm_proba = lstm.predict_proba(Xs_t)
        lstm_pred  = np.argmax(lstm_proba, axis=1)
        results["lstm"] = evaluate_model("LSTM", ys_test_seq, lstm_pred, lstm_proba)

    # ---- 5. Calibration ----
    log.info("Computing calibration curves…")
    calibration = {
        "random_forest": calibration_metrics(y_test, rf_proba),
        "mlp": calibration_metrics(y_test, mlp_proba),
    }

    # ---- 6. Comparison table ----
    comparison = _build_comparison_table(results)
    log.info("\n=== MODEL COMPARISON ===\n%s", comparison.to_string())

    # ---- 7. Save report ----
    report = {
        model: {
            "accuracy":        res["accuracy"],
            "confusion_matrix": res["confusion_matrix"],
            "report":          res["report"],
        }
        for model, res in results.items()
    }
    report["calibration"] = calibration

    report_path = os.path.join(REPORTS_DIR, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Metrics saved → %s", report_path)

    comparison_path = os.path.join(REPORTS_DIR, "comparison_table.csv")
    comparison.to_csv(comparison_path, index=False)
    log.info("Comparison table → %s", comparison_path)

    return results, calibration


def _build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        if "report" not in res:
            continue
        r = res["report"]
        rows.append({
            "Model":        res["name"],
            "Accuracy":     f"{res['accuracy']:.4f}",
            "F1 Home Win":  f"{r['Home Win']['f1-score']:.4f}",
            "F1 Draw":      f"{r['Draw']['f1-score']:.4f}",
            "F1 Away Win":  f"{r['Away Win']['f1-score']:.4f}",
            "Macro F1":     f"{r['macro avg']['f1-score']:.4f}",
            "Weighted F1":  f"{r['weighted avg']['f1-score']:.4f}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_evaluation()
