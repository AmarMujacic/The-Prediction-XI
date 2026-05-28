"""
shap_explain.py
---------------
SHAP (SHapley Additive exPlanations) for the football prediction models.

Provides:
  - compute_shap_rf()     : SHAP values for Random Forest (TreeExplainer - fast & exact)
  - compute_shap_mlp()    : SHAP values for MLP (DeepExplainer)
  - plot_shap_summary()   : Beeswarm / bar summary plot (saved to outputs/plots/)
  - plot_shap_waterfall() : Single-prediction waterfall (why THIS prediction?)
  - shap_for_prediction() : Returns top features for one match prediction (used in app)
"""

import os
import logging
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join("outputs", "models")
PLOTS_DIR  = os.path.join("outputs", "plots")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_rf():
    path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_mlp(n_features):
    from model import load_mlp
    with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "rb") as f:
        h = pickle.load(f)
    hp = h["hparams"]
    return load_mlp(
        os.path.join(MODELS_DIR, "mlp.pt"),
        n_features, hp["hidden_sizes"], hp["dropout"]
    )


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_rf(X_background: np.ndarray, X_explain: np.ndarray,
                    feat_cols: list[str]) -> shap.Explanation:
    """
    Compute SHAP values for the Random Forest using TreeExplainer.
    Returns a shap.Explanation object with shape (n_samples, n_features, n_classes).
    """
    log.info("Computing SHAP values for Random Forest…")
    rf = _load_rf()
    clf = rf.pipeline.named_steps["clf"]
    scaler = rf.pipeline.named_steps["scaler"]

    X_bg_scaled = scaler.transform(X_background)
    X_ex_scaled = scaler.transform(X_explain)

    explainer   = shap.TreeExplainer(clf, data=X_bg_scaled[:200])
    shap_values = explainer(X_ex_scaled)
    return shap_values, feat_cols


def compute_shap_mlp(X_background: np.ndarray, X_explain: np.ndarray,
                     feat_cols: list[str]) -> np.ndarray:
    """
    Compute SHAP values for the MLP using KernelExplainer on a small background.
    Returns array of shape (n_samples, n_features, n_classes).
    Note: slower than TreeExplainer — uses a 50-sample background.
    """
    log.info("Computing SHAP values for MLP (this takes ~1 min)…")
    n_features = len(feat_cols)
    mlp = _load_mlp(n_features)
    mlp.eval()

    def _predict_proba(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            return torch.softmax(mlp(t), dim=1).numpy()

    # Use a small background for speed
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_background), min(50, len(X_background)), replace=False)
    background = X_background[bg_idx]

    explainer   = shap.KernelExplainer(_predict_proba, background)
    # shap_values is a list of (n_explain, n_features) arrays — one per class
    shap_values = explainer.shap_values(X_explain[:50], nsamples=100)
    return shap_values, feat_cols


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_values, feat_cols: list[str],
                      X_explain: np.ndarray, model_name: str = "rf"):
    """
    Beeswarm summary plot — shows which features matter most globally
    and whether high/low values push toward each class.
    Saves one plot per class.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    class_names = ["Home Win", "Draw", "Away Win"]

    for cls_idx, cls_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(9, 6))

        if hasattr(shap_values, "values"):
            # shap.Explanation object (TreeExplainer)
            sv = shap_values.values[:, :, cls_idx]
            exp = shap.Explanation(
                values=sv,
                base_values=shap_values.base_values[:, cls_idx]
                            if shap_values.base_values.ndim > 1
                            else shap_values.base_values,
                data=shap_values.data,
                feature_names=feat_cols,
            )
            shap.plots.beeswarm(exp, max_display=15, show=False)
        else:
            # List of arrays (KernelExplainer)
            sv = shap_values[cls_idx] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(sv, X_explain, feature_names=feat_cols,
                              max_display=15, show=False)

        plt.title(f"SHAP Feature Impact — {cls_name} ({model_name.upper()})",
                  fontsize=13)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}_{cls_name.replace(' ', '_').lower()}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        log.info("Saved → %s", path)


def plot_shap_waterfall(shap_values, feat_cols: list[str],
                        sample_idx: int = 0, cls_idx: int = 0,
                        model_name: str = "rf"):
    """
    Waterfall plot for a single prediction — shows exactly which features
    pushed the model toward or away from the predicted class.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    class_names = ["Home Win", "Draw", "Away Win"]

    if hasattr(shap_values, "values"):
        sv = shap_values.values[sample_idx, :, cls_idx]
        bv = (shap_values.base_values[sample_idx, cls_idx]
              if shap_values.base_values.ndim > 1
              else shap_values.base_values[sample_idx])
        data = shap_values.data[sample_idx]
        exp = shap.Explanation(
            values=sv,
            base_values=bv,
            data=data,
            feature_names=feat_cols,
        )
        shap.plots.waterfall(exp, max_display=12, show=False)
    else:
        sv = shap_values[cls_idx][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx]
        shap.waterfall_plot(shap.Explanation(
            values=sv,
            base_values=0.0,
            feature_names=feat_cols,
        ), max_display=12, show=False)

    plt.title(f"SHAP Waterfall — {class_names[cls_idx]} (sample {sample_idx})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR,
                        f"shap_waterfall_{model_name}_{class_names[cls_idx].replace(' ','_').lower()}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# App helper — top contributing features for ONE prediction
# ---------------------------------------------------------------------------

def shap_for_prediction(
    feat_vec: np.ndarray,
    feat_cols: list[str],
    top_n: int = 8,
) -> dict:
    """
    Compute SHAP values for a single feature vector using the RF model.
    Returns a dict:
      {
        "Home Win":  [(feature_name, shap_value), ...],  # top_n features
        "Draw":      [...],
        "Away Win":  [...],
      }
    Fast enough to run on every app prediction (TreeExplainer).
    """
    rf = _load_rf()
    clf    = rf.pipeline.named_steps["clf"]
    scaler = rf.pipeline.named_steps["scaler"]

    x_scaled = scaler.transform(feat_vec.reshape(1, -1))
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_scaled)   # list of (1, n_feat) per class

    class_names = ["Home Win", "Draw", "Away Win"]
    result = {}
    for i, cls in enumerate(class_names):
        sv = shap_values[i][0] if isinstance(shap_values, list) else shap_values[0, :, i]
        pairs = sorted(zip(feat_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        result[cls] = pairs

    return result


# ---------------------------------------------------------------------------
# Entry point — run full SHAP analysis and save all plots
# ---------------------------------------------------------------------------

def run_shap_analysis():
    from train import load_split
    X_train, y_train, X_test, y_test, feat_cols, _ = load_split()

    log.info("=== SHAP ANALYSIS (Random Forest) ===")
    shap_vals, _ = compute_shap_rf(X_train, X_test[:200], feat_cols)
    plot_shap_summary(shap_vals, feat_cols, X_test[:200], model_name="rf")
    plot_shap_waterfall(shap_vals, feat_cols, sample_idx=0, cls_idx=0, model_name="rf")
    plot_shap_waterfall(shap_vals, feat_cols, sample_idx=0, cls_idx=1, model_name="rf")
    plot_shap_waterfall(shap_vals, feat_cols, sample_idx=0, cls_idx=2, model_name="rf")

    log.info("=== SHAP DONE — plots saved to %s ===", PLOTS_DIR)


if __name__ == "__main__":
    run_shap_analysis()
