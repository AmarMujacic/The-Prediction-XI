"""
train.py
--------
Full training pipeline:

  1. Load feature matrix
  2. Time-based train/test split (no shuffling across time)
  3. Class-weight computation to handle imbalance
  4. Optuna hyperparameter search for the MLP
  5. Final training of:
       - Random Forest baseline
       - TabPFN (transformer pretrained on tabular data)
       - Best MLP (from Optuna)
       - LSTM (sequence model, optional)
  6. Save all model artefacts to outputs/models/

Usage:
  python src/train.py                  # full pipeline
  python src/train.py --skip-optuna    # skip Optuna, use default MLP hparams
  python src/train.py --lstm           # also train LSTM
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

import optuna
from optuna.samplers import TPESampler

from model import (BaselineModel, FootballLSTM, FootballMLP, TabPFNModel,
                   XGBoostModel, LightGBMModel, EnsembleModel, count_parameters)
from features import get_feature_columns, normalise

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH  = os.path.join("data", "processed", "features.parquet")
MODELS_DIR     = os.path.join("outputs", "models")
METRICS_DIR    = os.path.join("outputs", "reports")
SEED           = 42
OPTUNA_TRIALS  = 30
TEST_SEASONS   = [2015]   # hold-out: 2015/2016 season
LSTM_SEQ_LEN   = 5        # number of prior matches fed to LSTM

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_split(path: str = FEATURES_PATH):
    """Time-based split: train on everything before TEST_SEASONS, test on them."""
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    feat_cols = get_feature_columns(df)

    train_df = df[~df["season_year"].isin(TEST_SEASONS)]
    test_df  = df[ df["season_year"].isin(TEST_SEASONS)]

    # Normalise (stats from train only)
    train_norm, test_norm, norm_stats = normalise(train_df, test_df, feat_cols)

    X_train = train_norm[feat_cols].values.astype(np.float32)
    y_train = train_norm["result"].values.astype(np.int64)
    X_test  = test_norm[feat_cols].values.astype(np.float32)
    y_test  = test_norm["result"].values.astype(np.int64)

    log.info("Train: %d samples | Test: %d samples | Features: %d",
             len(X_train), len(X_test), len(feat_cols))

    return X_train, y_train, X_test, y_test, feat_cols, norm_stats


def class_weights_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 256):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


# ---------------------------------------------------------------------------
# MLP training loop (used by Optuna and final training)
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hparams: dict,
    device: torch.device,
    epochs: int = 120,
    patience: int = 12,
    verbose: bool = True,
) -> tuple[FootballMLP, list, list]:
    """
    Train a FootballMLP with the given hyperparameters.
    Returns (trained_model, train_loss_history, val_loss_history).
    """
    model = FootballMLP(
        n_features=X_train.shape[1],
        hidden_sizes=hparams["hidden_sizes"],
        dropout=hparams["dropout"],
    ).to(device)

    cw = class_weights_tensor(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    loader = make_loaders(X_train, y_train, batch_size=hparams.get("batch_size", 256))

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    train_losses, val_losses = [], []
    best_val  = float("inf")
    best_state = None
    stall     = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall      = 0
        else:
            stall += 1

        if verbose and (epoch + 1) % 10 == 0:
            log.info("Epoch %3d | train_loss=%.4f | val_loss=%.4f", epoch + 1, epoch_loss, val_loss)

        if stall >= patience:
            log.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _optuna_objective(trial, X_train, y_train, X_val, y_val, device):
    # Sample architecture & training hparams
    n_layers = trial.suggest_int("n_layers", 2, 4)
    sizes    = [
        trial.suggest_categorical(f"h{i}", [64, 128, 256, 512])
        for i in range(n_layers)
    ]
    hparams = {
        "hidden_sizes":  sizes,
        "dropout":       trial.suggest_float("dropout", 0.1, 0.5),
        "lr":            trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [128, 256, 512]),
    }

    _, _, val_losses = train_mlp(
        X_train, y_train, X_val, y_val,
        hparams=hparams, device=device,
        epochs=60, patience=8, verbose=False,
    )
    return min(val_losses)


def run_optuna(X_train, y_train, X_val, y_val, device, n_trials=OPTUNA_TRIALS):
    log.info("=== OPTUNA SEARCH (%d trials) ===", n_trials)
    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda t: _optuna_objective(t, X_train, y_train, X_val, y_val, device),
        n_trials=n_trials,
    )

    best = study.best_trial
    log.info("Best trial val_loss=%.4f | params=%s", best.value, best.params)

    n_layers = best.params["n_layers"]
    best_hparams = {
        "hidden_sizes": [best.params[f"h{i}"] for i in range(n_layers)],
        "dropout":      best.params["dropout"],
        "lr":           best.params["lr"],
        "weight_decay": best.params["weight_decay"],
        "batch_size":   best.params["batch_size"],
    }
    return best_hparams, study


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------

def build_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = LSTM_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature rows into overlapping sequences of length `seq_len`.
    Row i uses features from rows [i-seq_len .. i-1] to predict label at i.
    Rows with insufficient history are skipped.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 80,
    patience: int = 12,
) -> tuple[FootballLSTM, list, list]:
    log.info("Building LSTM sequences…")
    Xs_train, ys_train = build_lstm_sequences(X_train, y_train, LSTM_SEQ_LEN)
    Xs_val,   ys_val   = build_lstm_sequences(X_val,   y_val,   LSTM_SEQ_LEN)

    n_feat = Xs_train.shape[2]
    model  = FootballLSTM(n_features=n_feat, hidden_size=128, num_layers=2, dropout=0.3).to(device)

    cw        = class_weights_tensor(ys_train, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(torch.from_numpy(Xs_train), torch.from_numpy(ys_train))
    loader   = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

    Xv_t = torch.from_numpy(Xs_val).to(device)
    yv_t = torch.from_numpy(ys_val).to(device)

    train_losses, val_losses = [], []
    best_val, best_state, stall = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(Xs_train)

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv_t), yv_t).item()
        scheduler.step(vl)

        train_losses.append(ep_loss)
        val_losses.append(vl)

        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall      = 0
        else:
            stall += 1

        if (epoch + 1) % 10 == 0:
            log.info("LSTM Epoch %3d | train=%.4f | val=%.4f", epoch + 1, ep_loss, vl)

        if stall >= patience:
            log.info("LSTM early stop at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_optuna: bool = False, train_lstm_flag: bool = False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Load data ---
    X_train, y_train, X_test, y_test, feat_cols, norm_stats = load_split()

    # Use a validation slice (last 15% of training) for early stopping / Optuna
    val_cut  = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:val_cut], y_train[:val_cut]
    X_val, y_val = X_train[val_cut:], y_train[val_cut:]

    # Save norm stats so the Streamlit app can replicate normalisation
    with open(os.path.join(MODELS_DIR, "norm_stats.pkl"), "wb") as f:
        pickle.dump(norm_stats, f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feat_cols, f)

    # -----------------------------------------------------------------------
    # 1. Baseline — Random Forest
    # -----------------------------------------------------------------------
    log.info("=== TRAINING BASELINE (Random Forest) ===")
    baseline = BaselineModel("random_forest")
    baseline.fit(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "baseline_rf.pkl"), "wb") as f:
        pickle.dump(baseline, f)
    log.info("Baseline saved.")

    # -----------------------------------------------------------------------
    # 2. TabPFN — transformer pretrained on tabular data
    # -----------------------------------------------------------------------
    log.info("=== TRAINING TabPFN ===")
    tabpfn = TabPFNModel(device="cpu")
    tabpfn.fit(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "tabpfn.pkl"), "wb") as f:
        pickle.dump(tabpfn, f)
    log.info("TabPFN saved.")

    # -----------------------------------------------------------------------
    # 3. MLP — hyperparameter search then final training
    # -----------------------------------------------------------------------
    if skip_optuna:
        best_hparams = {
            "hidden_sizes": [256, 128, 64],
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
        }
        log.info("Skipping Optuna. Using default hparams: %s", best_hparams)
    else:
        best_hparams, study = run_optuna(X_tr, y_tr, X_val, y_val, device)
        study_path = os.path.join(MODELS_DIR, "optuna_study.pkl")
        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    log.info("=== FINAL MLP TRAINING ===")
    mlp, train_losses, val_losses = train_mlp(
        X_tr, y_tr, X_val, y_val,
        hparams=best_hparams,
        device=device,
        epochs=150,
        patience=15,
        verbose=True,
    )
    log.info("MLP parameters: %d", count_parameters(mlp))
    torch.save(mlp.state_dict(), os.path.join(MODELS_DIR, "mlp.pt"))

    # Save training history for plotting
    history = {"train_loss": train_losses, "val_loss": val_losses, "hparams": best_hparams}
    with open(os.path.join(MODELS_DIR, "mlp_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    # -----------------------------------------------------------------------
    # 4. XGBoost
    # -----------------------------------------------------------------------
    log.info("=== TRAINING XGBoost ===")
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    log.info("XGBoost saved.")

    # -----------------------------------------------------------------------
    # 5. LightGBM
    # -----------------------------------------------------------------------
    log.info("=== TRAINING LightGBM ===")
    lgbm_model = LightGBMModel()
    lgbm_model.fit(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "lightgbm.pkl"), "wb") as f:
        pickle.dump(lgbm_model, f)
    log.info("LightGBM saved.")

    # -----------------------------------------------------------------------
    # 6. Ensemble — RF + XGB + LGBM + MLP + TabPFN (weighted by val accuracy)
    # -----------------------------------------------------------------------
    log.info("=== BUILDING ENSEMBLE ===")

    # Quick validation accuracies to set weights
    def _val_acc(model, X, y):
        preds = np.argmax(model.predict_proba(X), axis=1) \
                if not hasattr(model, "predict") else model.predict(X)
        return float(np.mean(preds == y))

    mlp_cpu = mlp.cpu()

    val_accs = {
        "rf":     _val_acc(baseline,   X_val, y_val),
        "xgb":    _val_acc(xgb_model,  X_val, y_val),
        "lgbm":   _val_acc(lgbm_model, X_val, y_val),
        "tabpfn": _val_acc(tabpfn,     X_val, y_val),
        "mlp":    float(np.mean(
            np.argmax(mlp_cpu.predict_proba(
                torch.from_numpy(X_val)), axis=1) == y_val)),
    }
    log.info("Validation accuracies for ensemble weights: %s",
             {k: f"{v:.4f}" for k, v in val_accs.items()})

    weights = list(val_accs.values())
    ensemble = EnsembleModel(
        models=[baseline, xgb_model, lgbm_model, tabpfn, mlp_cpu],
        weights=weights,
    )
    with open(os.path.join(MODELS_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    log.info("Ensemble saved (weighted by val accuracy).")

    # -----------------------------------------------------------------------
    # 7. LSTM (optional)
    # -----------------------------------------------------------------------
    if train_lstm_flag:
        log.info("=== TRAINING LSTM ===")
        lstm_model, lstm_train_losses, lstm_val_losses = train_lstm(
            X_tr, y_tr, X_val, y_val, device
        )
        torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "lstm.pt"))
        lstm_history = {"train_loss": lstm_train_losses, "val_loss": lstm_val_losses}
        with open(os.path.join(MODELS_DIR, "lstm_history.pkl"), "wb") as f:
            pickle.dump(lstm_history, f)
        log.info("LSTM saved.")

    log.info("=== ALL MODELS TRAINED & SAVED ===")
    log.info("Artefacts in: %s", MODELS_DIR)

    return {
        "baseline": baseline,
        "mlp": mlp,
        "mlp_hparams": best_hparams,
        "X_test": X_test,
        "y_test": y_test,
        "feat_cols": feat_cols,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna tuning")
    parser.add_argument("--lstm", action="store_true", help="Also train the LSTM model")
    args = parser.parse_args()
    main(skip_optuna=args.skip_optuna, train_lstm_flag=args.lstm)
