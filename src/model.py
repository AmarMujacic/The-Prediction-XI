"""
model.py
--------
Defines all model architectures used in this project:

  - FootballMLP   : Deep Multi-Layer Perceptron (main deep-learning model)
  - FootballLSTM  : LSTM for sequence-of-form-windows input (optional bonus)
  - BaselineModel : Thin sklearn wrapper (Random Forest / Logistic Regression)

PyTorch is used for deep models; scikit-learn for the baseline.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Deep Models — PyTorch
# ---------------------------------------------------------------------------

class FootballMLP(nn.Module):
    """
    Multi-Layer Perceptron for 3-class match outcome prediction.

    Architecture:
        Input(n_features)
        → Linear → BatchNorm → ReLU → Dropout
        → Linear → BatchNorm → ReLU → Dropout
        → Linear → BatchNorm → ReLU → Dropout
        → Linear(3)   [raw logits; use CrossEntropyLoss during training]

    Outputs logits (not softmax) so CrossEntropyLoss handles the numerics.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities as a numpy array (inference only)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


class FootballLSTM(nn.Module):
    """
    LSTM that processes a sequence of per-match feature vectors and predicts
    the outcome of the NEXT match.

    Input shape: (batch, seq_len, n_features)
    Output shape: (batch, 3)  — logits for H/D/A

    The sequence represents the last `seq_len` matches for the home team,
    giving the model a temporal view of form dynamics.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_size: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        # Use the last time-step hidden state
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.head(last)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Baseline — sklearn
# ---------------------------------------------------------------------------

class BaselineModel:
    """
    Thin wrapper around an sklearn pipeline.
    Provides a unified interface matching the deep models:
      .fit(X, y), .predict(X), .predict_proba(X)

    Default: Random Forest (good balance of accuracy and interpretability).
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        if model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 300),
                max_depth=kwargs.get("max_depth", 12),
                min_samples_leaf=kwargs.get("min_samples_leaf", 4),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                C=kwargs.get("C", 1.0),
                class_weight="balanced",
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}")

        # StandardScaler inside pipeline so normalisation is clean
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.model_type = model_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def __repr__(self):
        return f"BaselineModel(type={self.model_type})"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_mlp(path: str, n_features: int, hidden_sizes=(256, 128, 64), dropout=0.3) -> FootballMLP:
    model = FootballMLP(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
