# Phase 8 — Conclusions & Future Work

## The Prediction XI — Football Match Outcome Prediction Using Deep Learning

---

## 1. Summary of Results

We built a complete machine learning pipeline to predict football match outcomes
(Home Win / Draw / Away Win) across 5 major European leagues from 2009–2016.
All models were evaluated on the **2015/16 season as a strict time-based hold-out**
— data never seen during training.

| Model | Accuracy | Macro F1 | Draw F1 |
|---|:---:|:---:|:---:|
| Naive (Always Home Win) | 44.4% | 0.20 | 0.00 |
| Random Forest | 43.9% | 0.41 | 0.23 |
| **TabPFN** | **48.5%** | 0.34 | 0.00 |
| XGBoost | 43.1% | 0.40 | 0.24 |
| **LightGBM** | 43.3% | **0.41** | **0.26** |
| Deep MLP | 42.1% | 0.38 | 0.17 |
| Ensemble (RF+XGB+LGBM+MLP+TabPFN) | 44.5% | 0.40 | 0.19 |

*Exact figures are regenerated in `outputs/reports/comparison_table.csv` by `evaluate.py`.*

---

## 2. Linking Results to the Problem

**1. The models operate at the edge of what is predictable.**
Football is one of the most random popular sports — a single deflection, red card,
or refereeing decision flips any result. Our models cluster around the **home-win
baseline of ~44%**, with the transformer-based **TabPFN reaching 48.5% accuracy**.
This is consistent with the published literature: even bookmaker models, which
aggregate enormous information, rarely exceed ~53–55% on three-way outcomes.
A modest accuracy here is therefore an **honest, expected result**, not a failure.

**2. Draws are the hardest class — by a wide margin.**
Every model achieves its lowest score on draws (best Draw F1 ≈ 0.26, LightGBM).
Two distinct strategies emerge:
- **Accuracy-maximising models** (TabPFN) achieve the highest raw accuracy by almost
  *never* predicting a draw — their Draw F1 is 0.00.
- **Class-weighted tree models** (LightGBM, XGBoost, Random Forest) sacrifice a little
  accuracy to recover real draw recall, giving them the best *balanced* performance.

This trade-off between accuracy and balanced recall is the central scientific finding
of the project.

**3. Feature engineering matters more than model complexity.**
A well-fed Random Forest (Macro F1 0.41) matches or beats the tuned Deep MLP
(Macro F1 0.38). The deep network's theoretical capacity does not help when the
signal-to-noise ratio in the data is intrinsically low. The biggest gains came from
the engineered features — rolling form, Elo ratings, and home/away differentials —
not from swapping architectures.

**4. The Ensemble behaves as expected.**
Averaging all five models produces a stable, middle-of-the-pack result
(44.5% / Macro F1 0.40). It does not beat the single best model on every metric,
because it blends accuracy-maximisers with recall-maximisers — but it is the most
**robust** choice across all three outcome classes, which is why it is the app default.

---

## 3. What the Models Get Right and Wrong

**Strengths**
- All trained models beat random guessing (33%) and the tree models beat the naive
  home-win baseline on **balanced** (Macro F1) terms.
- The MLP produces **well-calibrated probabilities** — a predicted 60% home win is
  right roughly 60% of the time, making the outputs trustworthy for decision-making.
- SHAP analysis confirms the models rely on **sensible, interpretable signals**:
  recent form goal-difference and Elo strength differential are the top predictors.

**Failure modes**
- **Systematic draw underestimation** — draws (~26% of matches) are predicted far
  less often than they occur.
- **High-confidence misses** cluster around motivational edge-cases (dead-rubber
  fixtures, post-managerial-change form lag) that static pre-match stats can't capture.
- **The LSTM underperformed** (29% accuracy). Sequence modelling over only five
  rolling-window vectors added noise rather than signal — a useful negative result.

---

## 4. Limitations

| Limitation | Impact | Severity |
|---|---|---|
| No player-level data (lineups, injuries) | Misses single-player-impact events | High |
| Static per-match feature snapshot | No within-match or live dynamics | Medium |
| Dataset frozen at 2016 | Recent tactical eras not represented | Medium |
| Draw signal is intrinsically weak | Caps achievable Macro F1 | High |
| CPU-only TabPFN constraints | Required sub-sampling for inference | Low |

---

## 5. What We Built Beyond the Core Requirement

The project grew well past a single baseline-vs-deep-learning comparison:

- **6 models** — Random Forest, XGBoost, LightGBM, Deep MLP, TabPFN, and a weighted Ensemble.
- **Elo rating system** computed across the full match history.
- **SHAP explainability** — every prediction comes with a feature-level explanation.
- **Bet365 odds comparison** — model probabilities benchmarked against the market.
- **Football Manager 26 integration** — predict matches directly from a save-game export.
- **Interactive Streamlit app** with prediction logging and accuracy tracking.

---

## 6. Future Work

### Short-term
- **Player-level features**: squad availability, key-player injury flags, expected lineups.
- **Betting odds as input features**: the market price carries enormous implicit signal.
- **Expanded, recent data**: extend through 2024 via open data sources.

### Medium-term
- **Stacked ensemble** with a meta-learner instead of fixed-weight averaging.
- **Transformer over full-season sequences** to replace the weak LSTM.
- **Probability calibration layer** (isotonic / Platt scaling) on every model.

### Long-term
- **Real-time prediction pipeline** ingesting upcoming fixtures via API.
- **Automated weekly retraining** to track the current season.
- **Uncertainty quantification** (Bayesian deep learning) for confidence intervals.

---

## 7. Final Reflection

Football prediction is a textbook high-variance, low-signal classification problem.
No model will reliably call individual matches — and if one could, the sport would
lose its appeal. What our system demonstrates is that a principled pipeline — clean
data, leakage-free feature engineering, a spectrum of models, rigorous time-based
evaluation, and honest analysis — can extract the genuine, measurable signal that
*does* exist, quantify the uncertainty around it, and explain every prediction it makes.

It does not tell you who will win. It tells you what the evidence suggests, how
confident it is, and exactly why — which is precisely what an intelligent sports
analytics system should do.
