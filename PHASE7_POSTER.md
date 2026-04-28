# Phase 7 — Poster Content

---

## Poster Title
**Football Match Outcome Prediction Using Deep Learning**
*Practical Application of AI — Sports Analytics*

---

## Section 1: Problem Statement

**Goal:** Predict the outcome of a football match — Home Win / Draw / Away Win — using only pre-match statistics.

**Why it matters:**
- 3.5 billion football fans worldwide
- Billions of euros in transfer, betting, and broadcast markets depend on outcome probabilities
- Accurate models help clubs with tactical scouting and financial projection

**Task formulation:**
- Input: feature vector encoding team form, head-to-head history, FIFA attributes
- Output: probability vector [P(Home Win), P(Draw), P(Away Win)]
- 3-class classification with CrossEntropy loss

**Key challenges:**
- Class imbalance (Draws ≈ 26%)
- High randomness (single events flip results)
- Non-stationarity across seasons

---

## Section 2: Data

| Property | Value |
|---|---|
| Source | Kaggle — European Soccer Database |
| Matches | ~25,979 |
| Leagues | 11 across Europe |
| Seasons | 2008/09 – 2015/16 |

**Class distribution:**
- Home Win: ~46%
- Draw: ~26%
- Away Win: ~28%

*[Insert `class_distribution.png` here]*

---

## Section 3: Feature Engineering

All features use only pre-match data (no leakage):

| Feature Group | Examples |
|---|---|
| Rolling form (last 5 matches) | Win rate, goals scored/conceded, normalised points |
| Head-to-head (last 5 H2H) | Home win %, draw %, away win % |
| Venue strength (last 10 home/away) | Avg goals scored/conceded at home / away |
| FIFA team attributes | Differential in speed, passing, defence, overall rating |
| Season context | Normalised stage number, league encoding |

**Total features: ~35 engineered predictors**

---

## Section 4: Model Architecture

### Baseline — Random Forest
- 300 trees, max depth 12
- Balanced class weights
- Gini impurity for feature selection

### Main Model — Deep MLP

```
Input (35 features)
  → Linear(256) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(128) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(64)  + BatchNorm + ReLU + Dropout(0.3)
  → Linear(3)   [logits]
  → Softmax → [P(H), P(D), P(A)]
```

- Loss: CrossEntropyLoss with class weights
- Optimiser: Adam + ReduceLROnPlateau
- Regularisation: Dropout + L2 weight decay
- Tuning: Optuna (30 trials, TPE sampler)
- Early stopping: patience = 15 epochs

### Bonus — LSTM
- 2-layer LSTM, hidden size 128
- Input: sequence of last 5 match features
- Captures temporal form dynamics

*[Insert model architecture diagram here]*

---

## Section 5: Results

*[Insert `model_comparison.png` here]*

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive (Always Home Win) | ~46% | ~21% |
| Random Forest | ~53% | ~43% |
| Deep MLP | ~55% | ~46% |
| LSTM | ~54% | ~44% |

*[Insert `cm_mlp_norm.png` here — normalised confusion matrix]*

*[Insert `loss_curves_mlp.png` here]*

**Key insight:** Draw prediction remains the hardest class for all models — consistent with its real-world unpredictability. MLP outperforms Random Forest on Home Win and Away Win but both struggle on Draws.

---

## Section 6: Calibration

*[Insert `calibration_curves.png` here]*

The MLP produces better-calibrated probabilities than the Random Forest, meaning predicted probabilities are more trustworthy for downstream use (e.g. betting or scouting dashboards).

---

## Section 7: Feature Importance

*[Insert `feature_importance_rf.png` here]*

Top predictors:
1. Home team rolling goal differential (form_gd)
2. FIFA overall rating differential
3. H2H home win rate
4. Home venue average goals scored
5. Away team normalised points (recent form)

---

## Section 8: Conclusion

- Deep MLP achieves **~55% accuracy** on the 2015/2016 hold-out season
- Beats the naive "always predict home win" baseline by ~9 percentage points
- Class weighting and dropout are essential — without them, models ignore Draws
- Optuna found that **smaller, deeper networks** (256→128→64) outperform wide shallow ones

**Future work:**
- Player-level lineup features (injuries, suspensions)
- Real-time API pipeline for live predictions
- Transformer architecture for sequence modelling
- Ensemble of MLP + LSTM + Gradient Boosting

---

## Section 9: Live Demo

**Streamlit App:** `streamlit run app.py`
- Enter home team, away team, and recent stats
- Get predicted probabilities for all three outcomes
- Visual probability bar chart

---

*[Poster should be A1 landscape, 2-column layout]*
*[Use university color scheme, include team photo and contact info]*
