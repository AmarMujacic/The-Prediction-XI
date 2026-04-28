# Phase 1 — Problem Definition & 3-Minute Pitch

---

## 3-Minute Pitch Script

**[0:00 – 0:30] Hook & Real-World Importance**

"Football is the most watched sport on Earth — over 3.5 billion fans, billions of euros in betting markets, and club revenue decisions tied directly to match outcomes. Yet predicting the result of a 90-minute game remains one of the hardest classification problems in sports analytics. Today, we present a deep learning system that learns from historical match data to predict whether a game ends in a Home Win, Draw, or Away Win — with explainable probabilities, not just a guess."

---

**[0:30 – 1:00] Problem Definition**

"We frame this as a 3-class classification problem:
- Input: Historical statistics for both teams — recent form, head-to-head records, home/away strength, goal averages, and performance trends.
- Output: A probability distribution over three outcomes — Home Win (H), Draw (D), Away Win (A).

Why is this hard? Football is inherently noisy. A red card, an injury, or a referee decision can flip any match. Class imbalance is real — draws are historically underrepresented. And raw statistics often hide momentum, fatigue, and motivation."

---

**[1:00 – 1:45] Our Approach**

"We build a full ML pipeline:
1. Data from the European Soccer Database (Kaggle) — 25,000+ matches across 11 leagues over 8 seasons.
2. Feature engineering: rolling form windows, Elo-style strength ratings, H2H aggregates.
3. Two models: a Random Forest baseline and a deep MLP with dropout and early stopping — plus an optional LSTM for sequence modeling.
4. Rigorous evaluation: accuracy, F1-score per class, confusion matrix, and probability calibration.

Everything is modular, reproducible, and pushed to GitHub."

---

**[1:45 – 2:30] Challenges & How We Address Them**

"Three core challenges:
- **Class imbalance**: Draws are rare (~26% of matches). We use class-weighted loss and SMOTE oversampling.
- **Noise**: We do NOT try to predict single games perfectly — we aim for calibrated probabilities that beat bookmaker baselines.
- **Feature leakage**: We strictly split train/test by time (no future data leaks into past features)."

---

**[2:30 – 3:00] Project Plan & Closing**

"Our timeline: data pipeline by Week 8, models by Week 10, evaluation and visualizations by Week 12, poster and presentation Week 14.

The deliverable is not just a model — it's a reproducible research artifact: clean code, professional GitHub repo, interactive Streamlit app, and a poster. Our goal is to show that with the right features and architecture, deep learning can extract signal from the noise of football."

---

## Slide Content (Bullet Points)

### Slide 1 — Title
- **Football Match Outcome Prediction Using Deep Learning**
- Course: Practical Application of AI
- Team: [Names]
- Date: 2026

---

### Slide 2 — The Problem
- Predict result of a football match: **Home Win / Draw / Away Win**
- 3-class probabilistic classification
- Input: team statistics, form, head-to-head data
- Output: probability vector [P(H), P(D), P(A)]
- Real-world value: betting markets, scouting, tactical planning

---

### Slide 3 — Why This Is Hard
- **Noise**: single events (red cards, injuries) change outcomes
- **Class imbalance**: Draws ≈ 26%, rarely predicted well
- **Non-stationarity**: team quality changes across seasons
- **Feature leakage risk**: must use only past data to predict future

---

### Slide 4 — Data Sources
- **European Soccer Database** (Kaggle, SQLite) — 25,979 matches
- Covers: England, France, Germany, Italy, Spain + 6 more leagues
- Seasons: 2008/09 → 2015/16
- Features available: match stats, team attributes (FIFA ratings), player data

---

### Slide 5 — Feature Engineering Plan
- Last 5 match form (wins/draws/losses, goals scored/conceded)
- Head-to-head win rate (last 5 H2H meetings)
- Home vs away goal averages
- Elo-style team strength rating
- Goal difference trend (momentum)
- Season point tally at match date

---

### Slide 6 — Model Architecture Plan
| Model | Type | Purpose |
|---|---|---|
| Random Forest | Baseline | Interpretable benchmark |
| MLP (3 hidden layers) | Deep Learning | Main model |
| LSTM (optional) | Sequence Model | Capture form sequences |

- Loss: CrossEntropyLoss (multi-class)
- Regularization: Dropout, L2 weight decay
- Optimization: Adam + early stopping

---

### Slide 7 — Evaluation Plan
- Accuracy, Precision, Recall, F1 (per class + macro)
- Confusion matrix
- Calibration curve (are probabilities trustworthy?)
- Compare: Baseline vs MLP vs bookmaker naive baseline

---

### Slide 8 — Project Timeline
| Week | Milestone |
|---|---|
| 5 | Pitch (this presentation) |
| 8 | Data pipeline complete |
| 10 | Models trained & tuned |
| 12 | Evaluation, plots, Streamlit app |
| 14 | Poster + Final presentation |

---

### Slide 9 — Expected Outcomes
- Accuracy target: **>52%** (beating random = 33%, home-win naive ≈ 46%)
- Well-calibrated probability estimates
- Reproducible, documented codebase on GitHub
- Interactive prediction demo (Streamlit)

---

### Slide 10 — Q&A
- "How is this different from betting odds?" → We explain uncertainty, they just price it.
- "Can you predict a single game?" → No model can reliably; we predict distributions.
- "What would make it better?" → Real-time data, player-level features, weather, lineup info.
