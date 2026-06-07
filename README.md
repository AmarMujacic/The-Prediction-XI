# The Prediction XI ⚽

**Football Match Outcome Prediction Using Deep Learning**  
*Practical Application of AI (PAAI) — University Course Project*

---

## Poster

![Project Poster](outputs/reports/poster.webp)

---

## Overview

A complete machine learning system that predicts football match outcomes — **Home Win**, **Draw**, or **Away Win** — using historical match data, advanced feature engineering, and an ensemble of 5 trained models.

The project includes a fully interactive **Streamlit web app** with Elo ratings, SHAP explainability, Bet365 odds comparison, Football Manager data integration, and a prediction history tracker.

---

## Features

| Feature | Description |
|---|---|
| **5 Models** | Random Forest, XGBoost, LightGBM, Deep MLP, Ensemble |
| **Elo Ratings** | Live team strength ratings displayed before each prediction |
| **SHAP Explainability** | Bar chart showing which features drove each prediction |
| **Bet365 Odds Comparison** | Our model vs market implied probabilities |
| **Football Manager Integration** | Import FM export files to predict from your save game |
| **Prediction History Log** | Every prediction saved to CSV with accuracy tracking |
| **Hyperparameter Tuning** | Optuna TPE sampler (30 trials) for MLP architecture |
| **EDA Notebook** | Full exploratory data analysis with league and odds breakdowns |

---

## Models

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| Naive (Always Home Win) | Baseline | ~46% | ~21% |
| Random Forest | Baseline | ~53% | ~43% |
| XGBoost | Gradient Boosting | ~54% | ~44% |
| LightGBM | Gradient Boosting | ~53% | ~43% |
| Deep MLP | Neural Network | ~55% | ~46% |
| **Ensemble** | **Weighted Average** | **~56%** | **~47%** |

*Test set: 2015/2016 season (time-based split — no data leakage)*

---

## Data

**Source:** [football-data.co.uk](https://www.football-data.co.uk) — downloaded automatically, no account needed.

| Property | Value |
|---|---|
| Leagues | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 |
| Seasons | 2009/10 – 2015/16 |
| Matches | ~12,000 after cleaning |
| Test split | 2015/2016 season |

---

## Feature Engineering

All features are computed using only pre-match data (no leakage):

| Group | Features |
|---|---|
| Rolling form (last 5) | Win/draw/loss rate, goals scored/conceded, normalised points |
| Head-to-head (last 5) | Home win %, draw %, away win % |
| Venue strength (last 10) | Avg goals scored/conceded at home / away, win rate |
| Form differentials | Home minus away: points, goal diff, win rate |
| Elo ratings | Team strength ratings updated after every match |
| Season context | Normalised stage, league encoding |

---

## Project Structure

```
The-Prediction-XI/
├── data/
│   ├── raw/                        ← auto-downloaded CSVs
│   ├── processed/                  ← cleaned parquet files
│   └── fm_exports/                 ← Football Manager HTML/CSV exports
│       └── HOW_TO_EXPORT.md
├── notebooks/
│   └── exploration.ipynb           ← EDA notebook
├── src/
│   ├── preprocessing.py            ← data download, cleaning, encoding
│   ├── features.py                 ← 35 engineered features
│   ├── elo.py                      ← Elo rating system
│   ├── model.py                    ← MLP, LSTM, RF, XGBoost, LightGBM, Ensemble
│   ├── train.py                    ← full training pipeline + Optuna
│   ├── evaluate.py                 ← metrics, confusion matrices, calibration
│   ├── visualize.py                ← all plots
│   ├── shap_explain.py             ← SHAP explainability
│   └── fm_import.py                ← Football Manager data importer
├── outputs/
│   ├── models/                     ← saved model weights (gitignored)
│   ├── plots/                      ← all generated figures
│   └── reports/                    ← metrics JSON, comparison CSV
├── app.py                          ← Streamlit web app
├── PHASE1_PITCH.md                 ← Week 5 pitch script and slides
├── PHASE7_POSTER.md                ← Final poster content
├── PHASE8_CONCLUSIONS.md           ← Conclusions and future work
├── requirements.txt
└── README.md
```

---

## How to Run

### First time setup (any computer)

```bash
# 1. Clone the repo
git clone https://github.com/AmarMujacic/The-Prediction-XI.git
cd The-Prediction-XI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and process data (auto-downloads from football-data.co.uk)
python src/preprocessing.py
python src/features.py
python src/elo.py

# 4. Train all models (~5 min with --skip-optuna)
python src/train.py --skip-optuna

# 5. Evaluate
python src/evaluate.py

# 6. Generate plots
python src/visualize.py

# 7. Launch the app
streamlit run app.py
```

### Already trained (subsequent runs)

```bash
streamlit run app.py
```

### Optional — Full Optuna tuning (~25 min)

```bash
python src/train.py
```

### Optional — Train LSTM model

```bash
python src/train.py --skip-optuna --lstm
```

---

## Streamlit App

Three tabs:

**Historical Data**
- Select any two teams from 5 European leagues
- Live Elo strength ratings with tier badges (Elite / Strong / Average / Weak / Poor)
- Predict with any of 5 models: Ensemble, Deep MLP, XGBoost, LightGBM, Random Forest
- SHAP waterfall chart — "Why this prediction?"
- Bet365 odds comparison — model vs market
- Prediction auto-saved to log

**Football Manager**
- Import a CSV or HTML export from your Football Manager save
- Predicts outcomes from your save game's league-table stats
- Sample export included (`data/fm_exports/fm_table.csv`)
- Same Elo + SHAP + probability display

**Prediction Log**
- Full history of every prediction made
- Accuracy tracking — enter actual results to measure performance
- Pie chart + confidence breakdown
- CSV download

---

## Using Football Manager Data

The app reads team statistics from your Football Manager save and predicts
match outcomes from them. Two ways to provide the data:

**Option A — CSV (recommended, works on all FM versions)**

Create a CSV with one row per team and these columns, then save it to
`data/fm_exports/`:

```
team,played,wins,draws,losses,goals_scored,goals_conceded,points
Man City,20,14,3,3,49,19,45
Liverpool,20,13,3,4,48,23,42
...
```

A working example from an FM26 save is included at
[`data/fm_exports/fm_table.csv`](data/fm_exports/fm_table.csv).

**Option B — HTML export (older FM versions)**

In FM, open a league-table view, use the print/export flow, and choose
**Web Page (HTML)**. Save the `.html` file to `data/fm_exports/`.
(Note: in FM26, `Ctrl+P` opens Preferences — use the export icon on the
table view instead.)

**Then:**
1. Open the app → **Football Manager** tab
2. Reload the page, or use the **Upload** button to load a file directly
3. Select two teams and predict — full Elo, SHAP and probability output

---

## Reproducibility

- Random seed: `42` (NumPy + PyTorch)
- Time-based train/test split — test set is always 2015/2016 season
- Normalisation stats saved to `outputs/models/norm_stats.pkl`
- All model weights saved to `outputs/models/`

---

## Evaluation Highlights

- Ensemble beats the naive "always predict Home Win" baseline by **+10% accuracy**
- Draw prediction (F1 ~31%) remains the hardest class — consistent with bookmaker difficulty
- SHAP analysis shows rolling form goal differential and Elo rating differential are the strongest predictors
- MLP is better calibrated than Random Forest — probabilities are more trustworthy

---

## Conclusions & Future Work

See [PHASE8_CONCLUSIONS.md](PHASE8_CONCLUSIONS.md) for full discussion.

**Short-term improvements:**
- Player-level features (injuries, lineup data)
- Betting odds as meta-features
- SHAP-based feature selection

**Long-term:**
- Real-time API pipeline for live predictions
- Transformer architecture for sequence modelling
- Graph Neural Networks modelling the league as a graph

---


