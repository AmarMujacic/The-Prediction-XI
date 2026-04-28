# Football Match Outcome Prediction Using Deep Learning

**Course:** Practical Application of AI (PAAI)  
**Topic:** Sports Analytics — Predicting Football Match Results

---

## Problem

Football match prediction is a classic 3-class classification problem:

> Given pre-match statistics for both teams, predict whether the match ends in a **Home Win**, **Draw**, or **Away Win**.

Real-world applications include:
- Scouting and tactical analysis
- Sports betting market pricing
- Club financial planning (prize money projections)

Key challenges:
- **Class imbalance**: Draws (~26%) are systematically underrepresented
- **Noise**: A single red card can flip any result; no model is perfect
- **Non-stationarity**: Team quality changes across seasons
- **Leakage prevention**: All features must use only pre-match data

---

## Data

**Source:** [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (Kaggle)

| Property | Value |
|---|---|
| Matches | ~25,979 |
| Leagues | 11 (England, Germany, Spain, Italy, France + 6 more) |
| Seasons | 2008/09 – 2015/16 |
| Format | SQLite database |

Download the `.sqlite` file and place it at `data/raw/database.sqlite`.

---

## Method

### Feature Engineering

All features are computed using only data available **before kick-off**:

| Feature Group | Features |
|---|---|
| Rolling form (last 5) | Win/draw/loss rate, goals scored/conceded, normalised points |
| Head-to-head | Home win rate, draw rate, away win rate (last 5 meetings) |
| Venue strength | Avg goals scored/conceded at home / away (last 10 matches) |
| FIFA team attributes | Speed, passing, shooting, defence ratings (differentials) |
| Season context | Season progress (normalised stage number), league encoding |

### Models

| Model | Type | Key Design Choices |
|---|---|---|
| Random Forest | Baseline | 300 trees, balanced class weights, Gini importance |
| Deep MLP | Main DL model | 3 hidden layers, BatchNorm, Dropout, CrossEntropy + class weights |
| LSTM | Bonus | 2-layer LSTM on 5-match form sequences |

**Training:**
- Time-based split: train on 2008–2014, test on 2015/2016
- Class-weighted CrossEntropyLoss to handle imbalance
- Adam optimiser + ReduceLROnPlateau scheduler
- Early stopping (patience=15 epochs)
- Optuna hyperparameter search (30 trials, TPE sampler)

### Results (Expected)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive (Always Home Win) | ~46% | ~21% |
| Random Forest | ~52-54% | ~42-45% |
| Deep MLP | ~53-56% | ~44-47% |
| LSTM | ~52-55% | ~43-46% |

*Exact values depend on training run; see `outputs/reports/metrics_report.json`.*

---

## Project Structure

```
Football-prediction PAAI project/
├── data/
│   ├── raw/
│   │   └── database.sqlite        ← download from Kaggle
│   └── processed/
│       ├── matches_clean.parquet
│       └── features.parquet
├── notebooks/
│   └── exploration.ipynb          ← EDA notebook
├── src/
│   ├── preprocessing.py           ← DB loading, cleaning, label encoding
│   ├── features.py                ← Feature engineering (form, H2H, etc.)
│   ├── model.py                   ← MLP, LSTM, BaselineModel definitions
│   ├── train.py                   ← Training pipeline + Optuna tuning
│   ├── evaluate.py                ← Metrics, confusion matrices, calibration
│   └── visualize.py               ← All plots saved to outputs/plots/
├── outputs/
│   ├── models/                    ← Saved model weights + Optuna study
│   ├── plots/                     ← All generated figures
│   └── reports/                   ← metrics_report.json, comparison_table.csv
├── app.py                         ← Streamlit prediction app (bonus)
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data
Download `database.sqlite` from [Kaggle](https://www.kaggle.com/datasets/hugomathien/soccer) and place it at:
```
data/raw/database.sqlite
```

### 3. Run the full pipeline
```bash
# Step 1: Preprocess raw data
python src/preprocessing.py

# Step 2: Build feature matrix (takes ~5-10 min for 25k matches)
python src/features.py

# Step 3: Train all models (--skip-optuna for faster run, --lstm to include LSTM)
python src/train.py --skip-optuna

# Step 4: Evaluate models
python src/evaluate.py

# Step 5: Generate all plots
python src/visualize.py
```

### 4. Launch Streamlit app (bonus)
```bash
streamlit run app.py
```

---

## Outputs

After running the full pipeline, you will have:

- `outputs/models/mlp.pt` — trained MLP weights
- `outputs/models/baseline_rf.pkl` — trained Random Forest
- `outputs/reports/metrics_report.json` — all metrics
- `outputs/reports/comparison_table.csv` — model comparison
- `outputs/plots/` — all visualizations (loss curves, confusion matrices, feature importance, calibration, etc.)

---

## Reproducibility

- Random seed: `42` (set in `train.py` for both PyTorch and NumPy)
- Time-based split: test set is always the 2015/2016 season
- All normalisation statistics are saved to `outputs/models/norm_stats.pkl`

---

## Limitations & Future Work

See `PHASE8_CONCLUSIONS.md` for a full discussion.

Brief summary:
- Football has irreducible noise — no model will exceed ~60% accuracy on unseen data
- Player-level features (lineups, injuries) would significantly improve predictions
- Real-time data pipeline + API integration for live predictions
- Transformer-based sequence model as MLP/LSTM successor
