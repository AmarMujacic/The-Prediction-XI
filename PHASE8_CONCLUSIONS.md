# Phase 8 — Conclusions & Future Work

---

## Summary of Results

We built a complete deep learning pipeline to predict football match outcomes across 11 European leagues (2008–2016). The final test set was the 2015/2016 season — data the models never saw during training.

| Model | Accuracy | Macro F1 | Draw F1 |
|---|---|---|---|
| Naive (Always Home Win) | ~46% | ~21% | 0.00 |
| Random Forest | ~53% | ~43% | ~28% |
| Deep MLP | ~55% | ~46% | ~31% |
| LSTM | ~54% | ~44% | ~30% |

---

## Linking Results to the Problem

**1. We beat the trivial baseline significantly.**
The naive model (always predict Home Win) reaches ~46% accuracy but achieves zero F1 on Draw and Away Win. Our MLP improves Macro F1 from 21% to ~46% — more than doubling the model's ability to distinguish all three outcomes.

**2. Class imbalance is the hardest problem to solve.**
Even with class-weighted loss and SMOTE-style rebalancing, the Draw class consistently achieves the lowest F1 (~31%). This is expected: Draws are inherently unpredictable, with low signal in team statistics alone. Bookmaker odds — which aggregate vast amounts of market information — also misprice Draws more than any other outcome.

**3. Deep learning vs. Random Forest.**
The MLP outperforms the Random Forest by ~2–3% in accuracy and ~3% in Macro F1. The gap is modest but consistent across all evaluation runs. The LSTM shows similar performance to the MLP despite its greater architectural complexity, suggesting that the simple form statistics we use as sequence input do not contain strong sequential patterns beyond what a rolling-window aggregation already captures.

**4. Calibration matters.**
The MLP produces better-calibrated probability estimates than the Random Forest. This is critical for real-world use: a decision-maker who sees "P(Draw) = 0.35" needs to trust that the model is right ~35% of the time in that confidence band.

---

## Where the Model Fails

**High-confidence misclassifications:**
When the MLP assigns >70% probability to an outcome that does not occur, this almost always involves:
- Cup competitions or "dead rubber" league matches (motivation effects not captured)
- Teams with very recent managerial changes (form stats lag actual quality shift)
- Extreme weather or neutral venue matches

**Systematic Draw underestimation:**
All models underestimate Draw probability, outputting P(Draw) < 0.20 in most cases even when the true draw rate is ~26%. This reflects a real difficulty: draw-producing match dynamics (defensive tactics, fatigue, equal strength) are poorly encoded in our current features.

**League heterogeneity:**
Leagues differ significantly in home advantage and draw rates (e.g., Serie A has historically more draws than the Premier League). Although we include a league encoding, the model does not fully leverage these cross-league structural differences.

---

## Limitations

| Limitation | Impact | Severity |
|---|---|---|
| No player-level data (lineups, injuries) | Misses single-player-impact events | High |
| No real-time data | Cannot use current-season momentum | Medium |
| Feature representation is static per match | No within-season dynamic adjustment | Medium |
| Eight seasons only | Recent tactical shifts (high press, etc.) not represented | Low-Medium |
| SQLite dataset frozen at 2016 | Cannot be deployed for live predictions without new data source | High for deployment |

---

## What We Learned

1. **Feature engineering dominates model choice.** A well-engineered Random Forest is competitive with a tuned MLP. The deep model's advantage comes from its ability to learn non-linear interactions between features — particularly between FIFA ratings and recent form.

2. **Class weighting is non-negotiable.** Without it, all models collapse to predicting only Home Win and Away Win, completely ignoring Draws.

3. **Early stopping prevents severe overfitting.** The MLP validation loss diverges from training loss after approximately epoch 40–60; early stopping with patience=15 is effective.

4. **Optuna adds meaningful value.** The Optuna-tuned MLP consistently outperforms the hand-tuned default architecture by ~1–2% Macro F1, with the biggest gain coming from learning the optimal dropout rate and layer width combination.

---

## Suggested Improvements

### Short-term (implementable now)
- **Player-level features**: squad fitness score, key player availability (binary injury flag)
- **Betting odds as meta-features**: Bet365 or market-consensus odds carry vast implicit information
- **More data**: European leagues through 2024 via football-data.co.uk or Statsbomb open data
- **SHAP values**: Replace permutation importance with SHAP for per-prediction explanability

### Medium-term (research directions)
- **Gradient boosting ensemble**: XGBoost/LightGBM as a third model; ensemble with MLP via stacking
- **Transformer for sequence modelling**: Multi-head attention on season-long match history, superior to LSTM for long-term dependencies
- **Graph Neural Networks**: Model the league as a graph (teams = nodes, matches = directed edges)

### Long-term (production vision)
- **Real-time data pipeline**: Ingest live API feeds (e.g. football-data.org) before each matchday
- **Automated retraining**: Retrain weekly on the latest season data to stay current
- **Uncertainty quantification**: Bayesian deep learning to output not just probabilities but confidence intervals
- **Live dashboard**: Extend the Streamlit app with a full match-day schedule and probability tracker

---

## Final Reflection

Football prediction is a textbook example of a noisy, high-variance classification problem. No model will reliably predict individual match outcomes — the sport would lose its appeal if it could. What deep learning can do is identify systematic biases in team strength, form, and historical records that shift the probability distribution of outcomes in a measurable, actionable way.

This project demonstrates that a principled pipeline — clean data, thoughtful feature engineering, regularised deep learning, and rigorous evaluation — can extract meaningful signal from the noise of football. The resulting system does not tell you who will win; it tells you what the evidence suggests, and quantifies the uncertainty around that belief.

That is precisely what an intelligent sports analytics system should do.
