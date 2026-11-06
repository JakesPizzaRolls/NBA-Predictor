# NBA Predictor v1.0
### Machine Learning for NBA Game Outcome Prediction

An end-to-end machine learning project for predicting NBA game
outcomes using historical basketball data, player-level
statistics, rolling performance metrics, and Elo ratings.

The final model combines logistic regression with an ensemble
of three PyTorch neural networks.

## Model Performance

Historical out-of-sample evaluation (2019–2022):

| Metric | Result |
|--------|--------|
| Accuracy | 64.27% |
| ROC-AUC | 0.6883 |
| Log Loss | 0.6333 |
| Brier Score | 0.2214 |

These are historical backtest results, not live-season results.

## Architecture

The final model is a weighted ensemble:

- 50% logistic regression
- 50% average of three neural networks

Neural network architecture:

Input (177 features)
→ Dense (256)
→ Dense (128)
→ Dense (64)
→ Output (1 probability)

Neural networks are implemented using PyTorch.

## Features

177 pregame features covering:

- Historical team performance
- Rolling 5-, 10-, and 20-game statistics
- Offensive and defensive efficiency
- Player-level historical contributions
- Elo rating differentials
- Rest and scheduling information
- Home-court advantage

Features are designed to use only information available
before each game's tipoff.

## Dataset

The project was developed using historical NBA game data
spanning 1946 through 2023.

The frozen production model was trained on approximately
60,192 historical games.

Large raw datasets are excluded from this repository.

## Repository Structure

- `src/` — Training and feature engineering
- `models/` — Frozen ensemble and prediction interface
- `evaluation/` — Historical model evaluation
- `predictions/` — Saved prospective predictions
- `RESEARCH_NOTES.md` — Original research documentation

## Installation

Requires Python and the dependencies in requirements.txt.

```bash
git clone https://github.com/YOUR_USERNAME/nba-predictor.git
cd nba-predictor

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Predictions

The repository includes a saved feature snapshot for
November 6, 2026.

```bash
python models/predict.py \
  --input predictions/2026-11-06_features.csv \
  --output predictions/output.csv
```

The script loads the frozen ensemble and produces
home/away win probabilities for each matchup.

## Methodology

The project uses chronological training and evaluation
rather than random train/test splitting.

Experiments included:

- Neural network architecture comparisons
- Training-window comparisons
- Feature ablation studies
- Player-level feature evaluation
- Elo-based feature engineering
- Walk-forward evaluation

## Limitations

- Historical player features are approximate.
- Current injuries and confirmed lineups are not fully verified.
- Model probabilities are not guaranteed to be calibrated.
- Historical performance does not guarantee future accuracy.
- Historical feature construction and prospective feature
  collection may differ in coverage and completeness.

## Disclaimer

Research and educational purposes only.
Not financial or betting advice.
