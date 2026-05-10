# NHL Tracking Analytics Platform

A full-stack hockey analytics system that predicts scoring probability, offensive zone entry success, and player movement patterns using NHL play-by-play and tracking data.

---

## Project Structure

```
nhl-analytics/
├── data/
│   ├── raw/              # Raw API pulls (play-by-play, shifts, rosters)
│   └── processed/        # Cleaned, feature-engineered datasets
├── features/
│   ├── shot_features.py  # Shot-based xG features
│   ├── zone_features.py  # Zone entry features
│   └── sequence.py       # Spatio-temporal sequence construction
├── models/
│   ├── xg_model.py       # Expected goals (XGBoost + logistic baseline)
│   ├── zone_entry.py     # Zone entry success classifier
│   └── sequence_model.py # LSTM scoring chance predictor
|
├── nhl-frontend/         # Frontend folder handling the dashboard displays
|
├── utils/
│   ├── data_loader.py    # NHL API + MoneyPuck ingestion
│   ├── preprocessing.py  # Cleaning, encoding, normalization
│   └── evaluation.py     # Model evaluation, calibration, logging
├── reports/
│   └── generate_report.py # Stakeholder-ready HTML/PDF reports
├── notebooks/            # EDA and model development notebooks
├── tests/                # Unit tests for features and models
├── dashboard/            # React dashboard (see /dashboard)
├── pipeline.py           # End-to-end run script
├── config.py             # Central config (paths, hyperparameters)
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull data (MoneyPuck + NHL API)
python pipeline.py --stage ingest --seasons 2021 2022 2023

# 3. Feature engineering
python pipeline.py --stage features

# 4. Train all models
python pipeline.py --stage train

# 5. Evaluate and generate report
python pipeline.py --stage report --season 2023
```

---

## Models

### 1. Expected Goals (xG)
Predicts the probability a shot results in a goal.

- **Features**: shot distance, shot angle, shot type, game state (score, period, strength), prior event type and distance, rebounds, rush shots, shooter handedness
- **Architecture**: XGBoost classifier (primary) + logistic regression (calibration baseline)
- **Evaluation**: Log-loss, Brier score, AUC, calibration curve
- **Benchmark**: MoneyPuck public xG model

### 2. Zone Entry Success
Predicts whether a controlled zone entry results in a shot attempt within 10 seconds.

- **Features**: entry type (carry-in vs dump-in), entry location, score state, line combination, opposing defensive pair, zone time preceding entry
- **Architecture**: Gradient boosted classifier (XGBoost)
- **Evaluation**: Precision/recall, expected vs actual shot generation

### 3. Sequence LSTM — Scoring Chance Predictor
Predicts whether a possession sequence becomes a scoring chance within 10 seconds, using ordered event sequences.

- **Input**: Variable-length sequences of events (passes, shot attempts, zone entries, faceoffs) with spatial coordinates + game state
- **Architecture**: Bidirectional LSTM → attention layer → dense head
- **Evaluation**: AUC, temporal calibration, comparison to non-sequential xG baseline

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| [MoneyPuck](https://moneypuck.com/data.htm) | Shots, xG, game-level stats | Free CSV download |
| [NHL API](https://api-web.nhle.com) | Play-by-play, rosters, shifts | Public REST API |
| [Evolving Hockey](https://evolving-hockey.com) | RAPM, GAR, contract data | Subscription |
| [Hockey Reference](https://hockey-reference.com) | Historical stats | Scraping (see utils/data_loader.py) |

---

## Key Design Decisions

**Why XGBoost for xG?** Tree-based models handle hockey's interaction effects naturally (e.g. rebound shots from sharp angles behave very differently from regular shots). We add a logistic regression baseline for calibration comparison.

**Why LSTM for sequences?** Possession sequences are ordered and variable-length — ideal for recurrent models. The bidirectional architecture lets the model use both preceding and following context when predicting at each timestep. Attention weights become interpretable: you can see which events the model weights most when predicting a chance.

**Why separate xG and sequence models?** xG answers "given this shot, what's the probability it's a goal?" The sequence model answers "given this possession unfolding, will it produce a shot?" They're complementary and serve different analytical questions.

---

## Stakeholder Reporting

Reports are generated as self-contained HTML files readable by non-technical staff:

```bash
python reports/generate_report.py --team TOR --season 2023 --output reports/TOR_2023.html
```

Reports include:
- Team xG for/against over the season (interactive Plotly chart)
- Zone entry efficiency by line combination
- Top and bottom performers by xG differential
- Possession sequence heatmaps
- Plain-language interpretation of key findings

---

## Resume Bullets (as built)

- Built predictive models using NHL play-by-play and tracking data to estimate scoring probability and offensive zone entry success
- Developed spatio-temporal sequence models using bidirectional LSTMs and gradient boosting to analyze player movement and puck possession patterns
- Automated data ingestion, feature engineering, and model validation pipelines across 3 NHL seasons and 200k+ shot events
- Produced interactive analytical reports evaluating player decision-making, transition efficiency, and defensive coverage for non-technical hockey operations staff
