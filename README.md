# Adaptive Fraud Sentinel

**AIML-02 — Fraud/Anomaly Detection with Drift Monitoring**

Adaptive Fraud Sentinel is a small ML monitoring prototype for fintech transaction screening. It trains a supervised fraud model and an unsupervised anomaly detector, evaluates them with Precision-Recall AUC, and checks whether feature or score distributions have shifted enough to justify investigation or retraining.

This repository is an educational prototype, not a production financial-risk system.

## Why this is different from a normal fraud classifier

A static classifier assumes tomorrow’s transactions resemble yesterday’s. Payment traffic does not: spend, attack patterns, and seasonality change. This project treats **model health** as part of fraud detection.

### Example use cases

- Payment gateways: screen transactions while watching customer-behavior shift.
- Digital banks: surface emerging anomaly patterns before labels catch up.
- Wallets / UPI-like rails: rank novel attacks that were absent from training labels.
- Risk operations: set a fraud threshold from the relative cost of blocking a legitimate customer versus missing fraud.

## Requirements covered

- **Imbalance:** XGBoost `scale_pos_weight = n_legit / n_fraud` on the training window. Naive random oversampling is not used (see [docs/TECHNICAL_WRITEUP.md](docs/TECHNICAL_WRITEUP.md)).
- **Supervised model:** XGBoost with `eval_metric=aucpr` and early stopping on a time-ordered validation window.
- **Unsupervised model:** Isolation Forest trained on legitimate transactions only. Scores are ranking-only, not probabilities.
- **Evaluation:** PR-AUC for both models, plus ROC-AUC and a no-skill PR baseline equal to fraud prevalence.
- **Threshold:** chosen on validation by minimizing `FP × 1 + FN × 8`, then frozen for the later test window.
- **Drift:** Kolmogorov–Smirnov tests with a KS effect-size floor, plus PSI and a score-distribution check.
- **Alert:** investigation / retraining flag if ≥20% of monitored features are drifted, mean PSI ≥ 0.25, or model scores have shifted.
- **Bonus:** Amount and selected PCA features are shifted on purpose; monitoring is expected to fire on that simulated window.

## Dataset

Use the Kaggle **Credit Card Fraud Detection** dataset (`Time`, `V1`–`V28`, `Amount`, `Class`). About 284,807 rows and 492 frauds (~0.172%). The CSV is not committed (≈150 MB).

Download instructions and schema: [data/README.md](data/README.md).

Place the file at:

```text
data/creditcard.csv
```

## Setup

Use **Python 3.11** (3.14 may not yet have NumPy/SciPy wheels).

```bash
python3.11 -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python src/pipeline.py
```

Outputs (created by a real run; not invented in this README):

- `results/metrics.json`
- `results/drift_report.csv`
- `results/simulated_drift_report.csv`
- `results/figures/pr_curves.png`
- `results/figures/threshold_cost.png`
- `results/figures/drift_natural.png`
- `results/figures/drift_simulated.png`

If those files are absent, the experiment has not been executed in this checkout.

## Architecture

```text
Transactions (creditcard.csv)
        |
        v
Feature prep (hour_of_day, log_amount; drop raw Time)
        |
Time-ordered split: train 56% / val 14% / later test 30%
        |
   +----+----------------+
   |                     |
   v                     v
XGBoost              Isolation Forest
(scale_pos_weight)   (fit on Class=0 only)
   |                     |
   +----------+----------+
              v
     PR-AUC + ROC-AUC
              |
   Cost-sensitive threshold
        (tune on val, freeze on test)
              |
   KS + PSI + score-shift monitor
              |
      +-------+-------+
      |               |
   below bound     drift alert
      |               |
   continue     investigation /
                 retraining flag
```

## Threshold and cost policy

A 0.5 probability cutoff is not used as the business rule. False positives are priced as customer friction (cost 1). False negatives are priced higher (cost 8) as a stand-in for missed-fraud / chargeback loss. The ratio is **illustrative**, not a bank’s real loss table. Details: [docs/TECHNICAL_WRITEUP.md](docs/TECHNICAL_WRITEUP.md).

## Results

From a local run of `python src/pipeline.py` on the full Kaggle CSV (284,807 rows, 492 frauds). Source of truth: [results/metrics.json](results/metrics.json).

Later **test** window (85,443 transactions, prevalence 0.126%):

| Model | PR-AUC | ROC-AUC | No-skill PR (prevalence) |
| --- | ---: | ---: | ---: |
| XGBoost (supervised) | **0.783** | 0.975 | 0.00126 |
| Isolation Forest | 0.032 | 0.942 | 0.00126 |

Isolation Forest **ROC-AUC looks strong (0.94)** while **PR-AUC is only 0.032**. That gap is why this project does not rely on ROC-AUC under extreme imbalance.

Cost-sensitive threshold **0.895** (tuned on validation, frozen on test): test precision 0.87, recall 0.69, 11 false positives, 33 false negatives, 75 true positives. `scale_pos_weight` on train was 447 (356 frauds / 159,135 legit).

Drift alerts:

- **Natural** later window: 27/31 features marked drifted (rate 87%, mean PSI 0.39) → alert. Score-distribution KS = 0.167.
- **Simulated** Amount / V-feature shift: 29/31 features (rate 94%, mean PSI 0.42) → alert. Score KS = 0.198.

The two-day file already moves (different hours and spend mix in the later slice). The bonus simulation still **increases** both the drifted-feature count and the score KS, so monitoring is sensitive to an artificial shift on top of the natural one.

## Important limitation

The public dataset covers about two days and uses anonymized PCA features. The drift experiment is deliberately simulated. A production monitor would also track label delay, calibration, segment performance, and confirmed fraud outcomes.

## License / data

Code in this repository is for the AWS Student Builder Group technical round. The credit-card CSV remains subject to its Kaggle source terms and is not redistributed here.
