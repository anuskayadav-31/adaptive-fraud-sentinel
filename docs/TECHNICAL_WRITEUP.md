# Technical write-up: Adaptive Fraud Sentinel (AIML-02)

Educational prototype for fraud / anomaly detection with class imbalance handling, a supervised vs unsupervised comparison, Precision-Recall evaluation, and basic drift alerting. It is **not** a production banking system.

**Experiment status:** results below come from a real run of `python src/pipeline.py` on the full `creditcard.csv` (284,807 rows). Exact JSON is in `results/metrics.json`.

## Problem

A model trained once on historical payments can silently degrade when spend patterns, seasonality, or fraud tactics change. The prototype therefore treats **model health** (distribution and score shift) as part of detection, not as an afterthought.

## Data

Kaggle Credit Card Fraud Detection: `Time`, PCA features `V1`–`V28`, `Amount`, binary `Class`. Extreme imbalance (~0.172% fraud) and only about two days of data. PCA features are already anonymized, so domain rules on merchant category or device are not available.

**Split.** Rows are sorted by `Time`. The first 56% is training, the next 14% is validation, the last 30% is a later test window. The operating threshold is chosen on validation only and then frozen for the test window.

**Features.** Raw `Time` is dropped for modeling (it would mostly encode “earlier vs later” under a temporal split). We add `hour_of_day = (Time // 3600) % 24` and `log_amount = log1p(Amount)`, and keep `Amount` plus `V1`–`V28`.

## Class imbalance (requirement 1)

Naive random oversampling of fraud rows is **not** used. Duplicating the 492 frauds inflates tree splits on copies of the same points and overstates validation performance.

SMOTE in this PCA space is also a poor default: synthetic neighbors in rotated anonymous coordinates have no payment-semantics guarantee.

**Choice:** XGBoost `scale_pos_weight = n_legitimate / n_fraud` computed on the **training window**. Every original transaction is used once; the loss up-weights the minority class. This is class-weighted learning, not resampling.

The Isolation Forest is trained only on **legitimate** training rows so it models “normal” behavior rather than trying to resample fraud.

## Models (requirements 2–3)

| Approach | Role |
| --- | --- |
| **XGBoost** (supervised) | Uses fraud labels. `eval_metric=aucpr`, early stopping on validation PR-AUC. Outputs a fraud probability. |
| **Isolation Forest** (unsupervised) | No fraud labels at fit time (legit-only). `decision_function` is inverted so **higher = more anomalous**. That score is a **rank**, not a calibrated probability. |

Comparison is on the same later window using ranking metrics. Isolation Forest is used instead of an autoencoder to keep the prototype small, CPU-friendly, and reproducible.

**Observed (test window):** XGBoost PR-AUC **0.783** vs Isolation Forest **0.032**. Labels clearly help when fraud in the later window still resembles labeled historical fraud. Isolation Forest remains the fallback for attacks that never appeared as `Class=1` in training; it should not be expected to match a strong supervised ranker on this dataset.

## Why PR-AUC rather than relying on ROC-AUC (requirement 4)

With ~99.8% negatives, a trivial “never flag fraud” rule has a huge true-negative rate. ROC-AUC mixes that easy negative class into the curve, so scores can look strong while precision on fraud is poor.

**PR-AUC** (`average_precision_score`) is the area under precision vs recall. It only moves when the model ranks actual frauds above legitimate traffic. The no-skill PR-AUC equals **class prevalence** (~0.00172), which is the honest baseline.

The pipeline still **computes ROC-AUC** so the two numbers can be compared. Reporting should lead with PR-AUC; ROC-AUC is diagnostic, not the operating criterion.

**Observed:** Isolation Forest test ROC-AUC is **0.942** while PR-AUC is **0.032** (no-skill PR ≈ 0.0013). XGBoost ROC-AUC is 0.975 with PR-AUC 0.783. ROC would have called the two models “close”; PR-AUC shows they are not.

## Drift monitoring and alerts (requirements 5–6, bonus 8)

**Feature drift.** Two-sample Kolmogorov–Smirnov tests compare a sample of the training window with a sample of the later window (and a **simulated** later window). With tens of thousands of rows, tiny shifts get tiny p-values, so a feature is marked drifted only if:

- KS p-value \< 0.01, **and**
- KS statistic ≥ 0.05 (effect-size floor).

Population Stability Index (PSI) is also recorded; mean PSI ≥ 0.25 is a second alert trigger (common “investigate” band).

**Score drift.** The same KS + floor rule is applied to XGBoost probabilities (train vs later window, and train vs scores on the simulated features). This is a simple **model-confidence** monitor.

**Alert.** Flag investigation / retraining if the share of drifted features is ≥ 20%, or mean PSI ≥ 0.25, or the score distribution is drifted.

**Bonus simulation.** Amount is scaled by 1.8 with noise; `V1`, `V4`, and `V14` are rescaled. `log_amount` is recomputed.

**Observed:** the natural later window already alerts (27/31 features, mean PSI 0.39, score KS 0.167). That is expected on a two-day file: `hour_of_day` and spend mix differ between the first 56% and the last 30%. The simulated shift still raises the drifted-feature count to 29/31, mean PSI to 0.42, and score KS to 0.198, so the monitor reacts to the extra artificial drift rather than only firing on the simulation.

## Cost of errors and threshold (requirement 7)

A default 0.5 probability cutoff ignores operations cost.

- **False positive (FP):** block or friction on a legitimate customer. Cost **1** in the prototype (unit of ops/customer cost).
- **False negative (FN):** miss fraud. Cost **8** as an illustrative stand-in for chargeback / loss relative to one FP. This is **not** a real bank loss table.

The pipeline searches thresholds in `[0.05, 0.95]` on **validation** to minimize `FP × 1 + FN × 8`, then applies that threshold to the **test** window without retuning.

**Observed:** the cost minimum on validation is at **t = 0.895** (precision 0.81, recall 0.75, 5 FP, 7 FN). Frozen on test: precision 0.87, recall 0.69, 11 FP, 33 FN, 75 TP, estimated cost 275. The high threshold is a precision-first operating point: with `scale_pos_weight` already boosting fraud scores, 0.5 would over-flag; the cost grid pays for extra reviews only when they recover enough missed fraud. On this split, 0.895 still holds recall near 0.69 on the later window.

## Limitations

- Two days of data cannot represent seasonality or multi-month tactic change.
- PCA features hide raw behavioral signals.
- Labels are treated as ground truth; real fraud labels arrive late and are noisy.
- Drift simulation is artificial by design.
- No automatic model swap in production; the alert is an investigation flag.

## How to reproduce

See the repository README. After a successful run, interpret `results/metrics.json`, the two drift CSVs, and `results/figures/`.
