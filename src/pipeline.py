"""Adaptive Fraud Sentinel: supervised + unsupervised fraud detection with drift monitoring."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RANDOM_STATE = 42
FP_COST = 1.0
FN_COST = 8.0
DRIFT_P_ALPHA = 0.01
DRIFT_KS_FLOOR = 0.05
DRIFT_ALERT_RATE = 0.20
PSI_ALERT = 0.25


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(
            "Place Kaggle creditcard.csv in data/ before running. See data/README.md."
        )
    df = pd.read_csv(DATA)
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw Time with hour-of-day; add log-amount. Time spans ~2 days from t=0."""
    out = df.copy()
    out["hour_of_day"] = ((out["Time"] // 3600) % 24).astype(float)
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"Time", "Class"}
    return [c for c in df.columns if c not in drop]


def temporal_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Early window = train + val (time-ordered); later window = test."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.56)
    val_end = int(n * 0.70)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_columns(df)
    return df[cols], df["Class"].astype(int)


def make_supervised(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)

    n_fraud = int(y_train.sum())
    n_legit = int(len(y_train) - n_fraud)
    scale_pos_weight = n_legit / max(n_fraud, 1)

    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    X_train_p = prep.fit_transform(X_train)
    X_val_p = prep.transform(X_val)
    X_test_p = prep.transform(X_test)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=4,
        early_stopping_rounds=40,
    )
    model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)

    val_proba = model.predict_proba(X_val_p)[:, 1]
    test_proba = model.predict_proba(X_test_p)[:, 1]
    train_proba = model.predict_proba(X_train_p)[:, 1]
    return {
        "prep": prep,
        "model": model,
        "scale_pos_weight": float(scale_pos_weight),
        "n_train_fraud": n_fraud,
        "n_train_legit": n_legit,
        "X_train": X_train,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_proba": train_proba,
        "val_proba": val_proba,
        "test_proba": test_proba,
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    prevalence = float(y_true.mean())
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "prevalence": prevalence,
        "pr_auc_no_skill": prevalence,
    }


def isolation_forest_scores(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Fit on legitimate train rows only. Scores are ranking-only (higher = more anomalous)."""
    cols = feature_columns(train)
    normal = train.loc[train["Class"] == 0, cols]
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    A = prep.fit_transform(normal)
    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    detector.fit(A)
    def score(frame: pd.DataFrame) -> np.ndarray:
        raw = -detector.decision_function(prep.transform(frame[cols]))
        return np.asarray(raw, dtype=float)

    return score(val), score(test)


def choose_threshold(y_true, proba, fp_cost=FP_COST, fn_cost=FN_COST) -> dict:
    """Minimize expected operational cost on a held-out validation window."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    best = None
    curve = []
    for t in np.linspace(0.05, 0.95, 181):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = float(fp * fp_cost + fn * fn_cost)
        row = {
            "threshold": float(round(t, 4)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "estimated_cost": cost,
        }
        curve.append(row)
        if best is None or cost < best["estimated_cost"]:
            best = row
    return {"best": best, "curve": curve}


def apply_threshold(y_true, proba, threshold: float, fp_cost=FP_COST, fn_cost=FN_COST) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "estimated_cost": float(fp * fp_cost + fn * fn_cost),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    a_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    e_frac = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_frac = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = feature_columns(reference)
    rows = []
    for f in features:
        a = reference[f].dropna().to_numpy()
        b = current[f].dropna().to_numpy()
        if len(a) < 20 or len(b) < 20:
            continue
        stat, p = ks_2samp(a, b)
        psi = population_stability_index(a, b)
        drifted = bool(p < DRIFT_P_ALPHA and stat >= DRIFT_KS_FLOOR)
        rows.append(
            {
                "feature": f,
                "ks_stat": float(stat),
                "p_value": float(p),
                "psi": float(psi),
                "drifted": drifted,
            }
        )
    out = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    n = max(len(out), 1)
    drifted_n = int(out["drifted"].sum()) if len(out) else 0
    rate = drifted_n / n
    mean_psi = float(out["psi"].mean()) if len(out) else 0.0
    alert = bool(rate >= DRIFT_ALERT_RATE or mean_psi >= PSI_ALERT)
    summary = {
        "n_features": int(len(out)),
        "n_drifted": drifted_n,
        "drift_rate": float(rate),
        "mean_psi": mean_psi,
        "retraining_alert": alert,
        "rule": (
            f"Alert if share of features with KS p<{DRIFT_P_ALPHA} and "
            f"KS>={DRIFT_KS_FLOOR} is >={DRIFT_ALERT_RATE:.0%}, or mean PSI>={PSI_ALERT}."
        ),
    }
    return out, summary


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Shift Amount and selected latent features to mimic later-window concept/data drift."""
    later = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    amount_std = float(later["Amount"].std() or 1.0)
    later["Amount"] = later["Amount"] * 1.8 + rng.normal(0, amount_std * 0.08, len(later))
    later["Amount"] = later["Amount"].clip(lower=0)
    later["log_amount"] = np.log1p(later["Amount"])
    for col, scale in [("V1", 1.25), ("V4", 0.75), ("V14", 1.35)]:
        if col in later.columns:
            later[col] = later[col] * scale
    return later


def score_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict:
    a = np.asarray(reference_scores, dtype=float)
    b = np.asarray(current_scores, dtype=float)
    stat, p = ks_2samp(a, b)
    psi = population_stability_index(a, b)
    drifted = bool(p < DRIFT_P_ALPHA and stat >= DRIFT_KS_FLOOR)
    return {
        "ks_stat": float(stat),
        "p_value": float(p),
        "psi": float(psi),
        "drifted": drifted,
    }


def plot_pr_curves(y_test, xgb_scores, iforest_scores, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in (("XGBoost", xgb_scores), ("Isolation Forest", iforest_scores)):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, color="gray", linestyle="--", label=f"No-skill (prevalence={prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall on later test window")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cost_curve(curve: list[dict], chosen: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ts = [r["threshold"] for r in curve]
    costs = [r["estimated_cost"] for r in curve]
    ax.plot(ts, costs, label="Validation cost")
    ax.axvline(chosen["threshold"], color="red", linestyle="--", label=f"Chosen t={chosen['threshold']:.3f}")
    ax.set_xlabel("Fraud probability threshold")
    ax.set_ylabel(f"Cost (FP×{FP_COST:g} + FN×{FN_COST:g})")
    ax.set_title("Cost-sensitive threshold search (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_drift_bars(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    view = table.head(20).iloc[::-1]
    colors = ["#c0392b" if d else "#2980b9" for d in view["drifted"]]
    ax.barh(view["feature"], view["ks_stat"], color=colors)
    ax.axvline(DRIFT_KS_FLOOR, color="gray", linestyle="--", label=f"KS floor={DRIFT_KS_FLOOR}")
    ax.set_xlabel("KS statistic")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    raw = load_data()
    df = add_features(raw)
    train, val, test = temporal_splits(df)

    supervised = make_supervised(train, val, test)
    if_val, if_test = isolation_forest_scores(train, val, test)

    xgb_test_metrics = ranking_metrics(supervised["y_test"], supervised["test_proba"])
    xgb_val_metrics = ranking_metrics(supervised["y_val"], supervised["val_proba"])
    if_test_metrics = ranking_metrics(supervised["y_test"], if_test)
    if_val_metrics = ranking_metrics(supervised["y_val"], if_val)

    threshold_search = choose_threshold(supervised["y_val"], supervised["val_proba"])
    chosen = threshold_search["best"]
    test_operating_point = apply_threshold(
        supervised["y_test"], supervised["test_proba"], chosen["threshold"]
    )

    rng = np.random.default_rng(RANDOM_STATE)
    ref_n = min(50000, len(train))
    later_n = min(50000, len(test))
    reference = train.iloc[rng.choice(len(train), size=ref_n, replace=False)]
    later = test.iloc[rng.choice(len(test), size=later_n, replace=False)]
    natural_table, natural_summary = drift_report(reference, later)
    simulated = simulate_drift(later)
    simulated_table, simulated_summary = drift_report(reference, simulated)

    # Score-distribution monitor: train scores vs later-window model confidence.
    score_natural = score_drift(supervised["train_proba"], supervised["test_proba"])
    sim_later_features = simulated[feature_columns(simulated)]
    sim_scores = supervised["model"].predict_proba(
        supervised["prep"].transform(sim_later_features)
    )[:, 1]
    score_simulated = score_drift(supervised["train_proba"], sim_scores)

    plot_pr_curves(
        supervised["y_test"].to_numpy(),
        supervised["test_proba"],
        if_test,
        FIGURES / "pr_curves.png",
    )
    plot_cost_curve(threshold_search["curve"], chosen, FIGURES / "threshold_cost.png")
    plot_drift_bars(natural_table, "Natural later-window feature drift (KS)", FIGURES / "drift_natural.png")
    plot_drift_bars(simulated_table, "Simulated drift feature KS", FIGURES / "drift_simulated.png")

    report = {
        "dataset_rows": int(len(df)),
        "fraud_count": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "split": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "method": "time-ordered 56/14/30; threshold tuned on validation only",
        },
        "imbalance": {
            "method": "XGBoost scale_pos_weight = n_legit / n_fraud on the training window",
            "scale_pos_weight": supervised["scale_pos_weight"],
            "n_train_fraud": supervised["n_train_fraud"],
            "n_train_legit": supervised["n_train_legit"],
            "not_used": "naive random oversampling / SMOTE",
        },
        "supervised_xgboost": {
            "validation": xgb_val_metrics,
            "test": xgb_test_metrics,
        },
        "isolation_forest": {
            "note": "Trained on legitimate train rows only. Scores are inverted decision_function ranks, not calibrated probabilities.",
            "validation": if_val_metrics,
            "test": if_test_metrics,
        },
        "why_pr_auc": (
            "ROC-AUC can stay high when the negative class dominates because true-negative rate "
            "is easy to keep high. PR-AUC is the area under precision vs recall and is driven by "
            "the rare positive class; the no-skill baseline equals prevalence (~0.17%)."
        ),
        "cost_policy": {
            "fp_cost": FP_COST,
            "fn_cost": FN_COST,
            "justification": (
                "FP is treated as one unit of customer friction (blocked legitimate payment). "
                "FN is treated as 8x that cost as a stand-in for chargeback/loss on missed fraud. "
                "This is an illustrative operations policy, not a bank loss table."
            ),
            "validation_operating_point": chosen,
            "test_operating_point": test_operating_point,
        },
        "drift": {
            "natural_feature_shift": natural_summary,
            "simulated_feature_shift": simulated_summary,
            "natural_score_shift": score_natural,
            "simulated_score_shift": score_simulated,
            "natural_retraining_alert": bool(
                natural_summary["retraining_alert"] or score_natural["drifted"]
            ),
            "simulated_retraining_alert": bool(
                simulated_summary["retraining_alert"] or score_simulated["drifted"]
            ),
        },
        "artifacts": {
            "metrics": "results/metrics.json",
            "natural_drift_csv": "results/drift_report.csv",
            "simulated_drift_csv": "results/simulated_drift_report.csv",
            "figures": [
                "results/figures/pr_curves.png",
                "results/figures/threshold_cost.png",
                "results/figures/drift_natural.png",
                "results/figures/drift_simulated.png",
            ],
        },
    }

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    natural_table.to_csv(RESULTS / "drift_report.csv", index=False)
    simulated_table.to_csv(RESULTS / "simulated_drift_report.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
