"""
src/evaluate.py -- Kestrel Warranty Fraud Detection
Standalone evaluation script for saved models.

Usage:
    python src/evaluate.py                 # Evaluates on temporal validation split (Apr-Jun 2026)
    python src/evaluate.py --split val     # Validation split
    python src/evaluate.py --split train   # In-sample training split sanity check
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FraudFeatureEngineer, FEATURE_COLS
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_END = pd.Timestamp("2026-04-01")
VAL_END   = pd.Timestamp("2026-07-01")
FP_COST_INR = 640
MONTHLY_REVIEW_BUDGET = 40


def load_artifacts():
    """Load persisted models and feature engineering pipeline."""
    engineer = FraudFeatureEngineer.load(MODELS_DIR / "engineer.pkl")
    with open(MODELS_DIR / "baseline.pkl", "rb") as f:
        baseline = pickle.load(f)
    with open(MODELS_DIR / "lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(MODELS_DIR / "selected_model.pkl", "rb") as f:
        selected = pickle.load(f)
    with open(MODELS_DIR / "training_meta.json", "r") as f:
        meta = json.load(f)
    return engineer, baseline, lgbm, selected, meta


def print_evaluation(name: str, y_true, y_scores, amounts, n_months: int):
    """Print thorough operational metrics for a model."""
    roc = float(roc_auc_score(y_true, y_scores))
    pr = float(average_precision_score(y_true, y_scores))

    y_pred_05 = (y_scores >= 0.5).astype(int)
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05  = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05   = float(f1_score(y_true, y_pred_05, zero_division=0))
    cm_05   = confusion_matrix(y_true, y_pred_05)
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel() if cm_05.shape == (2, 2) else (0, 0, 0, 0)

    # Top 40 (1 month)
    idx_40 = np.argsort(y_scores)[::-1][:40]
    y_40 = np.array(y_true)[idx_40]
    amt_40 = np.array(amounts)[idx_40]
    tp_40 = int(y_40.sum())
    fp_40 = int((y_40 == 0).sum())
    saved_40 = float(amt_40[y_40 == 1].sum())
    wasted_40 = float(fp_40 * FP_COST_INR)
    net_40 = saved_40 - wasted_40

    # Top total (40 * n_months)
    k_total = MONTHLY_REVIEW_BUDGET * n_months
    idx_tot = np.argsort(y_scores)[::-1][:k_total]
    y_tot = np.array(y_true)[idx_tot]
    amt_tot = np.array(amounts)[idx_tot]
    tp_tot = int(y_tot.sum())
    fp_tot = int((y_tot == 0).sum())
    saved_tot = float(amt_tot[y_tot == 1].sum())
    wasted_tot = float(fp_tot * FP_COST_INR)
    net_tot = saved_tot - wasted_tot

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  ROC-AUC                : {roc:.4f}")
    print(f"  PR-AUC                 : {pr:.4f}")
    print(f"  Threshold 0.50:")
    print(f"    Precision            : {prec_05:.4f}")
    print(f"    Recall               : {rec_05:.4f}")
    print(f"    F1-score             : {f1_05:.4f}")
    print(f"    Confusion Matrix     : TN={tn_05} FP={fp_05} | FN={fn_05} TP={tp_05}")
    print(f"  Monthly Desk Budget (Top 40):")
    print(f"    Precision@40         : {tp_40/40:.4f} ({tp_40} TP, {fp_40} FP)")
    print(f"    Recall@40            : {tp_40/y_true.sum():.4f}")
    print(f"    INR Saved            : Rs.{saved_40:,.0f}")
    print(f"    INR Wasted (FP Cost) : Rs.{wasted_40:,.0f}")
    print(f"    Net Benefit          : Rs.{net_40:,.0f}")
    print(f"  Validation Total Budget (Top {k_total} over {n_months} mos):")
    print(f"    Precision@{k_total:<3d}        : {tp_tot/k_total:.4f} ({tp_tot} TP, {fp_tot} FP)")
    print(f"    Recall@{k_total:<3d}           : {tp_tot/y_true.sum():.4f}")
    print(f"    INR Saved            : Rs.{saved_tot:,.0f}")
    print(f"    INR Wasted (FP Cost) : Rs.{wasted_tot:,.0f}")
    print(f"    Net Benefit          : Rs.{net_tot:,.0f}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kestrel Fraud Detection models")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()

    print(f"\nLoading model artifacts from {MODELS_DIR}...")
    engineer, baseline, lgbm, selected, meta = load_artifacts()
    feat_cols = engineer.get_feature_cols()

    print("Loading data...")
    train_raw = pd.read_csv(ROOT / "train.csv", dtype=str)
    train_raw["submitted_at_dt"] = pd.to_datetime(train_raw["submitted_at"], errors="coerce")

    # Deduplicate keeping initial submission
    train_sorted = train_raw.sort_values("submitted_at_dt").copy()
    df_dedup = train_sorted.drop_duplicates(subset="claim_id", keep="first").copy()

    # Filter out blank labels
    blank_mask = df_dedup["is_fraud"].isnull() | (df_dedup["is_fraud"].str.strip() == "")
    df_labeled = df_dedup[~blank_mask].copy()
    df_labeled["is_fraud"] = df_labeled["is_fraud"].astype(int)

    if args.split == "val":
        df_eval = df_labeled[
            (df_labeled["submitted_at_dt"] >= TRAIN_END) &
            (df_labeled["submitted_at_dt"] < VAL_END)
        ].copy()
        split_name = "Validation (April 2026 - June 2026)"
        n_months = 3
    else:
        df_eval = df_labeled[df_labeled["submitted_at_dt"] < TRAIN_END].copy()
        split_name = "Training In-Sample (April 2025 - March 2026)"
        n_months = 12

    print(f"Evaluating Split: {split_name}")
    print(f"Total claims: {len(df_eval):,} | Fraud count: {df_eval['is_fraud'].sum()} ({df_eval['is_fraud'].mean():.2%})")

    # Feature transformation
    df_feat = engineer.transform(df_eval)
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat["is_fraud"].values.astype(int)
    amounts = pd.to_numeric(df_feat["claim_amount_inr"], errors="coerce").fillna(0).values

    # Evaluate Baseline
    y_scores_base = baseline.predict_proba(X)[:, 1]
    print_evaluation("Logistic Regression (Baseline)", y, y_scores_base, amounts, n_months)

    # Evaluate LightGBM
    y_scores_lgbm = lgbm.predict_proba(X)[:, 1]
    print_evaluation("LightGBM (Primary)", y, y_scores_lgbm, amounts, n_months)

    print(f"\nModel Selected in Training: {meta.get('selected_model')}")


if __name__ == "__main__":
    main()
