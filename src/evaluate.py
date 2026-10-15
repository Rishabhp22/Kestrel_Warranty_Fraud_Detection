"""
src/evaluate.py -- Kestrel Warranty Fraud Detection
Operational evaluation script for fraud risk models against desk review capacity.

Key Design Requirements (Task 2 Reviewer Feedback):
1. Measure performance against the actual operational capacity:
   The investigation desk reviews at most 40 claims per month (not a pooled 40-claim
   queue across the entire quarter).
2. Operational evaluation with separate top-40 review queues for each validation month
   (April 2026, May 2026, June 2026 = 120 total reviews across 3 months).
3. For each month and overall:
   - Eligible claims, confirmed fraud count, and base fraud rate.
   - Precision@40, Recall@40, True Positives (TP), False Positives (FP), False Negatives (FN).
   - PR-AUC and ROC-AUC where mathematically valid.
   - Net financial benefit under Ops Policy v4.1 (FP Cost = Rs. 640 [Rs. 380 goodwill + Rs. 260 contact]).
   - Breakdown by claim size: small claims (< Rs. 2,000) vs large claims (>= Rs. 2,000).
   - Pre-policy (April 2026) vs post-policy (May & June 2026).
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (
    FraudFeatureEngineer,
    FEATURE_COLS_V1,
    FEATURE_COLS_V2,
    POLICY_CHANGE_DATE,
    AUTO_APPROVE_THRESHOLD_INR,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODELS_DIR = ROOT / "models"
V2_MODELS_DIR = MODELS_DIR / "v2_candidate"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_END = pd.Timestamp("2026-04-01")
VAL_END = pd.Timestamp("2026-07-01")

# Operational Cost Assumptions (ops-policy.pdf §4)
FP_GOODWILL_COST_INR = 380.0     # Average customer goodwill compensation for delay
FP_SERVICE_CONTACT_INR = 260.0   # Blended service contact cost
FP_TOTAL_COST_INR = FP_GOODWILL_COST_INR + FP_SERVICE_CONTACT_INR  # Rs. 640
MONTHLY_REVIEW_BUDGET = 40       # Max claims investigation desk can review per month


def evaluate_monthly_operational_queue(
    y_scores: np.ndarray,
    df_eval: pd.DataFrame,
    model_name: str = "Model",
) -> dict:
    """
    Simulate operational review desk workflow with a separate top-40 queue per month.
    """
    work = df_eval.copy()
    work["fraud_score"] = y_scores
    work["submitted_at_dt"] = pd.to_datetime(work["submitted_at"], errors="coerce")
    work["month"] = work["submitted_at_dt"].dt.to_period("M").astype(str)
    work["amt"] = pd.to_numeric(work["claim_amount_inr"], errors="coerce").fillna(0.0)
    work["is_small"] = work["amt"] < AUTO_APPROVE_THRESHOLD_INR

    months = sorted(work["month"].unique())

    monthly_metrics = []
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tot_saved = 0.0
    tot_wasted = 0.0
    tot_small_caught = 0
    tot_large_caught = 0

    print(f"\n{'='*75}")
    print(f"  OPERATIONAL MONTHLY REVIEW EVALUATION: {model_name}")
    print(f"  Monthly Desk Budget: 40 Claims/Month | FP Cost: Rs. {FP_TOTAL_COST_INR:,.0f} (Goodwill Rs. {FP_GOODWILL_COST_INR:,.0f} + Contact Rs. {FP_SERVICE_CONTACT_INR:,.0f})")
    print(f"{'='*75}")

    for m in months:
        m_df = work[work["month"] == m].copy()
        n_eligible = len(m_df)
        n_fraud_total = int(m_df["is_fraud"].sum())
        base_rate = (n_fraud_total / n_eligible) if n_eligible > 0 else 0.0

        # Rank claims within this month strictly by model score descending
        m_sorted = m_df.sort_values(by="fraud_score", ascending=False)
        m_queue = m_sorted.head(MONTHLY_REVIEW_BUDGET)
        queue_size = len(m_queue)

        tp = int(m_queue["is_fraud"].sum())
        fp = int(queue_size - tp)
        fn = int(n_fraud_total - tp)

        # Financial impact
        tp_amounts = m_queue[m_queue["is_fraud"] == 1]["amt"]
        inr_saved = float(tp_amounts.sum())
        inr_wasted = float(fp * FP_TOTAL_COST_INR)
        net_benefit = inr_saved - inr_wasted

        # Claim size composition of true positives
        tp_small = int(((m_queue["is_fraud"] == 1) & (m_queue["is_small"])).sum())
        tp_large = int(tp - tp_small)

        prec_40 = (tp / queue_size) if queue_size > 0 else 0.0
        rec_40 = (tp / n_fraud_total) if n_fraud_total > 0 else 0.0

        # Discrimative metrics for the month
        try:
            roc = float(roc_auc_score(m_df["is_fraud"], m_df["fraud_score"]))
        except ValueError:
            roc = float("nan")

        try:
            pr_auc = float(average_precision_score(m_df["is_fraud"], m_df["fraud_score"]))
        except ValueError:
            pr_auc = float("nan")

        policy_status = "Pre-Policy (Inspection Required)" if m < "2026-05" else "Post-Policy (Auto-Approval < Rs.2k)"

        print(f"\n  [Month {m}] -- {policy_status}:")
        print(f"    Volume & Prevalence : {n_eligible:,} claims | {n_fraud_total} confirmed fraud ({base_rate:.2%})")
        print(f"    Discrimination      : ROC-AUC = {roc:.4f} | PR-AUC = {pr_auc:.4f}")
        print(f"    Top 40 Desk Queue   : TP={tp} (Small={tp_small}, Large={tp_large}) | FP={fp} | FN={fn}")
        print(f"    Operational Catch   : Precision@40 = {prec_40:.2%} ({tp}/{queue_size}) | Recall@40 = {rec_40:.2%} ({tp}/{n_fraud_total})")
        print(f"    Financial Outcome   : Saved = Rs. {inr_saved:,.0f} | Wasted = Rs. {inr_wasted:,.0f} | Net Benefit = Rs. {net_benefit:,.0f}")

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_saved += inr_saved
        tot_wasted += inr_wasted
        tot_small_caught += tp_small
        tot_large_caught += tp_large

        monthly_metrics.append({
            "month": m,
            "policy_status": policy_status,
            "eligible_claims": n_eligible,
            "confirmed_fraud": n_fraud_total,
            "base_fraud_rate": base_rate,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tp_small": tp_small,
            "tp_large": tp_large,
            "precision_at_40": prec_40,
            "recall_at_40": rec_40,
            "inr_saved": inr_saved,
            "inr_wasted": inr_wasted,
            "net_benefit": net_benefit,
        })

    total_reviews = tot_tp + tot_fp
    overall_fraud = int(df_eval["is_fraud"].sum())
    overall_prec = (tot_tp / total_reviews) if total_reviews > 0 else 0.0
    overall_rec = (tot_tp / overall_fraud) if overall_fraud > 0 else 0.0
    overall_net = tot_saved - tot_wasted

    overall_roc = float(roc_auc_score(df_eval["is_fraud"], y_scores))
    overall_pr_auc = float(average_precision_score(df_eval["is_fraud"], y_scores))

    print(f"\n  {'-'*71}")
    print(f"  QUARTERLY OPERATIONAL TOTAL ({total_reviews} Reviews over {len(months)} Months):")
    print(f"    Total Fraud Caught  : {tot_tp}/{total_reviews} reviews (Precision = {overall_prec:.2%})")
    print(f"    Total Catch Share   : {tot_tp}/{overall_fraud} total fraud (Recall = {overall_rec:.2%})")
    print(f"    Caught Composition  : Small Claims (< Rs.2k) = {tot_small_caught} | Large Claims = {tot_large_caught}")
    print(f"    False Alarms        : {tot_fp} legitimate claims held for investigation")
    print(f"    Total Loss Avoided  : Rs. {tot_saved:,.0f}")
    print(f"    Total Waste (FP Cost): Rs. {tot_wasted:,.0f}")
    print(f"    Net Operating Benefit: Rs. {overall_net:,.0f}")
    print(f"    Global ROC-AUC      : {overall_roc:.4f}")
    print(f"    Global PR-AUC       : {overall_pr_auc:.4f}")
    print(f"{'='*75}\n")

    return {
        "model_name": model_name,
        "monthly": monthly_metrics,
        "total_reviews": total_reviews,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
        "total_small_caught": tot_small_caught,
        "total_large_caught": tot_large_caught,
        "precision_at_capacity": overall_prec,
        "recall_at_capacity": overall_rec,
        "total_inr_saved": tot_saved,
        "total_inr_wasted": tot_wasted,
        "net_operating_benefit": overall_net,
        "overall_roc_auc": overall_roc,
        "overall_pr_auc": overall_pr_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Operational evaluation for Kestrel Fraud Detection models")
    parser.add_argument("--split", choices=["val", "train"], default="val", help="Data split to evaluate")
    parser.add_argument(
        "--model",
        choices=["candidate", "baseline", "all"],
        default="all",
        help="Model artifact to evaluate (candidate V2, baseline V1, or all)",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  Kestrel Warranty Fraud Detection -- Operational Evaluation Suite")
    print(f"{'='*70}")

    train_raw = pd.read_csv(ROOT / "train.csv", dtype=str)
    partners = pd.read_csv(ROOT / "partners.csv")
    products = pd.read_csv(ROOT / "products.csv")

    train_raw["submitted_at_dt"] = pd.to_datetime(train_raw["submitted_at"], errors="coerce")
    train_sorted = train_raw.sort_values("submitted_at_dt").copy()
    df_dedup = train_sorted.drop_duplicates(subset="claim_id", keep="first").copy()

    # Exclude unresolved blank claims from supervised evaluation
    blank_mask = df_dedup["is_fraud"].isnull() | (df_dedup["is_fraud"].str.strip() == "")
    df_labeled = df_dedup[~blank_mask].copy()
    df_labeled["is_fraud"] = df_labeled["is_fraud"].astype(int)

    if args.split == "val":
        df_eval = df_labeled[
            (df_labeled["submitted_at_dt"] >= TRAIN_END) &
            (df_labeled["submitted_at_dt"] < VAL_END)
        ].copy()
        split_name = "Temporal Validation (April 2026 - June 2026)"
    else:
        df_eval = df_labeled[df_labeled["submitted_at_dt"] < TRAIN_END].copy()
        split_name = "Training In-Sample (April 2025 - March 2026)"

    print(f"Split under evaluation : {split_name}")
    print(f"Total claims in split   : {len(df_eval):,} | Confirmed fraud: {df_eval['is_fraud'].sum()} ({df_eval['is_fraud'].mean():.2%})\n")

    # 1. Baseline Model Evaluation (V1 Production Model)
    if args.model in ["baseline", "all"]:
        prod_engineer_path = MODELS_DIR / "engineer.pkl"
        prod_model_path = MODELS_DIR / "selected_model.pkl"

        if prod_engineer_path.exists() and prod_model_path.exists():
            engineer_prod = FraudFeatureEngineer.load(prod_engineer_path)
            with open(prod_model_path, "rb") as f:
                model_prod = pickle.load(f)

            feats_prod = engineer_prod.transform(df_eval, partners, products)
            cols_prod = engineer_prod.get_feature_cols()
            scores_prod = model_prod.predict_proba(feats_prod[cols_prod].values)[:, 1]

            evaluate_monthly_operational_queue(
                scores_prod,
                df_eval,
                model_name=f"Baseline Model ({len(cols_prod)} Features -- Production)",
            )
        else:
            print(f"[!] Warning: Baseline artifacts not found in {MODELS_DIR}")

    # 2. Candidate Model Evaluation (V2 Candidate)
    if args.model in ["candidate", "all"]:
        v2_engineer_path = V2_MODELS_DIR / "engineer_v2.pkl"
        v2_model_path = V2_MODELS_DIR / "lgbm_v2.pkl"

        if v2_engineer_path.exists() and v2_model_path.exists():
            engineer_v2 = FraudFeatureEngineer.load(v2_engineer_path)
            with open(v2_model_path, "rb") as f:
                model_v2 = pickle.load(f)

            feats_v2 = engineer_v2.transform(df_eval, partners, products)
            cols_v2 = engineer_v2.get_feature_cols()
            scores_v2 = model_v2.predict_proba(feats_v2[cols_v2].values)[:, 1]

            evaluate_monthly_operational_queue(
                scores_v2,
                df_eval,
                model_name=f"Candidate Model V2 ({len(cols_v2)} Features -- Leakage-Safe)",
            )
        else:
            print(f"[*] Note: Candidate V2 artifacts not yet saved in {V2_MODELS_DIR}. Run src/train.py first.")


if __name__ == "__main__":
    main()
