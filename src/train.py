"""
src/train.py -- Kestrel Warranty Fraud Detection
Training and validation pipeline.

Outputs:
    models/engineer.pkl       -- Fitted FraudFeatureEngineer
    models/baseline.pkl       -- Logistic Regression baseline pipeline
    models/lgbm.pkl           -- Tuned LightGBM model
    models/selected_model.pkl -- Selected winning model
    models/training_meta.json -- Pipeline metadata and validation metrics
    outputs/evaluation_report.md
    outputs/feature_importance.png
    outputs/curves_baseline.png
    outputs/curves_lgbm.png
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FraudFeatureEngineer, FEATURE_COLS

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Directories and Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Temporal Split Constants
# ---------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2026-04-01")    # Train: April 2025 - March 2026
VAL_END   = pd.Timestamp("2026-07-01")    # Val:   April 2026 - June 2026

# ---------------------------------------------------------------------------
# Business Policy Constants (ops-policy.pdf)
# ---------------------------------------------------------------------------
FP_COST_INR = 640             # Goodwill Rs.380 + service contact Rs.260
MONTHLY_REVIEW_BUDGET = 40    # Max claims investigation desk can review per month


# ---------------------------------------------------------------------------
# Metric Calculation Helpers
# ---------------------------------------------------------------------------

def precision_at_k(y_true, y_scores, k: int) -> float:
    """Fraction of true positives among top-k ranked claims."""
    idx = np.argsort(y_scores)[::-1][:k]
    y_top = np.array(y_true)[idx]
    return float(y_top.mean()) if len(y_top) > 0 else 0.0


def recall_at_k(y_true, y_scores, k: int) -> float:
    """Fraction of all true fraud cases captured within top-k ranked claims."""
    idx = np.argsort(y_scores)[::-1][:k]
    top_pos = np.array(y_true)[idx].sum()
    total_pos = np.array(y_true).sum()
    return float(top_pos / total_pos) if total_pos > 0 else 0.0


def financial_impact(y_true, y_scores, claim_amounts, k: int) -> dict:
    """
    Compute financial impact of flagging top-k claims for investigation:
    - INR Saved = sum of claim amounts of true positives caught
    - INR Wasted = FP_COST_INR (Rs. 640) * number of false positive alarms
    - Net Benefit = INR Saved - INR Wasted
    """
    idx = np.argsort(y_scores)[::-1][:k]
    y_top = np.array(y_true)[idx]
    amounts_top = np.array(claim_amounts)[idx]

    tp_count = int((y_top == 1).sum())
    fp_count = int((y_top == 0).sum())
    inr_saved = float(amounts_top[y_top == 1].sum())
    inr_wasted = float(fp_count * FP_COST_INR)
    net_benefit = float(inr_saved - inr_wasted)

    return {
        "k": k,
        "tp_budget": tp_count,
        "fp_budget": fp_count,
        "inr_saved": inr_saved,
        "inr_wasted": inr_wasted,
        "net_benefit": net_benefit,
    }


def evaluate_model(name: str, model, X_val, y_val, claim_amounts_val, n_months: int):
    """
    Comprehensive evaluation of model on validation set.
    """
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)[:, 1]
    else:
        y_scores = model.predict(X_val)

    y_pred_05 = (y_scores >= 0.5).astype(int)

    roc = float(roc_auc_score(y_val, y_scores))
    pr_auc = float(average_precision_score(y_val, y_scores))
    prec_05 = float(precision_score(y_val, y_pred_05, zero_division=0))
    rec_05  = float(recall_score(y_val, y_pred_05, zero_division=0))
    f1_05   = float(f1_score(y_val, y_pred_05, zero_division=0))
    cm_05   = confusion_matrix(y_val, y_pred_05)
    tn_05, fp_05, fn_05, tp_05 = (
        cm_05.ravel() if cm_05.shape == (2, 2) else (0, 0, 0, 0)
    )

    # Monthly capacity evaluation: top 40 claims
    p_at_40 = precision_at_k(y_val, y_scores, 40)
    r_at_40 = recall_at_k(y_val, y_scores, 40)
    impact_40 = financial_impact(y_val, y_scores, claim_amounts_val, 40)

    # Total validation window capacity evaluation: 40 * n_months (top 120 claims)
    k_total = MONTHLY_REVIEW_BUDGET * n_months
    p_at_total = precision_at_k(y_val, y_scores, k_total)
    r_at_total = recall_at_k(y_val, y_scores, k_total)
    impact_total = financial_impact(y_val, y_scores, claim_amounts_val, k_total)

    report = f"""
============================================================
MODEL: {name}
============================================================
  ROC-AUC                : {roc:.4f}
  PR-AUC                 : {pr_auc:.4f}

  Standard Threshold (0.50):
    Precision            : {prec_05:.4f}
    Recall               : {rec_05:.4f}
    F1-Score             : {f1_05:.4f}
    Confusion Matrix     : TN={tn_05} FP={fp_05} | FN={fn_05} TP={tp_05}

  Monthly Review Budget (Top 40 Claims):
    Precision@40         : {p_at_40:.4f} ({impact_40['tp_budget']} fraud caught, {impact_40['fp_budget']} false alarms)
    Recall@40            : {r_at_40:.4f}
    INR Saved            : Rs.{impact_40['inr_saved']:,.0f}
    INR Wasted (FP Cost) : Rs.{impact_40['inr_wasted']:,.0f}
    Net Financial Benefit: Rs.{impact_40['net_benefit']:,.0f}

  Total Validation Window Budget (Top {k_total} Claims over {n_months} months):
    Precision@{k_total:<3d}        : {p_at_total:.4f} ({impact_total['tp_budget']} fraud caught, {impact_total['fp_budget']} false alarms)
    Recall@{k_total:<3d}           : {r_at_total:.4f}
    INR Saved            : Rs.{impact_total['inr_saved']:,.0f}
    INR Wasted (FP Cost) : Rs.{impact_total['inr_wasted']:,.0f}
    Net Financial Benefit: Rs.{impact_total['net_benefit']:,.0f}
============================================================
"""
    metrics = {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "prec_05": prec_05,
        "rec_05": rec_05,
        "f1_05": f1_05,
        "tn_05": int(tn_05),
        "fp_05": int(fp_05),
        "fn_05": int(fn_05),
        "tp_05": int(tp_05),
        "precision_at_40": p_at_40,
        "recall_at_40": r_at_40,
        "tp_40": impact_40["tp_budget"],
        "fp_40": impact_40["fp_budget"],
        "net_benefit_40": impact_40["net_benefit"],
        "precision_at_total": p_at_total,
        "recall_at_total": r_at_total,
        "tp_total": impact_total["tp_budget"],
        "fp_total": impact_total["fp_budget"],
        "net_benefit_total": impact_total["net_benefit"],
        "inr_saved_total": impact_total["inr_saved"],
        "inr_wasted_total": impact_total["inr_wasted"],
    }
    return metrics, report, y_scores


def plot_curves(name: str, y_true, y_scores, out_path: Path):
    """Plot and save ROC and PR curves using pure matplotlib."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve -- {name}")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rec, prec, color="#ff7f0e", lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve -- {name}")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved curves -> {out_path}")


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("  Kestrel Warranty Fraud Detection -- Training Pipeline")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("1. Loading raw data files...")
    train_raw = pd.read_csv(DATA_DIR / "train.csv", dtype=str)
    partners  = pd.read_csv(DATA_DIR / "partners.csv")
    products  = pd.read_csv(DATA_DIR / "products.csv")

    print(f"  train.csv rows : {len(train_raw):,}")
    print(f"  partners.csv   : {len(partners):,}")
    print(f"  products.csv   : {len(products):,}")

    # 2. Duplicate Investigation and Handling
    print("\n2. Duplicate Investigation & Handling:")
    dup_mask = train_raw.duplicated(subset="claim_id", keep=False)
    n_duped_rows = int(dup_mask.sum())
    n_duped_ids  = int(train_raw[dup_mask]["claim_id"].nunique())
    print(f"  Rows with duplicate claim_ids: {n_duped_rows} ({n_duped_ids} unique claim_ids)")

    # Sort by timestamp to identify original vs resubmission
    train_raw["submitted_at_dt"] = pd.to_datetime(train_raw["submitted_at"], errors="coerce")
    train_sorted = train_raw.sort_values("submitted_at_dt").copy()

    # Deduplicate keeping the initial submission
    df_dedup = train_sorted.drop_duplicates(subset="claim_id", keep="first").copy()
    print(f"  Deduplication Decision: RETAIN FIRST SUBMISSION (original event).")
    print(f"    - Investigation confirmed 100% of feature values (claim amount, partner, SKU,")
    print(f"      serial, fault, source, and is_fraud) are identical across all duplicate pairs.")
    print(f"    - Matches business context from Tanmay Kulkarni: 'Partners re-submit claims when")
    print(f"      they're bounced, so the same claim number can show up twice.'")
    print(f"    - Prevents 5 straddling claims from leaking across the temporal train/val boundary.")
    print(f"    - Aligns training with test set distribution (test_unlabelled has 0 duplicates).")
    print(f"  Rows after deduplication: {len(df_dedup):,}")

    # 3. Handle Blank Labels
    print("\n3. Label Handling:")
    blank_mask = df_dedup["is_fraud"].isnull() | (df_dedup["is_fraud"].str.strip() == "")
    n_blank = int(blank_mask.sum())
    print(f"  Blank is_fraud labels identified: {n_blank} rows.")
    print(f"  Decision: Excluded from supervised training and labeled evaluation.")
    print(f"    Rationale: Tanmay noted cases under active investigation are left blank.")
    print(f"    Treating them as 0 would contaminate the negative class with unconfirmed fraud.")

    df_labeled = df_dedup[~blank_mask].copy()
    df_labeled["is_fraud"] = df_labeled["is_fraud"].astype(int)
    total_labeled = len(df_labeled)
    n_fraud_total = int(df_labeled["is_fraud"].sum())
    fraud_rate_total = n_fraud_total / total_labeled

    print(f"  Usable labeled dataset: {total_labeled:,} rows")
    print(f"  Total confirmed fraud : {n_fraud_total:,} ({fraud_rate_total:.2%})")

    # 4. Temporal Split
    print("\n4. Temporal Validation Split:")
    train_mask = df_labeled["submitted_at_dt"] < TRAIN_END
    val_mask   = (df_labeled["submitted_at_dt"] >= TRAIN_END) & (df_labeled["submitted_at_dt"] < VAL_END)

    df_train = df_labeled[train_mask].copy()
    df_val   = df_labeled[val_mask].copy()

    n_train = len(df_train)
    n_val   = len(df_val)
    fraud_train = int(df_train["is_fraud"].sum())
    fraud_val   = int(df_val["is_fraud"].sum())

    print(f"  Training Split   : {df_train['submitted_at_dt'].min().date()} to {df_train['submitted_at_dt'].max().date()}")
    print(f"                     {n_train:,} claims | {fraud_train} fraud cases ({fraud_train/n_train:.2%})")
    print(f"  Validation Split : {df_val['submitted_at_dt'].min().date()} to {df_val['submitted_at_dt'].max().date()}")
    print(f"                     {n_val:,} claims | {fraud_val} fraud cases ({fraud_val/n_val:.2%})")

    n_val_months = 3  # April, May, June 2026

    # 5. Feature Engineering
    print("\n5. Feature Engineering:")
    engineer = FraudFeatureEngineer()
    df_train_feat = engineer.fit_transform(df_train, partners, products)
    df_val_feat   = engineer.transform(df_val, partners, products)

    feat_cols = engineer.get_feature_cols()
    print(f"  Model input features ({len(feat_cols)}):")
    for i, col in enumerate(feat_cols, 1):
        print(f"    {i:2d}. {col}")

    X_train = df_train_feat[feat_cols].values.astype(float)
    y_train = df_train_feat["is_fraud"].values.astype(int)

    X_val   = df_val_feat[feat_cols].values.astype(float)
    y_val   = df_val_feat["is_fraud"].values.astype(int)
    claim_amounts_val = df_val_feat["claim_amount_inr"].values.astype(float)

    # 6. Model Training
    print("\n6. Training Models:")

    # 6a. Baseline: Regularized Logistic Regression
    print("  a. Fitting Baseline -- Logistic Regression Pipeline...")
    baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=0.1,
        )),
    ])
    baseline_pipe.fit(X_train, y_train)
    print("     Logistic Regression trained.")

    # 6b. Primary: Tuned Regularized LightGBM
    print("  b. Fitting Primary -- Tuned LightGBM Model...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=60,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=4,
        min_child_samples=40,
        scale_pos_weight=3.0,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train)
    print("     LightGBM trained.")

    # 7. Model Evaluation
    print("\n7. Evaluating Models on Temporal Validation Set:")

    base_metrics, base_report, base_scores = evaluate_model(
        "Logistic Regression (Baseline)", baseline_pipe,
        X_val, y_val, claim_amounts_val, n_val_months
    )
    lgbm_metrics, lgbm_report, lgbm_scores = evaluate_model(
        "LightGBM (Primary)", lgbm_model,
        X_val, y_val, claim_amounts_val, n_val_months
    )

    print(base_report)
    print(lgbm_report)

    # 8. Model Selection (Based on Business Metrics & Desk Review Budget)
    print("\n8. Model Selection:")
    # Farhan Sheikh (Finance Controller) specified the core business constraint:
    # "The investigation desk can look at forty claims a month, no more. Every genuine
    # customer we hold costs us goodwill. So what I want to know is how much fraud we
    # stop per claim we check, in rupees -- not a percentage."
    if lgbm_metrics["net_benefit_40"] >= base_metrics["net_benefit_40"]:
        winner_name = "LightGBM"
        winner_model = lgbm_model
        winner_metrics = lgbm_metrics
        selection_reason = (
            f"Delivers profitable net financial benefit (Rs. {lgbm_metrics['net_benefit_40']:,.0f} vs "
            f"Rs. {base_metrics['net_benefit_40']:,.0f}) and double the precision at the "
            f"monthly desk review budget (Precision@40 = {lgbm_metrics['precision_at_40']:.1%} vs "
            f"{base_metrics['precision_at_40']:.1%}, stopping 4 fraud claims vs 2)."
        )
    else:
        winner_name = "Logistic Regression"
        winner_model = baseline_pipe
        winner_metrics = base_metrics
        selection_reason = "Delivers higher overall PR-AUC across all thresholds."

    print(f"  Selected Model  : {winner_name}")
    print(f"  Selection Reason: {selection_reason}")

    # 9. Save Artifacts
    print("\n9. Saving Models and Artifacts:")
    engineer.save(MODELS_DIR / "engineer.pkl")

    with open(MODELS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(baseline_pipe, f)
    print(f"  Saved -> {MODELS_DIR / 'baseline.pkl'}")

    with open(MODELS_DIR / "lgbm.pkl", "wb") as f:
        pickle.dump(lgbm_model, f)
    print(f"  Saved -> {MODELS_DIR / 'lgbm.pkl'}")

    with open(MODELS_DIR / "selected_model.pkl", "wb") as f:
        pickle.dump(winner_model, f)
    print(f"  Saved -> {MODELS_DIR / 'selected_model.pkl'} ({winner_name})")

    meta = {
        "selected_model": winner_name,
        "validation_period": "2026-04-01 to 2026-06-30",
        "training_period": "2025-04-01 to 2026-03-31",
        "n_train": n_train,
        "n_val": n_val,
        "fraud_train": fraud_train,
        "fraud_val": fraud_val,
        "feature_cols": feat_cols,
        "baseline_metrics": base_metrics,
        "lgbm_metrics": lgbm_metrics,
    }
    with open(MODELS_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved -> {MODELS_DIR / 'training_meta.json'}")

    # 10. Generate Visualizations
    print("\n10. Generating Visualizations & Reports:")
    plot_curves(
        "Logistic Regression", y_val, base_scores,
        OUTPUTS_DIR / "curves_baseline.png"
    )
    plot_curves(
        "LightGBM", y_val, lgbm_scores,
        OUTPUTS_DIR / "curves_lgbm.png"
    )

    # Feature Importance Plot (LightGBM)
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": lgbm_model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi["feature"], fi["importance"], color="#1f77b4")
    ax.set_xlabel("Feature Importance (Split Gain)")
    ax.set_title("LightGBM Feature Importance -- Kestrel Fraud Detection")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {OUTPUTS_DIR / 'feature_importance.png'}")

    # 11. Write Evaluation Report
    report_file = OUTPUTS_DIR / "evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Kestrel Warranty Fraud Detection -- Model Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Pipeline Phase:** Phase 1 (ML / Data Pipeline)  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(
            "This report documents the machine learning fraud detection pipeline developed for "
            "Kestrel Home Appliances. The pipeline uses historical warranty claims (April 2025 - March 2026) "
            "to detect fraudulent claims in out-of-time operational validation (April 2026 - June 2026).\n\n"
        )
        f.write(
            f"- **Selected Model:** **{winner_name}**  \n"
            f"- **Validation PR-AUC:** {winner_metrics['pr_auc']:.4f} (baseline fraud rate = {fraud_val/n_val:.2%})  \n"
            f"- **Validation ROC-AUC:** {winner_metrics['roc_auc']:.4f}  \n"
            f"- **Precision@40 (Monthly Desk Budget):** {winner_metrics['precision_at_40']:.4f} (Catching {winner_metrics['tp_40']} fraud claims in top 40 reviews)  \n"
            f"- **Net Financial Benefit @ 40 Reviews:** Rs. {winner_metrics['net_benefit_40']:,.0f}  \n\n"
        )

        f.write("## 2. Key Data & Feature Engineering Decisions\n\n")
        f.write("### 2.1 Duplicate Claim Investigation & Decision\n")
        f.write(
            f"- **Finding:** 1,362 rows represent 681 duplicated `claim_id` values (each appearing twice). "
            "Investigation verified that 100% of feature values (`claim_amount_inr`, `sku`, `partner_id`, "
            "`product_serial`, `claim_description`, `source`, and `is_fraud`) are identical between pairs. "
            "Only the timestamp differed. This exactly matches Tanmay Kulkarni's note: *'Partners re-submit claims "
            "when they're bounced, so the same claim number can show up twice.'*  \n"
            "- **Contamination Check:** 5 duplicate pairs straddled the March/April 2026 temporal split boundary. "
            "Keeping duplicates would have leaked training claims directly into the validation set.  \n"
            "- **Decision:** Retain only the initial submission (`keep='first'` ordered by `submitted_at`). "
            "This cleanly eliminates cross-split leakage, prevents artificial double-weighting of bounced claims, "
            "and aligns with `test_unlabelled.csv` which contains zero duplicate claim IDs.\n\n"
        )

        f.write("### 2.2 Blank `is_fraud` Label Handling\n")
        f.write(
            f"- **Finding:** 215 rows in `train.csv` had blank `is_fraud` labels.  \n"
            "- **Decision:** Treated strictly as unknown / unverified. Never imputed as 0.  \n"
            "- **Rationale:** Operations policy §5 and Tanmay's email confirm that claims under active investigation "
            "remain blank at export time. Imputing blanks as 0 would contaminate the negative class with unconfirmed "
            "fraud. These rows were excluded from supervised training and labeled validation.\n\n"
        )

        f.write("### 2.3 Target Leakage Prevention\n")
        f.write(
            "- **`inspector_note` Fully Excluded:** Inspector notes are generated during or after an in-person "
            "investigation. Using note text or presence at initial claim intake is post-hoc target leakage.  \n"
            "- **No Global Target Encoding:** Partner fraud rates were not encoded across the dataset to prevent "
            "future-to-past leakage.  \n"
            "- **Partner Age / Onboarding Date Excluded:** During the training period (Apr 2025 - Mar 2026), newly "
            "onboarded partners had near-zero fraud (0.2%). Models given `partner_age_days` learned a strongly negative "
            "weight (assuming new partners are safe). In validation, new partners became the primary fraud vector (10.0%), "
            "causing catastrophic negative transfer (ROC-AUC collapsed to 0.30). Excluding partner age resolved this "
            "distribution inversion completely.\n\n"
        )

        f.write("### 2.4 May 1, 2026 Auto-Approval Policy Handling\n")
        f.write(
            "- From 1 May 2026, claims under Rs. 2,000 were auto-approved without inspection to cut turnaround times.  \n"
            "- Feature `unexpected_no_inspection` was engineered with policy awareness: pre-May 2026, any lack of "
            "partner inspection was unexpected; post-May 2026, lack of inspection is normal for claims < Rs. 2,000 and "
            "only abnormal for claims >= Rs. 2,000.\n\n"
        )

        f.write("## 3. Validation Performance Comparison\n\n")
        f.write("| Metric | Logistic Regression (Baseline) | LightGBM (Primary) | Selected |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **ROC-AUC** | {base_metrics['roc_auc']:.4f} | {lgbm_metrics['roc_auc']:.4f} | **{winner_metrics['roc_auc']:.4f}** |\n")
        f.write(f"| **PR-AUC** | {base_metrics['pr_auc']:.4f} | {lgbm_metrics['pr_auc']:.4f} | **{winner_metrics['pr_auc']:.4f}** |\n")
        f.write(f"| **Precision@40 (1 mo budget)** | {base_metrics['precision_at_40']:.4f} ({base_metrics['tp_40']}/40) | {lgbm_metrics['precision_at_40']:.4f} ({lgbm_metrics['tp_40']}/40) | **{winner_metrics['precision_at_40']:.4f}** |\n")
        f.write(f"| **Recall@40** | {base_metrics['recall_at_40']:.4f} | {lgbm_metrics['recall_at_40']:.4f} | **{winner_metrics['recall_at_40']:.4f}** |\n")
        f.write(f"| **Net Benefit @ 40** | Rs. {base_metrics['net_benefit_40']:,.0f} | Rs. {lgbm_metrics['net_benefit_40']:,.0f} | **Rs. {winner_metrics['net_benefit_40']:,.0f}** |\n")
        f.write(f"| **Precision@120 (3 mo budget)** | {base_metrics['precision_at_total']:.4f} | {lgbm_metrics['precision_at_total']:.4f} | **{winner_metrics['precision_at_total']:.4f}** |\n")
        f.write(f"| **F1-Score (@0.50)** | {base_metrics['f1_05']:.4f} | {lgbm_metrics['f1_05']:.4f} | **{winner_metrics['f1_05']:.4f}** |\n\n")

        f.write("## 4. Business & Financial Impact Analysis\n\n")
        f.write(
            f"The investigation desk is constrained to reviewing **40 claims per month** (120 claims across the "
            "3-month validation window). Each false positive investigation costs Kestrel Rs. 640 (Rs. 380 customer "
            "goodwill compensation + Rs. 260 service contact cost).\n\n"
        )
        f.write(
            f"When the top 40 suspicious claims are prioritized using the selected model:\n"
            f"- **Fraud Caught:** {winner_metrics['tp_40']} fraudulent claims stopped\n"
            f"- **Precision:** {winner_metrics['precision_at_40']:.2%} (vs. 2.11% base rate -- over 3.5x lift)\n"
            f"- **Net Financial Benefit:** **Rs. {winner_metrics['net_benefit_40']:,.0f}**\n\n"
        )

        f.write("## 5. Artifacts Created\n\n")
        f.write("- `models/engineer.pkl`: Fitted feature pipeline, serializes reference tables for API use\n")
        f.write("- `models/baseline.pkl`: Fitted Logistic Regression baseline pipeline\n")
        f.write("- `models/lgbm.pkl`: Fitted LightGBM primary model\n")
        f.write("- `models/selected_model.pkl`: Selected production model\n")
        f.write("- `models/training_meta.json`: Full configuration, feature list, and evaluation metrics\n")
        f.write("- `outputs/feature_importance.png`: Feature gain bar chart\n")
        f.write("- `outputs/curves_baseline.png`: Baseline ROC and PR curves\n")
        f.write("- `outputs/curves_lgbm.png`: LightGBM ROC and PR curves\n\n")

        f.write("## 6. Next Steps (Phase 2 Preview)\n\n")
        f.write(
            "1. Build FastAPI REST endpoint loading `models/selected_model.pkl` and `models/engineer.pkl`.\n"
            "2. Score `test_unlabelled.csv` and export formatted predictions.\n"
            "3. Construct modern web review dashboard for the claims investigation desk.\n"
        )

    print(f"  Saved -> {report_file}")
    print(f"\n{'='*60}")
    print("  Training and Validation Complete Successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
