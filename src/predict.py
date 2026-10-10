"""
src/predict.py -- Generate final predictions on test_unlabelled.csv.

Outputs:
    outputs/predictions.csv (claim_id,score)
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FraudFeatureEngineer

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def generate_predictions():
    print("=" * 60)
    print("  Generating Final Predictions -- Kestrel Fraud Detection")
    print("=" * 60)

    # 1. Load artifacts
    print("\n1. Loading models and test data...")
    engineer_path = MODELS_DIR / "engineer.pkl"
    model_path = MODELS_DIR / "selected_model.pkl"

    if not engineer_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {MODELS_DIR}")

    engineer = FraudFeatureEngineer.load(engineer_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(ROOT / "test_unlabelled.csv")
    sample_sub = pd.read_csv(ROOT / "sample_submission.csv")

    n_test = len(test_df)
    print(f"  Loaded test_unlabelled.csv: {n_test:,} rows")
    print(f"  Loaded sample_submission.csv: {len(sample_sub):,} rows")

    # 2. Extract features
    print("\n2. Transforming test features...")
    feat_cols = engineer.get_feature_cols()
    test_feats = engineer.transform(test_df)
    X_test = test_feats[feat_cols].values.astype(float)
    print(f"  Feature matrix shape: {X_test.shape}")

    # 3. Generate raw probability scores
    print("\n3. Generating model predictions (raw probabilities)...")
    raw_scores = model.predict_proba(X_test)[:, 1]

    # Verify score range
    assert (raw_scores >= 0.0).all() and (raw_scores <= 1.0).all(), "Scores must be within [0, 1]!"
    assert not np.isnan(raw_scores).any(), "Scores contain NaN values!"

    # 4. Construct prediction dataframe
    pred_df = pd.DataFrame({
        "claim_id": test_df["claim_id"],
        "score": raw_scores
    })

    # 5. Verification checks
    print("\n4. Running strict verification checks...")
    # Check column names
    assert list(pred_df.columns) == ["claim_id", "score"], f"Columns mismatch: {list(pred_df.columns)}"
    print("  [PASSED] Exactly 2 columns: ['claim_id', 'score']")

    # Check row count
    assert len(pred_df) == 2252, f"Expected 2,252 rows, got {len(pred_df)}"
    print(f"  [PASSED] Exactly 2,252 data rows (2,253 lines with header)")

    # Check ordering matches test_unlabelled and sample_submission
    assert (pred_df["claim_id"].values == test_df["claim_id"].values).all(), "Order does not match test_unlabelled!"
    print("  [PASSED] Claim ID ordering matches test_unlabelled.csv exactly")

    assert (pred_df["claim_id"].values == sample_sub["claim_id"].values).all(), "Claim IDs do not match sample_submission!"
    print("  [PASSED] Claim IDs match sample_submission.csv exactly")

    # 6. Save output
    out_file = OUTPUTS_DIR / "predictions.csv"
    pred_df.to_csv(out_file, index=False)
    print(f"\n5. Saved predictions to: {out_file}")

    # Verify file lines on disk
    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    line_count = len(lines)
    print(f"  File line count on disk: {line_count} (header + 2,252 data rows)")
    assert line_count == 2253, f"Expected 2,253 lines in CSV, got {line_count}"
    print("  [PASSED] Verified exactly 2,253 lines in predictions.csv")

    # 7. Print summary statistics
    min_score = float(raw_scores.min())
    mean_score = float(raw_scores.mean())
    max_score = float(raw_scores.max())
    p50_score = float(np.median(raw_scores))
    p90_score = float(np.percentile(raw_scores, 90))
    p99_score = float(np.percentile(raw_scores, 99))

    print("\n" + "=" * 60)
    print("  SCORE DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Min Score      : {min_score:.6f}")
    print(f"  Median Score   : {p50_score:.6f}")
    print(f"  Mean Score     : {mean_score:.6f}")
    print(f"  90th Percentile: {p90_score:.6f}")
    print(f"  99th Percentile: {p99_score:.6f}")
    print(f"  Max Score      : {max_score:.6f}")

    # Risk band distribution (operational thresholds)
    high_count = int((raw_scores >= 0.30).sum())
    med_count = int(((raw_scores >= 0.10) & (raw_scores < 0.30)).sum())
    low_count = int((raw_scores < 0.10).sum())
    print("\n  Operational Risk Bands:")
    print(f"    HIGH   (score >= 0.30)        : {high_count:4d} ({high_count/n_test:.2%})")
    print(f"    MEDIUM (0.10 <= score < 0.30) : {med_count:4d} ({med_count/n_test:.2%})")
    print(f"    LOW    (score < 0.10)         : {low_count:4d} ({low_count/n_test:.2%})")

    # 8. Print Top 10 Highest Risk Claims
    print("\n" + "=" * 60)
    print("  TOP 10 HIGHEST-RISK CLAIMS (Test Set)")
    print("=" * 60)
    top10_idx = np.argsort(raw_scores)[::-1][:10]
    top10_df = test_df.iloc[top10_idx].copy()
    top10_df["fraud_score"] = raw_scores[top10_idx]

    print(f"{'Rank':<5} {'Claim ID':<10} {'Score':<8} {'Amount':<10} {'Prior':<6} {'Partner':<9} {'Description'}")
    print("-" * 65)
    for rank, (idx, row) in enumerate(top10_df.iterrows(), 1):
        desc = str(row["claim_description"])[:25]
        print(f"{rank:<5} {row['claim_id']:<10} {row['fraud_score']:<8.4f} Rs.{float(row['claim_amount_inr']):>7,.0f} {row['customer_prior_claims']:<6} {row['partner_id']:<9} {desc}")
    print("=" * 60 + "\n")

    return pred_df


if __name__ == "__main__":
    generate_predictions()
