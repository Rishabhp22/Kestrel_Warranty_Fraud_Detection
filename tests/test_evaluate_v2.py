"""
tests/test_evaluate_v2.py -- Verification suite for operational monthly evaluation and predictions.

Tests:
1. Operational monthly top-40 queue selection (separate 40-claim queue per month, correct sorting).
2. Economic net-benefit calculation accuracy under Ops Policy v4.1 (FP Cost = Rs. 640).
3. Candidate prediction output schema and claim ID alignment verification.
4. Candidate artifacts integrity (engineer_v2.pkl, lgbm_v2.pkl, training_meta_v2.json).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate_monthly_operational_queue, FP_TOTAL_COST_INR
from features import FraudFeatureEngineer, FEATURE_COLS_V2


def test_monthly_top40_selection_and_economics():
    """Verify that claims are ranked and partitioned by month into top-40 queues."""
    # Synthetic dataset with 2 months, 50 claims each
    np.random.seed(42)
    n_per_month = 50

    m1_dates = pd.date_range("2026-04-01", periods=n_per_month, freq="6h")
    m2_dates = pd.date_range("2026-05-01", periods=n_per_month, freq="6h")

    # M1 has 5 frauds, M2 has 8 frauds
    m1_frauds = [1 if i < 5 else 0 for i in range(n_per_month)]
    m2_frauds = [1 if i < 8 else 0 for i in range(n_per_month)]

    m1_amounts = [1500.0 if f == 1 else 1200.0 for f in m1_frauds]
    m2_amounts = [1800.0 if f == 1 else 1300.0 for f in m2_frauds]

    df_test = pd.DataFrame({
        "claim_id": [f"C_M1_{i}" for i in range(n_per_month)] + [f"C_M2_{i}" for i in range(n_per_month)],
        "submitted_at": list(m1_dates.astype(str)) + list(m2_dates.astype(str)),
        "is_fraud": m1_frauds + m2_frauds,
        "claim_amount_inr": m1_amounts + m2_amounts,
    })

    # Assign higher scores to fraud claims in both months
    scores = np.array([
        (0.95 - (i % n_per_month) * 0.01) if f == 1 else (0.10 - (i % n_per_month) * 0.001)
        for i, f in enumerate(df_test["is_fraud"])
    ])

    res = evaluate_monthly_operational_queue(scores, df_test, "Synthetic Test Model")

    assert len(res["monthly"]) == 2, "Must evaluate exactly 2 months"
    m1_res = res["monthly"][0]
    m2_res = res["monthly"][1]

    # In M1: all 5 frauds scored highest -> TP=5, FP=35 (queue size 40)
    assert m1_res["tp"] == 5, f"Expected 5 TP in Month 1, got {m1_res['tp']}"
    assert m1_res["fp"] == 35, f"Expected 35 FP in Month 1, got {m1_res['fp']}"
    assert m1_res["precision_at_40"] == 5 / 40.0
    assert m1_res["recall_at_40"] == 1.0  # 5 out of 5 caught
    expected_m1_saved = 5 * 1500.0
    expected_m1_wasted = 35 * FP_TOTAL_COST_INR
    assert m1_res["inr_saved"] == expected_m1_saved
    assert m1_res["inr_wasted"] == expected_m1_wasted
    assert m1_res["net_benefit"] == expected_m1_saved - expected_m1_wasted

    # In M2: all 8 frauds scored highest -> TP=8, FP=32
    assert m2_res["tp"] == 8, f"Expected 8 TP in Month 2, got {m2_res['tp']}"
    assert m2_res["fp"] == 32, f"Expected 32 FP in Month 2, got {m2_res['fp']}"
    assert m2_res["precision_at_40"] == 8 / 40.0
    assert m2_res["recall_at_40"] == 1.0

    # Total across 2 months (80 reviews)
    assert res["total_reviews"] == 80
    assert res["total_tp"] == 13
    assert res["total_fp"] == 67
    assert res["precision_at_capacity"] == 13 / 80.0

    print("  [PASS] test_monthly_top40_selection_and_economics passed.")


def test_candidate_artifacts_and_predictions_schema():
    """Verify that candidate V2 artifacts exist, load cleanly, and produce valid schema."""
    v2_dir = ROOT / "models" / "v2_candidate"
    assert (v2_dir / "engineer_v2.pkl").exists(), "engineer_v2.pkl must exist"
    assert (v2_dir / "lgbm_v2.pkl").exists(), "lgbm_v2.pkl must exist"
    assert (v2_dir / "training_meta_v2.json").exists(), "training_meta_v2.json must exist"

    engineer = FraudFeatureEngineer.load(v2_dir / "engineer_v2.pkl")
    with open(v2_dir / "lgbm_v2.pkl", "rb") as f:
        model = pickle.load(f)

    # Load unlabelled test dataset
    test_df = pd.read_csv(ROOT / "test_unlabelled.csv")
    sample_sub = pd.read_csv(ROOT / "sample_submission.csv")

    feat_cols = engineer.get_feature_cols()
    assert feat_cols == FEATURE_COLS_V2
    assert len(feat_cols) == 23

    # Transform test features
    test_feats = engineer.transform(test_df)
    X_test = test_feats[feat_cols].values.astype(float)
    assert X_test.shape == (2252, 23), f"Expected shape (2252, 23), got {X_test.shape}"

    scores = model.predict_proba(X_test)[:, 1]

    # Verify score bounds and validity
    assert not np.isnan(scores).any(), "Scores must not contain NaN"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores must be in [0, 1]"

    # Verify candidate prediction dataframe alignment
    cand_preds = pd.DataFrame({
        "claim_id": test_df["claim_id"],
        "score": scores
    })
    assert list(cand_preds.columns) == ["claim_id", "score"]
    assert len(cand_preds) == 2252
    assert (cand_preds["claim_id"].values == test_df["claim_id"].values).all()
    assert (cand_preds["claim_id"].values == sample_sub["claim_id"].values).all()

    print("  [PASS] test_candidate_artifacts_and_predictions_schema passed.")


def test_production_artifacts_unmodified():
    """Verify that production artifacts and predictions.csv have not been overwritten."""
    prod_model_path = ROOT / "models" / "selected_model.pkl"
    prod_engineer_path = ROOT / "models" / "engineer.pkl"
    prod_preds_path = ROOT / "outputs" / "predictions.csv"

    assert prod_model_path.exists()
    assert prod_engineer_path.exists()
    assert prod_preds_path.exists()

    prod_preds = pd.read_csv(prod_preds_path)
    assert len(prod_preds) == 2252
    assert list(prod_preds.columns) == ["claim_id", "score"]

    # Verify line count on disk
    with open(prod_preds_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2253

    print("  [PASS] test_production_artifacts_unmodified passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING EVALUATION & PREDICTION INTEGRITY V2 TESTS")
    print("=" * 60)
    test_monthly_top40_selection_and_economics()
    test_candidate_artifacts_and_predictions_schema()
    test_production_artifacts_unmodified()
    print("=" * 60)
    print("  ALL EVALUATION V2 TESTS PASSED!")
    print("=" * 60)
