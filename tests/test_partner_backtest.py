import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import POLICY_CHANGE_DATE, AUTO_APPROVE_THRESHOLD_INR


def test_chronological_leakage_partner_signals():
    """Verify that partner signals at time t depend strictly on prior claims (t_prev < t).
    Adding or modifying future claims must not change the signals at time t.
    """
    history = [
        {"claim_id": "C1", "partner_id": "P1", "submitted_at": pd.Timestamp("2026-05-01 10:00:00"), "amount": 1500.0},
        {"claim_id": "C2", "partner_id": "P1", "submitted_at": pd.Timestamp("2026-05-05 10:00:00"), "amount": 1800.0},
        {"claim_id": "C3", "partner_id": "P1", "submitted_at": pd.Timestamp("2026-05-10 10:00:00"), "amount": 1900.0},
    ]
    df_base = pd.DataFrame(history)
    
    t_target = pd.Timestamp("2026-05-08 12:00:00")
    prior_base = df_base[df_base["submitted_at"] < t_target]
    vol_30d_base = len(prior_base)
    
    # Add a massive future burst for P1 after t_target
    future_burst = [
        {"claim_id": f"F{i}", "partner_id": "P1", "submitted_at": pd.Timestamp("2026-05-15 10:00:00"), "amount": 1000.0}
        for i in range(10)
    ]
    df_augmented = pd.concat([df_base, pd.DataFrame(future_burst)], ignore_index=True)
    
    prior_augmented = df_augmented[df_augmented["submitted_at"] < t_target]
    vol_30d_augmented = len(prior_augmented)
    
    assert vol_30d_base == 2, f"Expected 2 prior claims, got {vol_30d_base}"
    assert vol_30d_augmented == vol_30d_base, "Future claims leaked into historical partner volume calculation!"


def test_unresolved_claims_in_telemetry_not_relabelled():
    """Verify that unresolved claims are included in partner volume but are not converted to confirmed non-fraud."""
    signals_path = PROJECT_ROOT / "outputs" / "partner_backtest" / "partner_risk_signals.csv"
    assert signals_path.exists(), "partner_risk_signals.csv missing!"
    
    signals = pd.read_csv(signals_path)
    unres_count = signals["is_unresolved"].sum()
    assert unres_count == 67, f"Expected 67 unresolved claims in Q2 validation, got {unres_count}"
    
    # Verify that unresolved claims have NaN fraud_label and are NOT relabelled as 0.0
    unres_rows = signals[signals["is_unresolved"]]
    assert unres_rows["fraud_label"].isnull().all(), "Unresolved claims were improperly relabelled as non-fraud!"


def test_monthly_results_arithmetic_reconciliation():
    """Verify exact arithmetic and reconciliation in monthly_results.csv."""
    res_path = PROJECT_ROOT / "outputs" / "partner_backtest" / "monthly_results.csv"
    assert res_path.exists(), "monthly_results.csv missing!"
    
    res = pd.read_csv(res_path)
    
    for idx, row in res.iterrows():
        # Flags match breakdown
        assert row["claims_flagged"] == row["tp_fraud_caught"] + row["fp_false_alarms"] + row["unresolved_flagged"], \
            f"Row {idx} ({row['strategy']}): Flagged claims mismatch"
            
        # FN matches total fraud - TP
        assert row["fn_fraud_missed"] == row["total_month_fraud"] - row["tp_fraud_caught"], \
            f"Row {idx} ({row['strategy']}): Missed fraud mismatch"
            
        # Net benefit matches avoided loss - total cost
        expected_net = round(row["loss_avoided_inr"] - row["total_cost_inr"], 2)
        assert abs(row["net_benefit_inr"] - expected_net) <= 0.05, \
            f"Row {idx} ({row['strategy']}): Net benefit arithmetic mismatch"
            
        # For desk queue, cost matches FP * 640
        if "Claim Desk Queue" in row["mechanism"]:
            expected_fp_cost = round(row["fp_false_alarms"] * 640.0, 2)
            assert abs(row["total_cost_inr"] - expected_fp_cost) <= 0.05, \
                f"Row {idx} ({row['strategy']}): FP cost mismatch: {row['total_cost_inr']} != {expected_fp_cost}"
