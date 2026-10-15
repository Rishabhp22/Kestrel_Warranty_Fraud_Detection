"""
tests/test_features_v2.py -- Verification suite for Version 2 feature engineering.

Tests:
1. No current-claim or future-claim leakage in historical features.
2. Correct 30-day rolling window boundaries.
3. Cold-start handling: new/unseen partners and missing/malformed timestamps.
4. Unresolved claims included in volume telemetry without label contamination.
5. Identical feature schema during training and inference.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (
    FraudFeatureEngineer,
    FEATURE_COLS_V1,
    FEATURE_COLS_V2,
    NEW_TELEMETRY_COLS,
    POLICY_CHANGE_DATE,
    AUTO_APPROVE_THRESHOLD_INR,
)


def make_dummy_references():
    partners = pd.DataFrame([
        {"partner_id": "P1", "city": "Pune", "partner_type": "authorised_service_centre"},
        {"partner_id": "P2", "city": "Jaipur", "partner_type": "franchise"},
    ])
    products = pd.DataFrame([
        {"sku": "SKU1", "family": "Mixer Grinder", "list_price_inr": 4000.0, "warranty_months": 24},
        {"sku": "SKU2", "family": "Induction Cooktop", "list_price_inr": 3000.0, "warranty_months": 12},
    ])
    return partners, products


def test_leakage_and_future_independence():
    """Verify that future claims and current claims do not leak into telemetry features."""
    partners, products = make_dummy_references()
    engineer = FraudFeatureEngineer(version="v2")

    # Chronological claims for P1
    claims = pd.DataFrame([
        {
            "claim_id": "C1",
            "submitted_at": "2026-04-10 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 50,
            "claim_amount_inr": 1500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
            "is_fraud": 0,
        },
        {
            "claim_id": "C2",
            "submitted_at": "2026-04-20 12:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 60,
            "claim_amount_inr": 3500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
            "is_fraud": 1,
        },
        {
            "claim_id": "C3",
            "submitted_at": "2026-05-02 09:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 72,
            "claim_amount_inr": 1800.0,
            "photo_attached": "Y",
            "partner_inspected": "N",
            "source": "crm",
            "customer_prior_claims": 0,
            "is_fraud": 1,
        },
    ])

    feats = engineer.fit_transform(claims, partners, products)

    # C1 is the very first claim for P1 -> prior count must be 0, days_since must be 999.0
    assert feats.loc[0, "partner_prior_claim_count"] == 0.0, "C1 must have 0 prior claims"
    assert feats.loc[0, "days_since_partner_last_claim"] == 999.0, "C1 must default to 999.0 days since last"
    assert feats.loc[0, "partner_claims_last_30d"] == 0.0, "C1 must have 0 claims in last 30d"

    # C2 has only C1 prior
    assert feats.loc[1, "partner_prior_claim_count"] == 1.0, "C2 must have exactly 1 prior claim"
    assert round(feats.loc[1, "days_since_partner_last_claim"], 2) == 10.08, "C2 days since last must match diff from C1"
    assert feats.loc[1, "partner_claims_last_30d"] == 1.0, "C2 must have 1 prior claim in 30d"

    # C3 has C1 and C2 prior
    assert feats.loc[2, "partner_prior_claim_count"] == 2.0, "C3 must have exactly 2 prior claims"
    # C3 is post-May 1, 2026 -> is_post_may_policy must be 1.0, post_policy_small_claim must be 1.0 (amount=1800 < 2000)
    assert feats.loc[2, "is_post_may_policy"] == 1.0, "C3 must be flagged as post-May policy"
    assert feats.loc[2, "post_policy_small_claim"] == 1.0, "C3 must be flagged as post-policy small claim"
    assert feats.loc[0, "is_post_may_policy"] == 0.0, "C1 must be pre-May policy"
    assert feats.loc[0, "post_policy_small_claim"] == 0.0, "C1 must not be post-policy small claim"

    print("  [PASS] test_leakage_and_future_independence passed.")


def test_30_day_window_boundaries():
    """Verify that claims strictly outside the 30-day window are excluded from rolling counts."""
    partners, products = make_dummy_references()
    engineer = FraudFeatureEngineer(version="v2")

    claims = pd.DataFrame([
        # Day 0
        {
            "claim_id": "C1",
            "submitted_at": "2026-04-01 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 50,
            "claim_amount_inr": 1500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        },
        # Day 15
        {
            "claim_id": "C2",
            "submitted_at": "2026-04-16 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 65,
            "claim_amount_inr": 3500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        },
        # Day 35: C1 is 34 days prior (outside 30d window), C2 is 19 days prior (inside window)
        {
            "claim_id": "C3",
            "submitted_at": "2026-05-05 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 84,
            "claim_amount_inr": 1200.0,
            "photo_attached": "Y",
            "partner_inspected": "N",
            "source": "crm",
            "customer_prior_claims": 0,
        },
    ])

    feats = engineer.fit_transform(claims, partners, products)

    # For C3, partner_prior_claim_count = 2, but partner_claims_last_30d = 1 (only C2)
    assert feats.loc[2, "partner_prior_claim_count"] == 2.0
    assert feats.loc[2, "partner_claims_last_30d"] == 1.0, "C1 should fall outside the 30-day window"
    # C2 amount was 3500 (>= 2000), so small claim ratio in last 30d should be 0.0
    assert feats.loc[2, "partner_small_claim_ratio_30d"] == 0.0, "C2 is not a small claim"

    print("  [PASS] test_30_day_window_boundaries passed.")


def test_cold_start_and_missing_timestamps():
    """Verify safe defaults for unseen partners, zero counts, and missing timestamps."""
    partners, products = make_dummy_references()
    engineer = FraudFeatureEngineer(version="v2")

    claims_train = pd.DataFrame([
        {
            "claim_id": "C1",
            "submitted_at": "2026-04-01 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 50,
            "claim_amount_inr": 1500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        }
    ])
    engineer.fit_transform(claims_train, partners, products)

    # Transform claims with an unseen partner P99 and a missing timestamp
    claims_new = pd.DataFrame([
        {
            "claim_id": "C_NEW",
            "submitted_at": "2026-04-15 10:00:00",
            "partner_id": "P99_UNSEEN",
            "sku": "SKU1",
            "days_since_purchase": 30,
            "claim_amount_inr": 1000.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        },
        {
            "claim_id": "C_NOTIME",
            "submitted_at": "INVALID_OR_MISSING_DATE",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 40,
            "claim_amount_inr": 2500.0,
            "photo_attached": "N",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        },
    ])

    feats_new = engineer.transform(claims_new)

    # Check P99_UNSEEN
    assert feats_new.loc[0, "partner_prior_claim_count"] == 0.0
    assert feats_new.loc[0, "days_since_partner_last_claim"] == 999.0
    assert feats_new.loc[0, "partner_claims_last_30d"] == 0.0
    assert feats_new.loc[0, "partner_small_claim_ratio_30d"] == 0.0

    # Check missing timestamp row
    assert feats_new.loc[1, "partner_prior_claim_count"] == 0.0
    assert feats_new.loc[1, "days_since_partner_last_claim"] == 999.0
    assert feats_new.loc[1, "is_post_may_policy"] == 0.0
    assert feats_new.loc[1, "post_policy_small_claim"] == 0.0

    # Verify no NaN values in any engineered feature column
    assert not feats_new[FEATURE_COLS_V2].isnull().any().any(), "No NaN permitted in feature matrix"

    print("  [PASS] test_cold_start_and_missing_timestamps passed.")


def test_unresolved_claims_in_telemetry():
    """Verify that claims with blank is_fraud are counted in telemetry volume but kept clean."""
    partners, products = make_dummy_references()
    engineer = FraudFeatureEngineer(version="v2")

    claims_train = pd.DataFrame([
        {
            "claim_id": "C_BLANK",
            "submitted_at": "2026-04-01 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 50,
            "claim_amount_inr": 1500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
            "is_fraud": np.nan,  # Unresolved outcome
        },
        {
            "claim_id": "C_NEXT",
            "submitted_at": "2026-04-05 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 55,
            "claim_amount_inr": 1800.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
            "is_fraud": 0,
        },
    ])

    feats = engineer.fit_transform(claims_train, partners, products)

    # C_NEXT must count C_BLANK in volume telemetry
    assert feats.loc[1, "partner_prior_claim_count"] == 1.0
    assert feats.loc[1, "partner_claims_last_30d"] == 1.0
    assert feats.loc[1, "partner_small_claim_ratio_30d"] == 1.0  # C_BLANK was 1500 < 2000

    print("  [PASS] test_unresolved_claims_in_telemetry passed.")


def test_identical_feature_schema():
    """Verify that training and inference output exactly identical ordered feature schemas."""
    partners, products = make_dummy_references()
    engineer = FraudFeatureEngineer(version="v2")

    claims_train = pd.DataFrame([
        {
            "claim_id": "C1",
            "submitted_at": "2026-04-01 10:00:00",
            "partner_id": "P1",
            "sku": "SKU1",
            "days_since_purchase": 50,
            "claim_amount_inr": 1500.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "source": "crm",
            "customer_prior_claims": 0,
        }
    ])
    train_feats = engineer.fit_transform(claims_train, partners, products)
    val_feats = engineer.transform(claims_train)

    train_cols = [c for c in train_feats.columns if c in FEATURE_COLS_V2]
    val_cols = [c for c in val_feats.columns if c in FEATURE_COLS_V2]

    assert train_cols == val_cols == FEATURE_COLS_V2
    assert len(train_cols) == 23

    print("  [PASS] test_identical_feature_schema passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING FRAUD FEATURE ENGINEER V2 TESTS")
    print("=" * 60)
    test_leakage_and_future_independence()
    test_30_day_window_boundaries()
    test_cold_start_and_missing_timestamps()
    test_unresolved_claims_in_telemetry()
    test_identical_feature_schema()
    print("=" * 60)
    print("  ALL FEATURE ENGINEER V2 TESTS PASSED!")
    print("=" * 60)
