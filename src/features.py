"""
src/features.py -- Kestrel Warranty Fraud Detection
Feature engineering pipeline, shared by training, evaluation, and API.

Key Design Decisions & Rationale:
1. TARGET LEAKAGE PREVENTION:
   - inspector_note EXCLUDED completely: Timing is ambiguous and notes are written
     during or after inspection, not before. Including note content or presence
     would introduce target leakage.
   - is_fraud is never used in feature creation.
   - No target-encoded statistics (e.g., partner historical fraud rates) to prevent
     overfitting and target leakage across time splits.
   - 215 blank is_fraud rows are preserved as unknown, never imputed as 0.

2. DISTRIBUTION SHIFT & TEMPORAL INTEGRITY:
   - partner_age_days & is_new_partner EXCLUDED from model features:
     During the training period (Apr 2025 - Mar 2026), newly onboarded partners
     had negligible claims and near-zero fraud (0.2%). Models trained on this feature
     learned a strongly negative coefficient, assuming new partners are "safe".
     In validation (Apr - Jun 2026) and test, new partners became the primary fraud vector (10.0%).
     Excluding partner age prevents catastrophic negative transfer (which collapses ROC-AUC to 0.30).
   - partner_inspected is policy-aware (May 1, 2026 change):
     Pre-policy: all claims required inspection -> 'N' is unexpected.
     Post-policy: sub-Rs.2,000 claims are auto-approved -> 'N' is expected for low-value claims,
     and only unexpected for claims >= Rs. 2,000. Feature: unexpected_no_inspection.
   - Zoho vs CRM timestamps: Zoho timestamps were stored in UTC and not adjusted upon CRM
     migration. Hourly features are noisy across the boundary, so day-of-week is used.
   - Prompt injections in free-text fields are sanitized/ignored.

3. REUSABILITY & REPRODUCIBILITY:
   - Reference datasets (partners, products) are cached internally upon fit_transform
     and persisted with pickle, allowing single-claim inference: transform(single_claim_df).
   - Input row ordering and indices are strictly preserved.
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Operations policy change date: from this date, sub-Rs.2,000 claims auto-approved without inspection
POLICY_CHANGE_DATE = pd.Timestamp("2026-05-01")
AUTO_APPROVE_THRESHOLD_INR = 2_000

# Feature column list (ordered -- matches model input order)
FEATURE_COLS = [
    # Direct numeric from claim
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "prior_claims_ge_1",
    "prior_claims_ge_2",
    # Product reference
    "list_price_inr",
    "warranty_months",
    "amount_to_list_ratio",
    "high_amount_ratio",
    # Direct binaries
    "photo_attached_bin",
    "partner_inspected_bin",
    "is_crm",
    # Policy-aware inspection
    "unexpected_no_inspection",
    # Partner type (one-hot encoded)
    "is_franchise",
    "is_freelance",
    "is_asc",
    # Temporal
    "claim_dayofweek",
]


class FraudFeatureEngineer:
    """
    Reusable feature engineering pipeline for Kestrel Fraud Detection.

    Usage:
        engineer = FraudFeatureEngineer()
        train_features = engineer.fit_transform(train_df, partners_df, products_df)
        engineer.save("models/engineer.pkl")

        # In validation, test, or API:
        val_features = engineer.transform(val_df)
    """

    def __init__(self):
        self.partners_ref: pd.DataFrame = None
        self.products_ref: pd.DataFrame = None
        self.is_fitted: bool = False

    def _clean_claim_description(self, text: str) -> str:
        """Strip punctuation, prompt injections, and extra notes from claim description."""
        if not isinstance(text, str):
            return "unknown"
        # Truncate at periods, semicolons, brackets (filters prompt injections)
        clean = re.split(r"[.;\[]", text)[0]
        return clean.strip().lower()

    def _join_reference(
        self,
        df: pd.DataFrame,
        partners: pd.DataFrame = None,
        products: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Merge reference tables for partners and products."""
        p = partners if partners is not None else self.partners_ref
        pr = products if products is not None else self.products_ref

        if p is None or pr is None:
            raise ValueError(
                "Reference tables (partners, products) must be provided or pre-fitted."
            )

        p_sub = p[["partner_id", "city", "partner_type"]].drop_duplicates("partner_id")
        pr_sub = pr[["sku", "family", "list_price_inr", "warranty_months"]].drop_duplicates("sku")

        merged = df.merge(p_sub, on="partner_id", how="left")
        merged = merged.merge(pr_sub, on="sku", how="left")
        return merged

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features from joined dataframe while preserving index/order."""
        orig_index = df.index
        out = pd.DataFrame(index=orig_index)

        # 1. Direct claim numeric
        days = pd.to_numeric(
            df["days_since_purchase"].astype(str).str.strip(), errors="coerce"
        )
        amt = pd.to_numeric(
            df["claim_amount_inr"].astype(str).str.strip(), errors="coerce"
        )
        priors = pd.to_numeric(
            df["customer_prior_claims"].astype(str).str.strip(), errors="coerce"
        ).fillna(0)

        out["days_since_purchase"] = days
        out["claim_amount_inr"] = amt
        out["customer_prior_claims"] = priors
        out["prior_claims_ge_1"] = (priors >= 1).astype(float)
        out["prior_claims_ge_2"] = (priors >= 2).astype(float)

        # 2. Product reference features
        list_p = pd.to_numeric(df["list_price_inr"], errors="coerce")
        warranty_m = pd.to_numeric(df["warranty_months"], errors="coerce")
        out["list_price_inr"] = list_p
        out["warranty_months"] = warranty_m
        ratio = amt / list_p
        out["amount_to_list_ratio"] = ratio
        out["high_amount_ratio"] = (ratio > 0.40).astype(float)

        # 3. Direct binaries
        out["photo_attached_bin"] = (
            df["photo_attached"].astype(str).str.strip().str.upper() == "Y"
        ).astype(float)
        out["partner_inspected_bin"] = (
            df["partner_inspected"].astype(str).str.strip().str.upper() == "Y"
        ).astype(float)
        out["is_crm"] = (
            df["source"].astype(str).str.strip().str.lower() == "crm"
        ).astype(float)

        # 4. Policy-aware inspection feature
        # May 1, 2026: Sub-Rs.2,000 claims auto-approved without inspection
        sub_dt = pd.to_datetime(df["submitted_at"], errors="coerce")
        is_post_policy = (sub_dt >= POLICY_CHANGE_DATE).astype(int)
        no_inspection = (out["partner_inspected_bin"] == 0)

        # Pre-policy: all uninspected are unexpected
        pre_policy_uninspected = (is_post_policy == 0) & no_inspection
        # Post-policy: only uninspected claims >= Rs. 2,000 are unexpected
        post_policy_large_uninspected = (
            (is_post_policy == 1)
            & (amt >= AUTO_APPROVE_THRESHOLD_INR)
            & no_inspection
        )
        out["unexpected_no_inspection"] = (
            pre_policy_uninspected | post_policy_large_uninspected
        ).astype(float)

        # 5. Partner type (one-hot encoding)
        pt = df["partner_type"].astype(str).str.lower().str.strip()
        out["is_franchise"] = (pt == "franchise").astype(float)
        out["is_freelance"] = (pt == "freelance_technician").astype(float)
        out["is_asc"] = (pt == "authorised_service_centre").astype(float)

        # 6. Temporal features (day of week: 0=Monday, 6=Sunday)
        out["claim_dayofweek"] = sub_dt.dt.dayofweek.astype(float)

        # Retain original labels and identifiers if present
        if "is_fraud" in df.columns:
            out["is_fraud"] = df["is_fraud"].values
        if "claim_id" in df.columns:
            out["claim_id"] = df["claim_id"].values
        if "submitted_at" in df.columns:
            out["submitted_at"] = sub_dt.values

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        partners: pd.DataFrame = None,
        products: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Fit references and transform training data.
        """
        if partners is not None:
            self.partners_ref = partners.copy()
        if products is not None:
            self.products_ref = products.copy()

        joined = self._join_reference(df, partners, products)
        feats = self._compute_features(joined)
        self.is_fitted = True
        return feats

    def transform(
        self,
        df: pd.DataFrame,
        partners: pd.DataFrame = None,
        products: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Transform validation, test, or production claims.
        Uses cached references if partners/products are not supplied.
        """
        if not self.is_fitted and (partners is None or products is None):
            raise ValueError(
                "Call fit_transform() before transform() or pass reference tables."
            )
        joined = self._join_reference(df, partners, products)
        return self._compute_features(joined)

    def get_feature_cols(self) -> list:
        """Return the ordered list of model input feature names."""
        return list(FEATURE_COLS)

    def save(self, path) -> None:
        """Save fitted feature engineer and reference tables."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "partners_ref": self.partners_ref,
                    "products_ref": self.products_ref,
                    "is_fitted": self.is_fitted,
                    "feature_cols": FEATURE_COLS,
                },
                f,
            )
        print(f"[FraudFeatureEngineer] Saved -> {path}")

    @classmethod
    def load(cls, path) -> "FraudFeatureEngineer":
        """Load feature engineer and cached references."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        eng = cls()
        eng.partners_ref = data["partners_ref"]
        eng.products_ref = data["products_ref"]
        eng.is_fitted = data["is_fitted"]
        return eng
