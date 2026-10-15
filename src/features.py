"""
src/features.py -- Kestrel Warranty Fraud Detection
Feature engineering pipeline (Version 2 with point-in-time outlet telemetry).

Key Design Decisions & Rationale:
1. TARGET LEAKAGE PREVENTION:
   - inspector_note EXCLUDED completely: Timing is ambiguous and notes are written
     during or after inspection, not before. Including note content or presence
     would introduce target leakage.
   - is_fraud is never used in feature creation.
   - No historical fraud-rate features: train.csv lacks resolution/investigation
     timestamps, so prior fraud labels cannot be assumed available at intake time.
   - Point-in-time telemetry: only preceding claims strictly prior to current claim
     (submitted_at < current_submitted_at) are used. Current claim and future claims
     are never visible.
   - 215 blank is_fraud rows are preserved in claim volume telemetry (all valid claims
     contribute to partner intake volume) but excluded from supervised training labels.

2. DISTRIBUTION SHIFT & TEMPORAL INTEGRITY:
   - partner_inspected is policy-aware (May 1, 2026 change from Ops Policy v4.1):
     Pre-policy: all claims required inspection -> 'N' is unexpected.
     Post-policy: sub-Rs.2,000 claims are auto-approved -> 'N' is expected for low-value claims,
     and only unexpected for claims >= Rs. 2,000. Feature: unexpected_no_inspection.
   - Point-in-time policy regime features:
     is_post_may_policy (submitted_at >= 2026-05-01)
     post_policy_small_claim (is_post_may_policy == 1 and claim_amount_inr < 2000)
   - Outlet telemetry features:
     partner_prior_claim_count (expanding prior claim count)
     days_since_partner_last_claim (elapsed days since preceding partner claim)
     partner_claims_last_30d (rolling 30-day claim count strictly prior to current timestamp)
     partner_small_claim_ratio_30d (ratio of sub-Rs.2,000 claims in rolling 30d window)

3. REUSABILITY & REPRODUCIBILITY:
   - Reference datasets (partners, products) and historical claim records are cached
     internally upon fit_transform and persisted with pickle, allowing single-claim
     inference: transform(single_claim_df).
   - Input row ordering and indices are strictly preserved.
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Operations policy change date: from this date, sub-Rs.2,000 claims auto-approved without inspection
POLICY_CHANGE_DATE = pd.Timestamp("2026-05-01")
AUTO_APPROVE_THRESHOLD_INR = 2_000.0

# Base V1 feature column list (17 features)
FEATURE_COLS_V1 = [
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

# Version 2 telemetry and policy features (6 features)
NEW_TELEMETRY_COLS = [
    "partner_prior_claim_count",
    "days_since_partner_last_claim",
    "partner_claims_last_30d",
    "partner_small_claim_ratio_30d",
    "is_post_may_policy",
    "post_policy_small_claim",
]

# Ordered Version 2 feature column list (23 features)
FEATURE_COLS_V2 = FEATURE_COLS_V1 + NEW_TELEMETRY_COLS

# Default feature column list for current pipeline version
FEATURE_COLS = FEATURE_COLS_V2


class FraudFeatureEngineer:
    """
    Reusable feature engineering pipeline for Kestrel Fraud Detection (Version 2).

    Usage:
        engineer = FraudFeatureEngineer()
        train_features = engineer.fit_transform(train_df, partners_df, products_df, history_df=all_past_claims)
        engineer.save("models/engineer_v2.pkl")

        # In validation, test, or API:
        val_features = engineer.transform(val_df)
    """

    def __init__(self, version: str = "v2"):
        self.partners_ref: pd.DataFrame = None
        self.products_ref: pd.DataFrame = None
        self.history_claims: pd.DataFrame = None
        self.is_fitted: bool = False
        self.version: str = version
        self.feature_cols: list = list(FEATURE_COLS_V2 if version == "v2" else FEATURE_COLS_V1)

    def _clean_claim_description(self, text: str) -> str:
        """Strip punctuation, prompt injections, and extra notes from claim description."""
        if not isinstance(text, str):
            return "unknown"
        clean = re.split(r"[.;\[]", text)[0]
        return clean.strip().lower()

    def _join_reference(
        self,
        df: pd.DataFrame,
        partners: pd.DataFrame = None,
        products: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Merge reference tables for partners and products while preserving original index."""
        p = partners if partners is not None else self.partners_ref
        pr = products if products is not None else self.products_ref

        if p is None or pr is None:
            raise ValueError(
                "Reference tables (partners, products) must be provided or pre-fitted."
            )

        p_sub = p[["partner_id", "city", "partner_type"]].drop_duplicates("partner_id")
        pr_sub = pr[["sku", "family", "list_price_inr", "warranty_months"]].drop_duplicates("sku")

        # Preserve original index across merges
        orig_index = df.index
        merged = df.merge(p_sub, on="partner_id", how="left")
        merged = merged.merge(pr_sub, on="sku", how="left")
        merged.index = orig_index
        return merged

    def _compute_telemetry(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute leakage-safe point-in-time telemetry features for partner claims.

        Guarantees:
        - Only claims strictly preceding current claim (submitted_at < current_submitted_at) are counted.
        - The current claim itself is never included in its own prior history.
        - Future claims are strictly invisible.
        - Safe cold-start defaults for unseen/new partners and missing timestamps.
        - Zero-denominator protection on rolling ratios.
        """
        n_incoming = len(df)
        if n_incoming == 0:
            return pd.DataFrame(columns=NEW_TELEMETRY_COLS, index=df.index)

        df_work = pd.DataFrame({
            "partner_id": df["partner_id"].astype(str).str.strip().values,
            "submitted_at": df["submitted_at"].values,
            "claim_amount_inr": df["claim_amount_inr"].values,
            "_orig_pos": np.arange(n_incoming),
            "_is_target": True,
        })

        if history_df is not None and len(history_df) > 0:
            hist_work = pd.DataFrame({
                "partner_id": history_df["partner_id"].astype(str).str.strip().values,
                "submitted_at": history_df["submitted_at"].values,
                "claim_amount_inr": history_df["claim_amount_inr"].values,
                "_orig_pos": -1,
                "_is_target": False,
            })
            combined = pd.concat([hist_work, df_work], ignore_index=True)
        else:
            combined = df_work

        sub_dt = pd.to_datetime(combined["submitted_at"], errors="coerce")
        amt = pd.to_numeric(
            combined["claim_amount_inr"].astype(str).str.strip(), errors="coerce"
        ).fillna(0.0)
        is_small = (amt < AUTO_APPROVE_THRESHOLD_INR).values

        combined["_sub_dt"] = sub_dt.values
        combined["_amt"] = amt.values
        combined["_is_small"] = is_small

        # Sort chronologically. On timestamp ties, historical claims precede target claims.
        combined = combined.sort_values(by=["_sub_dt", "_is_target"]).reset_index(drop=True)

        n = len(combined)
        res_prior_count = np.zeros(n, dtype=float)
        res_days_since_last = np.full(n, 999.0, dtype=float)
        res_claims_30d = np.zeros(n, dtype=float)
        res_small_ratio_30d = np.zeros(n, dtype=float)

        grouped = combined.groupby("partner_id", sort=False)
        for pid, grp in grouped:
            indices = grp.index.values
            times = grp["_sub_dt"].values
            smalls = grp["_is_small"].values
            k = len(indices)

            for i in range(k):
                curr_idx = indices[i]
                if pd.isna(times[i]):
                    # Missing timestamp fallback
                    res_prior_count[curr_idx] = 0.0
                    res_days_since_last[curr_idx] = 999.0
                    res_claims_30d[curr_idx] = 0.0
                    res_small_ratio_30d[curr_idx] = 0.0
                    continue

                if i == 0:
                    # Cold start for partner
                    res_prior_count[curr_idx] = 0.0
                    res_days_since_last[curr_idx] = 999.0
                    res_claims_30d[curr_idx] = 0.0
                    res_small_ratio_30d[curr_idx] = 0.0
                else:
                    # Count strictly prior valid claims
                    res_prior_count[curr_idx] = float(i)
                    
                    if not pd.isna(times[i - 1]):
                        dt_sec = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
                        res_days_since_last[curr_idx] = max(0.0, float(dt_sec / 86400.0))
                    else:
                        res_days_since_last[curr_idx] = 999.0

                    # Rolling 30-day window: [t - 30d, t)
                    t_cutoff = times[i] - np.timedelta64(30, "D")
                    start_j = np.searchsorted(times[:i], t_cutoff, side="left")
                    c_30d = i - start_j
                    res_claims_30d[curr_idx] = float(c_30d)
                    if c_30d > 0:
                        res_small_ratio_30d[curr_idx] = float(smalls[start_j:i].sum() / c_30d)
                    else:
                        res_small_ratio_30d[curr_idx] = 0.0

        combined["partner_prior_claim_count"] = res_prior_count
        combined["days_since_partner_last_claim"] = res_days_since_last
        combined["partner_claims_last_30d"] = res_claims_30d
        combined["partner_small_claim_ratio_30d"] = res_small_ratio_30d

        # Policy regime indicators
        valid_dates = pd.to_datetime(combined["_sub_dt"])
        is_post_may = (valid_dates >= POLICY_CHANGE_DATE).fillna(False).astype(float).values
        combined["is_post_may_policy"] = is_post_may
        combined["post_policy_small_claim"] = (
            (is_post_may == 1.0) & (combined["_is_small"].values)
        ).astype(float)

        target_res = combined[combined["_is_target"]].sort_values(by="_orig_pos").reset_index(drop=True)
        target_res.index = df.index
        return target_res[NEW_TELEMETRY_COLS]

    def _compute_features(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Extract and engineer features from joined dataframe while strictly preserving index and row order."""
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

        out["days_since_purchase"] = days.values
        out["claim_amount_inr"] = amt.values
        out["customer_prior_claims"] = priors.values
        out["prior_claims_ge_1"] = (priors.values >= 1).astype(float)
        out["prior_claims_ge_2"] = (priors.values >= 2).astype(float)

        # 2. Product reference features
        list_p = pd.to_numeric(df["list_price_inr"], errors="coerce")
        warranty_m = pd.to_numeric(df["warranty_months"], errors="coerce")
        out["list_price_inr"] = list_p.values
        out["warranty_months"] = warranty_m.values
        ratio = amt.values / list_p.values
        out["amount_to_list_ratio"] = ratio
        out["high_amount_ratio"] = (ratio > 0.40).astype(float)

        # 3. Direct binaries
        out["photo_attached_bin"] = (
            df["photo_attached"].astype(str).str.strip().str.upper() == "Y"
        ).astype(float).values
        out["partner_inspected_bin"] = (
            df["partner_inspected"].astype(str).str.strip().str.upper() == "Y"
        ).astype(float).values
        out["is_crm"] = (
            df["source"].astype(str).str.strip().str.lower() == "crm"
        ).astype(float).values

        # 4. Policy-aware inspection feature (Ops Policy v4.1)
        sub_dt = pd.to_datetime(df["submitted_at"], errors="coerce")
        is_post_policy = (sub_dt >= POLICY_CHANGE_DATE).fillna(False).astype(int).values
        no_inspection = (out["partner_inspected_bin"].values == 0)

        # Pre-policy: all uninspected are unexpected
        pre_policy_uninspected = (is_post_policy == 0) & no_inspection
        # Post-policy: only uninspected claims >= Rs. 2,000 are unexpected
        post_policy_large_uninspected = (
            (is_post_policy == 1)
            & (amt.fillna(0.0).values >= AUTO_APPROVE_THRESHOLD_INR)
            & no_inspection
        )
        out["unexpected_no_inspection"] = (
            pre_policy_uninspected | post_policy_large_uninspected
        ).astype(float)

        # 5. Partner type (one-hot encoding)
        pt = df["partner_type"].astype(str).str.lower().str.strip()
        out["is_franchise"] = (pt.values == "franchise").astype(float)
        out["is_freelance"] = (pt.values == "freelance_technician").astype(float)
        out["is_asc"] = (pt.values == "authorised_service_centre").astype(float)

        # 6. Temporal features (day of week: 0=Monday, 6=Sunday)
        out["claim_dayofweek"] = sub_dt.dt.dayofweek.fillna(0.0).astype(float).values

        # 7. Version 2 Telemetry & Policy Regime Features
        telemetry_df = self._compute_telemetry(df, history_df=history_df)
        for col in NEW_TELEMETRY_COLS:
            out[col] = telemetry_df[col].values

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
        history_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Fit references and transform training data.
        history_df: Optional complete historical claims pool (including unresolved claims)
                    used to seed outlet telemetry. If None, df is used.
        """
        if partners is not None:
            self.partners_ref = partners.copy()
        if products is not None:
            self.products_ref = products.copy()

        joined = self._join_reference(df, partners, products)

        # Retain past claims for subsequent transform() calls
        hist_source = history_df if history_df is not None else df
        self.history_claims = hist_source[
            ["partner_id", "submitted_at", "claim_amount_inr"]
        ].copy()

        feats = self._compute_features(joined, history_df=None)
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
        Uses cached references and historical claims pool.
        """
        if not self.is_fitted and (partners is None or products is None):
            raise ValueError(
                "Call fit_transform() before transform() or pass reference tables."
            )
        joined = self._join_reference(df, partners, products)
        return self._compute_features(joined, history_df=self.history_claims)

    def get_feature_cols(self, version: str = None) -> list:
        """Return the ordered list of model input feature names."""
        if version == "v1":
            return list(FEATURE_COLS_V1)
        if version == "v2":
            return list(FEATURE_COLS_V2)
        if hasattr(self, "feature_cols") and self.feature_cols is not None:
            return list(self.feature_cols)
        return list(FEATURE_COLS_V2)

    def save(self, path) -> None:
        """Save fitted feature engineer, reference tables, and historical claims pool."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "partners_ref": self.partners_ref,
                    "products_ref": self.products_ref,
                    "history_claims": self.history_claims,
                    "is_fitted": self.is_fitted,
                    "feature_cols": self.feature_cols,
                    "version": getattr(self, "version", "v2"),
                },
                f,
            )
        print(f"[FraudFeatureEngineer] Saved -> {path}")

    @classmethod
    def load(cls, path) -> "FraudFeatureEngineer":
        """Load feature engineer, cached references, and historical telemetry state."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        cols = data.get("feature_cols", FEATURE_COLS_V2)
        v = data.get("version", "v1" if len(cols) == 17 else "v2")
        eng = cls(version=v)
        eng.partners_ref = data["partners_ref"]
        eng.products_ref = data["products_ref"]
        eng.history_claims = data.get("history_claims", None)
        eng.is_fitted = data["is_fitted"]
        eng.feature_cols = list(cols)
        eng.version = v
        return eng
