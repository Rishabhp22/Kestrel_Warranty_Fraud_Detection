"""
api/explainer.py -- Human-readable explanation generator using TreeSHAP.

Generates 3-5 clear, business-friendly explanations for model predictions.
CRITICAL LEAKAGE PREVENTION: inspector_note is NEVER used as a feature or explanation.
"""

from typing import List
import pandas as pd


def get_risk_level(score: float) -> str:
    """
    Classify score into presentation/operational risk bands:
    - score < 0.10: LOW
    - 0.10 <= score < 0.30: MEDIUM
    - score >= 0.30: HIGH
    
    NOTE: These are operational risk triage bands, not calibrated posterior probabilities.
    """
    if score >= 0.30:
        return "HIGH"
    elif score >= 0.10:
        return "MEDIUM"
    else:
        return "LOW"


def generate_human_reasons(
    row_data: dict,
    features_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    top_n: int = 4
) -> List[str]:
    """
    Generate 3 to 5 plain-English explanations using native LightGBM TreeSHAP contributions.
    """
    X = features_df[feature_cols].values.astype(float)

    # Compute exact TreeSHAP contributions in log-odds space
    contribs = model.booster_.predict(X, pred_contrib=True)[0, :-1]

    # Sort features by positive impact on fraud score (highest risk drivers first)
    pairs = list(zip(feature_cols, features_df.iloc[0], contribs))
    pairs.sort(key=lambda x: -x[2])

    reasons = []
    seen_categories = set()

    for col, val, shap_val in pairs:
        desc = None

        if col in ["amount_to_list_ratio", "high_amount_ratio"] and "amount_ratio" not in seen_categories:
            ratio = float(features_df.iloc[0].get("amount_to_list_ratio", 0))
            if ratio > 0.40:
                desc = f"Claim amount is unusually high relative to product list price ({ratio:.0%} of retail price)"
            elif ratio > 0.25:
                desc = f"Claim amount represents a significant fraction of product list price ({ratio:.0%})"
            seen_categories.add("amount_ratio")

        elif col in ["customer_prior_claims", "prior_claims_ge_1", "prior_claims_ge_2"] and "priors" not in seen_categories:
            priors = int(features_df.iloc[0].get("customer_prior_claims", 0))
            if priors >= 2:
                desc = f"Customer has multiple prior warranty claims on file ({priors} prior claims)"
            elif priors == 1:
                desc = "Customer has a prior warranty claim history"
            seen_categories.add("priors")

        elif col == "unexpected_no_inspection" and "inspect" not in seen_categories:
            if val == 1.0:
                desc = "Missing mandatory partner physical inspection under operational policy"
                seen_categories.add("inspect")

        elif col == "partner_inspected_bin" and "inspect" not in seen_categories:
            if val == 0.0:
                desc = "No physical partner inspection recorded for this claim"
                seen_categories.add("inspect")

        elif col == "is_franchise" and "partner_type" not in seen_categories:
            if val == 1.0:
                desc = "Submitted by a franchise partner (statistically elevated risk profile)"
                seen_categories.add("partner_type")

        elif col == "claim_amount_inr" and "amount_abs" not in seen_categories:
            amt = float(val)
            if amt >= 3000:
                desc = f"High absolute claim payout amount requested (Rs. {amt:,.0f})"
                seen_categories.add("amount_abs")

        elif col == "days_since_purchase" and "purchase_days" not in seen_categories:
            d = float(val)
            if d < 90:
                desc = f"Early-failure claim filed shortly after purchase ({int(d)} days)"
                seen_categories.add("purchase_days")
            elif d > 365:
                desc = f"Claim filed after extended usage period ({int(d)} days since purchase)"
                seen_categories.add("purchase_days")

        elif col == "photo_attached_bin" and "photo" not in seen_categories:
            if val == 0.0:
                desc = "No photographic documentation attached by the partner"
                seen_categories.add("photo")

        if desc and desc not in reasons:
            reasons.append(desc)
            if len(reasons) >= top_n:
                break

    # Ensure at least 3 human-readable reasons are provided
    dsp = int(features_df.iloc[0].get("days_since_purchase", 0))
    amt = float(features_df.iloc[0].get("claim_amount_inr", 0))
    ratio = float(features_df.iloc[0].get("amount_to_list_ratio", 0))

    if len(reasons) < 3 and "photo" not in seen_categories and features_df.iloc[0].get("photo_attached_bin", 1) == 0:
        reasons.append("No supporting photographic documentation attached")
        seen_categories.add("photo")

    if len(reasons) < 3 and "purchase_days" not in seen_categories:
        reasons.append(f"Claim submitted at {dsp} days post-purchase")
        seen_categories.add("purchase_days")

    if len(reasons) < 3 and "amount_ratio" not in seen_categories and ratio > 0:
        reasons.append(f"Claim amount represents {ratio:.0%} of appliance list price")
        seen_categories.add("amount_ratio")

    if len(reasons) < 3:
        reasons.append(f"Claim value recorded at Rs. {amt:,.0f}")

    if len(reasons) < 3:
        reasons.append("Operational claim profile evaluated against risk parameters")

    return reasons[:5]
