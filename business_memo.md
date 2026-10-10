# Business Memorandum

**To:** Ritu Deshpande, Head of D2C Operations; Farhan Sheikh, Finance Controller; Meenal Joshi, Service Desk Manager  
**From:** Analytics & Engineering — Warranty Fraud Detection Project  
**Date:** October 2026  
**Re:** Kestrel Warranty Fraud Risk Model — Operational Readiness and Capability Assessment  
**Classification:** Internal — Operations Leadership

---

## 1. Purpose

This memo summarises the capability, limitations, and operational value of the machine learning fraud risk scoring system developed for Kestrel Home Appliances warranty claims. The system ranks incoming warranty claims by fraud likelihood to help the investigation desk allocate its limited monthly review capacity more effectively.

---

## 2. Operational Context

The investigation desk can review at most **40 claims per month**. Across the Q2 2026 test batch of 2,252 unlabelled claims, the historical fraud rate observed in supervised training was approximately 2.1% — roughly 1 in every 47 claims. In this low-prevalence setting, a randomly selected queue would yield approximately 1 confirmed fraud per 40 reviews. The model does meaningfully better than that.

---

## 3. Verified Model Performance (Out-of-Time Validation)

All metrics below are drawn from out-of-time validation on April–June 2026 claims, using a model trained strictly on April 2025–March 2026 data. No future information was used in training.

| Metric | LightGBM (Selected) | Logistic Regression (Baseline) |
|---|---|---|
| **Precision@40** (desk budget) | **10.0%** (4 frauds caught in top 40) | 5.0% (2 caught) |
| **Net Financial Benefit @ 40 reviews** | **+Rs. 4,122** | -Rs. 9,245 |
| Validation ROC-AUC | 0.6136 | 0.6238 |
| Validation PR-AUC | 0.0569 | 0.0653 |
| Recall@40 | 8.9% | 4.4% |

**Net benefit calculation basis:** Each false-positive investigation costs Kestrel Rs. 640 (Rs. 380 goodwill compensation + Rs. 260 service contact cost). The LightGBM model catches 4 frauds in the top 40, yielding a positive net benefit. The logistic regression baseline produces -Rs. 9,245 under the same assumptions.

The LightGBM model's ROC-AUC and PR-AUC are marginally lower than the baseline. It was selected because it **doubles Precision@40** and produces a **positive net financial benefit** — the metric Farhan Sheikh correctly identified as operationally meaningful.

---

## 4. What the Score Means — and What It Does Not

The model assigns each claim a **risk score** between 0 and 1. This score is an operational **ranking instrument**, not a calibrated probability of fraud. A score of 0.45 does not mean there is a 45% chance the claim is fraudulent.

**Important distinctions:**

- A **HIGH risk score** (>=0.30) flags a claim for priority human review. It is **not** a determination of fraud.
- A **LOW risk score** (<0.10) indicates the claim profile is consistent with routine cases. It is **not** a clearance.
- All fraud determinations remain with authorised human investigators. The model provides prioritisation, not verdicts.
- Genuine customers whose claims are held for investigation must be handled fairly, consistent with Kestrel's goodwill policy. Meenal Joshi's concern about painting all new partners with the same brush is reflected in the feature design: partner identity is not used directly; only the claim-level behavioural pattern is scored.

---

## 5. Key Design Decisions and Leakage Prevention

| Decision | Rationale |
|---|---|
| **`inspector_note` fully excluded** | Post-investigation notes are written after a claim is already under review. Using them in the model would be post-hoc target leakage and would make the system non-operational at claim intake. |
| **Partner onboarding date excluded** | During training, new partners showed near-zero fraud (0.2%); in validation they became the primary fraud vector (10.0%). Including this feature caused catastrophic score inversion (ROC-AUC collapsed to 0.30). Excluding it restored stable generalisation. |
| **Blank `is_fraud` labels excluded** | Zoho legacy export mapped active investigations to 0. Imputing these as non-fraud would have contaminated the negative training class. |
| **Temporal (out-of-time) validation** | Model trained on pre-April 2026 data; validated on April-June 2026 claims only. No future-to-past leakage. |

---

## 6. Groq AI Investigation Assistant

The system includes an AI-powered investigation assistance layer using Groq (llama-3.3-70b-versatile). Its role is to:

- Translate the model's TreeSHAP feature contributions into plain-language investigation notes for desk agents.
- Suggest claim-specific verification steps.
- Identify what cannot be determined from telemetry alone (e.g., physical part condition, original invoice records).

**The Groq assistant does not change or recalculate model scores.** It is an explanation and triage support tool. If the Groq API is unconfigured or unavailable, all prediction and scoring functionality operates normally without interruption. The assistant is **not** evidence of improved model accuracy and must not be cited as such in performance reporting.

---

## 7. Limitations and Recommended Next Steps

- **Model accuracy is moderate.** ROC-AUC of 0.61 reflects the genuine difficulty of detecting rare fraud from structured claim metadata alone, without access to real-time serial number databases or payment history.
- **The 40-claim desk budget is a binding constraint.** Expanding the review desk from 40 to 120 claims/month would provide a materially stronger fraud catch rate at modest additional cost.
- **Continued monitoring is recommended.** As new partner cohorts grow, fraud patterns may shift. The model should be retrained on an updated training window every 6-12 months.
- **No demographic or customer identity data was used** in the model, reducing regulatory risk associated with algorithmic decision-making.

---

*This memo references verified metrics from `outputs/evaluation_report.md`. All financial estimates use operational cost parameters from Kestrel Operations Policy v4.1.*
