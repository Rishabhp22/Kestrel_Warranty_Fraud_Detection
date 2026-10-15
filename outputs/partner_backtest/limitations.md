# Limitations of Partner-Level Risk Monitoring & Backtesting

**Document Date:** October 14, 2026  
**Project:** Kestrel Warranty Fraud Detection  
**Scope:** Offline Backtest Limitations, Timestamp Constraints, & Sample Size Sensitivity  

---

## 1. Absence of Resolution Timestamps & Label Availability Constraints

### 1.1 The Underlying Structural Issue
In the raw warranty dataset (`train.csv`), `submitted_at` is the only timestamp provided for each claim. There are **no timestamps recording**:
- When a warranty claim was selected for review.
- When the physical or desk investigation was completed.
- When the claim was officially confirmed as fraudulent or legitimate.

### 1.2 Impact on Feature Engineering
In warranty and insurance claims operations, fraud investigations typically require between 14 to 90 days to confirm fraud (e.g., waiting for parts recovery, engineer inspection, or legal audit). 
- If we were to calculate a feature such as `partner_historical_fraud_rate` or `partner_prior_fraud_count` based on past claims, we would be assuming that the fraud status of all prior claims was already known at the exact minute the new claim was submitted.
- Because this assumption is unprovable without resolution timestamps, **using historical fraud labels as partner features constitutes future-label leakage**.
- **Auditor Decision:** All partner risk monitoring signals in this backtest are strictly restricted to **unsupervised operational telemetry** (claim arrival velocity, 7-day/30-day volume bursts, and small-claim proportions). No ground-truth fraud labels were used in computing partner signals.

---

## 2. Extreme Partner Concentration & Small Sample Bias

### 2.1 The Two-Partner Anomaly (`SP3160` and `SP3232`)
The backtest results for the post-May policy period (May and June 2026) are overwhelmingly dominated by only **two service partners**:
- `SP3160`: 4 frauds in May, 6 frauds in June (10 total confirmed frauds).
- `SP3232`: 3 frauds in May, 4 frauds in June (7 total confirmed frauds).

Together, these two partners account for **17 out of 36 post-May frauds (47.2%)**, and **100% of the frauds identified by the Partner Audit Strategy (14 out of 14 caught)**.

### 2.2 Analytical Risk
- When an entire operational policy's apparent success hinges on detecting two specific entities in a network of 365 partners, the sample size is dangerously small ($N_{\text{partners}} = 2$).
- If `SP3160` and `SP3232` had altered their submission timing, dispersed claims across sister entities, or kept volume below the threshold, the Partner Audit Policy would have caught **zero fraud**, incurring ₹20,720 in unrecovered audit overhead.
- **Cautionary Rule:** Do not extrapolate the $+₹8,004$ net benefit observed in June as a permanent, network-wide operational certainty.

---

## 3. Regime Shift & Temporal Instability

### 3.1 Divergence Between Pre-Policy and Post-Policy Regimes
The backtest demonstrates that partner-level volume monitoring is highly regime-dependent:

| Operational Regime | Pre-Policy (April 2026) | Post-Policy (May–June 2026) |
| :--- | :--- | :--- |
| **Fraud Profile** | High-value, decentralized across 8 partners | Low-value (< ₹2,000), clustered in rogue partners |
| **Partner Volume Fraud Correlation** | Near zero (max fraud per partner was 2) | Very high (dealers submitting 6–7 small claims/mo) |
| **Partner Audit Net Benefit (Top 2)** | **-₹7,600.00 (Pure Loss)** | **+₹6,665.00 (Positive Recovery)** |
| **Claim-Level Model (V2 LightGBM)** | **+₹3,983.00 (Profitable)** | **-₹44,027.00 (Operating Deficit)** |

### 3.2 Implications
- A static partner audit policy applied across all months fails in decentralized environments like April.
- Conversely, a claim-level model queue fails in auto-approved small-claim environments like May and June.
- No single strategy was universally profitable across all three months of Q2.

---

## 4. Ratio Instability in Low-Volume Partners

### 4.1 Small-Denominator Inflation
In a dealership network where the median partner submits only 2 claims per month:
- A benign partner who submits 1 claim in a week following zero claims in previous weeks will exhibit a calculated `velocity_ratio` exceeding $4.0$:
  $$\text{Velocity Ratio} = \frac{1/7}{(0 + 1)/30} = \frac{0.143}{0.033} \approx 4.29$$
- Similarly, if that single claim is ₹1,400, their `small_claim_ratio_30d` becomes $1.0$ (100%).
- Raw ratio formulas without volume regularization artificially elevate low-volume benign partners, causing false positive spikes.
- **Correction Applied:** In this backtest, partner signals were regularized using absolute small-claim volume (`partner_small_vol_30d = partner_vol_30d * partner_small_ratio_30d`), which successfully suppressed single-claim noise.

---

## 5. Operational Cost & Dealer Audit Assumptions

### 5.1 Financial Parameters Used
The backtest relied on the following explicit operational assumptions:
1. **Partner Dealer Audit Cost:** Assumed at **₹2,500.00** per audit (covering field auditor time, dealer dispatch, or forensic ledger review).
2. **Clean Claim Friction Under Audit:** Assumed at **₹260.00** per legitimate claim held (customer service contact and notification overhead). Goodwill loss (₹380) is assumed waived if the dealer audit is conducted without antagonizing end-customers directly.
3. **Claim Desk Review Friction (Ops Policy v4.1):** Exactly **₹640.00** per false positive (₹380 goodwill + ₹260 service contact).

### 5.2 Sensitivity Warning
- If on-site physical dealer audits cost ₹5,000 or more (e.g. travel to remote tier-3 towns, legal representation), the quarterly net benefit of the Top 2 Partner Audit Strategy drops from **-₹935** to **-₹15,935**.
- If dealer audits lead to dealer churn or litigation costs, unmeasured indirect losses could occur.

---

## 6. Unresolved Ground-Truth Labels (67 Claims)

- 67 claims in Q2 validation have blank `is_fraud` labels (25 in April, 25 in May, 17 in June).
- In accordance with audit standards, these claims were kept in historical volume telemetry but excluded from numerator/denominator evaluations of fraud catch.
- If these unlabelled claims contain latent fraud, reported recall is an underestimate; if they are clean, reported precision is unaffected.

---

## 7. Summary Statement

Partner-level monitoring provides a powerful operational lens for detecting clustered small-claim fraud that bypasses individual claim-level scoring. However, its effectiveness is **strictly contingent on fraudsters exhibiting high-volume behavioral bursts**. It is an operational complement to claim-level ML models, not a universal replacement.
