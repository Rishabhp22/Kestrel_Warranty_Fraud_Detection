# Kestrel Warranty Fraud Detection — Offline Partner-Level Risk Monitoring Backtest Report

**Report Date:** October 14, 2026  
**Auditor / Specialist:** Senior Fraud Analytics Specialist & ML System Auditor  
**Scope:** Point-in-Time Partner Monitoring, Chronological Backtesting, Baseline Comparisons, & Operational Economic Simulation  
**Evaluation Dataset:** Strict Temporal Q2 Validation Split (April, May, June 2026; $N = 2,198$ claims: 2,131 labeled, 67 unresolved)  
**Safety & Immutability:** Strict Read-Only Audit. No production models, feature pipelines, predictions (`outputs/predictions.csv`), or submission files were modified, retrained, or overwritten.

---

## 1. Executive Summary & Core Backtest Findings

This backtest evaluated whether point-in-time partner-level monitoring signals add measurable operational value beyond claim-level machine learning models (V1 and V2 LightGBM), particularly in detecting post-May 2026 small-claim fraud rings.

### Summary Comparison Table (Q2 Validation Totals: April – June 2026)

| Operational Strategy | Operational Mechanism | Total Claims Flagged | Partners Flagged | Confirmed Fraud (TP) | False Alarms (FP) | Unresolved Claims | Precision (%) | Recall (%) | Small Fraud (< ₹2k) Caught | Gross Loss Avoided | Total Operational Cost | Net Operating Benefit (Q2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Selection Baseline (Seed=42)** | Claim Desk Queue (40/mo) | 120 | 114 | 2 | 115 | 3 | 1.71% | 4.44% | 2 / 40 | ₹2,662.00 | ₹73,600.00 | **-₹70,938.00** |
| **Pure Partner Risk Signal** | Claim Desk Queue (40/mo) | 120 | 38 | 15 | 100 | 5 | 13.04% | 33.33% | 15 / 40 | ₹20,557.00 | ₹64,000.00 | **-₹43,443.00** |
| **V1 LightGBM Baseline** | Claim Desk Queue (40/mo) | 120 | 107 | 5 | 110 | 5 | 4.35% | 11.11% | 2 / 40 | ₹29,157.00 | ₹70,400.00 | **-₹41,243.00** |
| **V2 LightGBM Candidate** | Claim Desk Queue (40/mo) | 120 | 104 | 6 | 110 | 4 | 5.17% | 13.33% | 3 / 40 | ₹30,356.00 | ₹70,400.00 | **-₹40,044.00** |
| **Hybrid (V2 Model + Partner Signal)** | Claim Desk Queue (40/mo) | 120 | 102 | 6 | 110 | 4 | 5.17% | 13.33% | 3 / 40 | ₹31,144.00 | ₹70,400.00 | **-₹39,256.00** |
| **Partner Volume Baseline** | Claim Desk Queue (40/mo) | 120 | 32 | **16** | **98** | 6 | **14.04%** | **35.56%** | **16 / 40** | ₹23,753.00 | ₹62,720.00 | **-₹38,967.00** |
| **Partner Dealer Audit (Top 5 Partners/Mo)** | Partner Audit (5 Partners/mo) | 83 | 15 | **16** | **65** | 2 | **19.75%** | **35.56%** | **16 / 40** | ₹22,461.00 | ₹54,400.00 | **-₹31,939.00** |
| **Partner Dealer Audit (Top 3 Partners/Mo)** | Partner Audit (3 Partners/mo) | 50 | 9 | 14 | 35 | 1 | 28.57% | 31.11% | 14 / 40 | ₹19,785.00 | ₹31,600.00 | **-₹11,815.00** |
| **Partner Dealer Audit (Top 2 Partners/Mo)** | Partner Audit (2 Partners/mo) | 37 | 6 | 14 | **22** | 1 | **38.89%** | **31.11%** | 14 / 40 | ₹19,785.00 | **₹20,720.00** | **-₹935.00** |

*(Note: In evaluations with unresolved claims, precision is computed on labeled claims: $\frac{\text{TP}}{\text{TP} + \text{FP}}$. Unresolved claims are tracked separately).*

---

### Core Auditor Answers to Backtest Questions

1. **Did partner-level monitoring outperform V1 and V2?**
   - **Under the fixed 40-claim desk review queue (Ops Policy v4.1):** **Partially.** Prioritizing claims by partner volume/signals dramatically increased fraud sensitivity on small claims (catching 16 frauds vs. 6 for V2 and 5 for V1), but because false alarms still cost ₹640 per claim, the net benefit remained deeply negative (-₹38,967 for Volume Baseline vs. -₹40,044 for V2).
   - **Under the Partner Dealer Audit mechanism (separate capacity):** **Yes, substantially.** Auditing the Top 2 partners per month caught 14 small frauds with only 22 false alarms, cutting total costs from ₹70,400 down to ₹20,720 and reducing the quarterly loss to **-₹935** (nearly break-even). In post-policy May and June combined, it generated **+₹6,665.00 positive net recovery**.
2. **Has any approach demonstrated positive net benefit across Q2?**
   - **Across all of Q2 (April, May, June):** **NO.** Every strategy incurred an overall operating loss over the full 3-month quarter. Negative net benefit is an operating loss, NOT a positive outcome.
   - **Post-Policy Specifically (May and June):** **YES.** The **Partner Dealer Audit Policy (Top 2 Partners/Mo)** achieved a positive net benefit of **+₹8,004.00** in June 2026, and **+₹6,665.00** across May + June combined.
3. **Do one or two partners dominate the apparent improvement?**
   - **YES, ABSOLUTELY.** **Two partners (`SP3160` and `SP3232`) account for 100% of all fraud caught by the partner audit policy (14 out of 14 frauds)**. If these two partners are removed, the strategy caught zero fraud. The apparent success is concentrated in a tiny sample ($N=2$).
4. **Is the performance stable across months?**
   - **NO.** In April (pre-policy regime), partner volume monitoring failed completely (TP=0, net benefit = -₹7,600). It only succeeded post-May when fraudsters exploited auto-approval via high-volume claim bursts.

---

## 2. Chronological Information Ledger & Scoring Date Constraints

### 2.1 Information Available at Scoring Time $t$
In `train.csv`, the only available timestamp is `submitted_at`. There are **no resolution or investigation completion timestamps**.
- When claim $i$ is submitted at time $t_i$, operations knows:
  1. The historical timestamps of all claims submitted prior to $t_i$ ($t_j < t_i$).
  2. The claimed amounts (`claim_amount_inr`), product SKUs, and service partner IDs of prior claims.
  3. The regulatory policy date (`POLICY_CHANGE_DATE = 2026-05-01`) and auto-approval threshold (`AUTO_APPROVE_THRESHOLD_INR = 2000.0`).
- Operations **does NOT know**:
  - The ground-truth fraud label (`is_fraud`) of prior claims, because investigations take 14–90 days to close.
- **Rule of Truth:** No historical partner fraud-rate or label-derived features were built. All partner monitoring signals are strictly derived from **unsupervised claim volume, frequency, and small-claim proportions**.

### 2.2 Point-in-Time Partner Risk Signals Formulations
For claim $i$ from partner $P$ at timestamp $t_i$:
1. **Partner Trailing Volume ($[t_i - 30\text{d}, t_i)$):**
   $$\text{partner\_vol\_30d} = \sum_{j \in \text{Ledger}(P), t_j < t_i} \mathbb{I}(t_j \ge t_i - 30\text{d})$$
2. **Partner Trailing Short-Window Volume ($[t_i - 7\text{d}, t_i)$):**
   $$\text{partner\_vol\_7d} = \sum_{j \in \text{Ledger}(P), t_j < t_i} \mathbb{I}(t_j \ge t_i - 7\text{d})$$
3. **Partner Claim Velocity Ratio:**
   $$\text{partner\_velocity\_ratio} = \frac{\text{partner\_vol\_7d} / 7.0}{(\text{partner\_vol\_30d} + 1.0) / 30.0}$$
4. **Partner Small-Claim Volume & Ratio ($< ₹2,000$):**
   $$\text{partner\_small\_vol\_30d} = \sum_{j \in \text{Ledger}(P), t_j < t_i} \mathbb{I}(t_j \ge t_i - 30\text{d} \land \text{Amount}_j < 2000)$$
   $$\text{partner\_small\_ratio\_30d} = \frac{\text{partner\_small\_vol\_30d}}{\text{partner\_vol\_30d} + 1\text{e-5}}$$
5. **Partner Policy Surge Signal:**
   $$\text{partner\_policy\_surge} = \begin{cases} \text{partner\_small\_ratio\_30d} \times \max(1.0, \text{partner\_velocity\_ratio}) \times (1 + \ln(1 + \text{partner\_vol\_30d})) & \text{if } t_i \ge \text{2026-05-01} \\ \max(1.0, \text{partner\_velocity\_ratio}) \times \ln(1 + \text{partner\_vol\_30d}) & \text{if } t_i < \text{2026-05-01} \end{cases}$$

---

## 3. Month-by-Month Operational Backtest Results

### 3.1 April 2026 (Pre-Policy Regime: All claims eligible for desk review)
*Volume: 734 claims (709 labeled, 25 unresolved) | Confirmed Fraud: 9 (1.27%) | Small Fraud (< ₹2k): 5/9*

| Strategy | Mechanism | Claims Flagged | Partners Flagged | TP (Fraud) | FP (Clean) | Unresolved | Precision (%) | Recall (%) | Loss Avoided | Total Cost | Net Benefit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Selection** | Desk Queue (40) | 40 | 38 | 1 | 37 | 2 | 2.63% | 11.11% | ₹1,136.00 | ₹23,680.00 | **-₹22,544.00** |
| **Partner Volume Baseline** | Desk Queue (40) | 40 | 12 | 0 | 38 | 2 | 0.00% | 0.00% | ₹0.00 | ₹24,320.00 | **-₹24,320.00** |
| **Pure Partner Risk Signal**| Desk Queue (40) | 40 | 12 | 0 | 39 | 1 | 0.00% | 0.00% | ₹0.00 | ₹24,960.00 | **-₹24,960.00** |
| **V1 LightGBM Baseline** | Desk Queue (40) | 40 | 36 | 3 | 35 | 2 | 7.89% | 33.33% | ₹25,184.00 | ₹22,400.00 | **+₹2,784.00** |
| **V2 LightGBM Candidate** | Desk Queue (40) | 40 | 36 | **4** | 35 | 1 | **10.26%** | **44.44%** | **₹26,383.00** | ₹22,400.00 | **+₹3,983.00** |
| **Hybrid (V2 + Partner)** | Desk Queue (40) | 40 | 36 | 3 | 36 | 1 | 7.69% | 33.33% | ₹25,184.00 | ₹23,040.00 | **+₹2,144.00** |
| **Partner Audit (Top 2)** | Partner Audit (2) | 10 | 2 | 0 | 10 | 0 | 0.00% | 0.00% | ₹0.00 | ₹7,600.00 | **-₹7,600.00** |
| **Partner Audit (Top 3)** | Partner Audit (3) | 13 | 3 | 0 | 13 | 0 | 0.00% | 0.00% | ₹0.00 | ₹10,880.00 | **-₹10,880.00** |

*April Analysis:* In April, fraud was scattered across 8 different low-volume partners. Claim-level ML models (V2 LightGBM) excelled, catching high-value claims (₹7,000–₹10,000) and generating **+₹3,983 profit**. Partner volume monitoring completely failed, generating zero catches.

---

### 3.2 May 2026 (Post-Policy Regime: Auto-Approval < ₹2,000)
*Volume: 734 claims (709 labeled, 25 unresolved) | Confirmed Fraud: 14 (1.97%) | Small Fraud (< ₹2k): 14/14 (100%)*

| Strategy | Mechanism | Claims Flagged | Partners Flagged | TP (Fraud) | FP (Clean) | Unresolved | Precision (%) | Recall (%) | Loss Avoided | Total Cost | Net Benefit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Selection** | Desk Queue (40) | 40 | 39 | 0 | 39 | 1 | 0.00% | 0.00% | ₹0.00 | ₹24,960.00 | **-₹24,960.00** |
| **V1 LightGBM Baseline** | Desk Queue (40) | 40 | 39 | 1 | 38 | 1 | 2.56% | 7.14% | ₹1,995.00 | ₹24,320.00 | **-₹22,325.00** |
| **V2 LightGBM Candidate** | Desk Queue (40) | 40 | 38 | 1 | 38 | 1 | 2.56% | 7.14% | ₹1,995.00 | ₹24,320.00 | **-₹22,325.00** |
| **Hybrid (V2 + Partner)** | Desk Queue (40) | 40 | 36 | 2 | 36 | 2 | 5.26% | 14.29% | ₹3,982.00 | ₹23,040.00 | **-₹19,058.00** |
| **Pure Partner Risk Signal**| Desk Queue (40) | 40 | 11 | 4 | 35 | 1 | 10.26% | 28.57% | ₹6,001.00 | ₹22,400.00 | **-₹16,399.00** |
| **Partner Volume Baseline** | Desk Queue (40) | 40 | 10 | 5 | 34 | 1 | 12.82% | 35.71% | ₹7,996.00 | ₹21,760.00 | **-₹13,764.00** |
| **Partner Audit (Top 5)** | Partner Audit (5) | 33 | 5 | 6 | 27 | 0 | 18.18% | 42.86% | ₹8,677.00 | ₹19,520.00 | **-₹10,843.00** |
| **Partner Audit (Top 3)** | Partner Audit (3) | 20 | 3 | 4 | 16 | 0 | 20.00% | 28.57% | ₹6,001.00 | ₹11,660.00 | **-₹5,659.00** |
| **Partner Audit (Top 2)** | Partner Audit (2) | 13 | 2 | **4** | **9** | 0 | **30.77%** | **28.57%** | **₹6,001.00** | **₹7,340.00** | **-₹1,339.00** |

*May Analysis:* 100% of fraud shifted to sub-₹2,000 claims. While claim-level models caught only 1 fraud (-₹22,325 loss), Partner Audit (Top 2) captured 4 frauds (all from `SP3160`), achieving 30.77% precision and cutting losses to -₹1,339.

---

### 3.3 June 2026 (Post-Policy Regime: Auto-Approval < ₹2,000)
*Volume: 730 claims (713 labeled, 17 unresolved) | Confirmed Fraud: 22 (3.09%) | Small Fraud (< ₹2k): 21/22 (95.5%)*

| Strategy | Mechanism | Claims Flagged | Partners Flagged | TP (Fraud) | FP (Clean) | Unresolved | Precision (%) | Recall (%) | Loss Avoided | Total Cost | Net Benefit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Selection** | Desk Queue (40) | 40 | 37 | 1 | 39 | 0 | 2.50% | 4.55% | ₹1,526.00 | ₹24,960.00 | **-₹23,434.00** |
| **Hybrid (V2 + Partner)** | Desk Queue (40) | 40 | 30 | 1 | 38 | 1 | 2.56% | 4.55% | ₹1,978.00 | ₹24,320.00 | **-₹22,342.00** |
| **V1 LightGBM Baseline** | Desk Queue (40) | 40 | 32 | 1 | 37 | 2 | 2.63% | 4.55% | ₹1,978.00 | ₹23,680.00 | **-₹21,702.00** |
| **V2 LightGBM Candidate** | Desk Queue (40) | 40 | 30 | 1 | 37 | 2 | 2.63% | 4.55% | ₹1,978.00 | ₹23,680.00 | **-₹21,702.00** |
| **Pure Partner Risk Signal**| Desk Queue (40) | 40 | 15 | 11 | 26 | 3 | 29.73% | 50.00% | ₹14,556.00 | ₹16,640.00 | **-₹2,084.00** |
| **Partner Volume Baseline** | Desk Queue (40) | 40 | 10 | 11 | 26 | 3 | 29.73% | 50.00% | ₹15,757.00 | ₹16,640.00 | **-₹883.00** |
| **Partner Audit (Top 5)** | Partner Audit (5) | 28 | 5 | 10 | 16 | 2 | 38.46% | 45.45% | ₹13,784.00 | ₹16,660.00 | **-₹2,876.00** |
| **Partner Audit (Top 3)** | Partner Audit (3) | 17 | 3 | 10 | 6 | 1 | 62.50% | 45.45% | ₹13,784.00 | ₹9,060.00 | **+₹4,724.00** |
| **Partner Audit (Top 2)** | Partner Audit (2) | 14 | 2 | **10** | **3** | 1 | **76.92%** | **45.45%** | **₹13,784.00** | **₹5,780.00** | **+₹8,004.00** |

*June Analysis:* In June, rogue dealers accelerated claim submissions. The **Partner Audit Policy (Top 2 Partners: `SP3160` and `SP3232`) achieved 76.92% precision (10 frauds, 3 false alarms)**. Because audit costs were consolidated ($2 \times ₹2,500 + 3 \times ₹260 = ₹5,780$), it yielded **+₹8,004.00 net operating recovery**, vastly outperforming all claim-level models.

---

## 4. Concentration, Sample Size & Stability Analysis

### 4.1 Two Partners Account for 100% of Caught Fraud
Inspection of the individual claims caught by the Partner Audit Policy reveals extreme concentration:

```
May 2026 Audit Catches (4 frauds):
  - Claim WC710015 (SP3160, Rs. 1,603)
  - Claim WC710334 (SP3160, Rs. 984)
  - Claim WC710504 (SP3160, Rs. 1,427)
  - Claim WC710597 (SP3160, Rs. 1,987)
  -> 100% from Partner SP3160.

June 2026 Audit Catches (10 frauds):
  - SP3160: 6 frauds (WC710662, WC710698, WC710875, WC710876, WC710993, WC711200) -> Rs. 7,799
  - SP3232: 4 frauds (WC710641, WC710984, WC711041, WC711164)                      -> Rs. 5,985
  -> 100% from Partners SP3160 and SP3232.
```

### 4.2 Stability Risk Assessment
- **Sample Size Fragility:** Out of 365 active service partners, exactly **2 partners** (`SP3160` and `SP3232`) drove the entire positive recovery in June.
- **Vulnerability to Adversarial Adaptation:** If rogue partners learn that submitting $>5$ claims per month triggers an audit, they can easily bypass this filter by dispersing claims across multiple shop fronts or keeping volume below 4 claims/month.
- **Regime Fragility:** In pre-policy conditions (April), the partner audit policy had **0% precision** and lost ₹7,600.

---

## 5. Cost-Benefit Arithmetic & Explicit Policy Assumptions

### 5.1 Claim Desk Review Queue Model (Ops Policy v4.1)
- Fixed review budget: **40 claims/month** (120 reviews/quarter).
- Avoided Loss: Actual sum of `claim_amount_inr` for confirmed frauds in the top 40.
- FP Cost: Exactly **₹640.00** per non-fraudulent claim held (₹380 goodwill loss + ₹260 customer service contact overhead).
- Unresolved claims: Excluded from loss avoided and false-positive cost (held without charge).

$$\text{Net Benefit}_{\text{Desk}} = \sum_{i \in \text{TP}} \text{Amount}_i - (N_{\text{FP}} \times ₹640)$$

### 5.2 Partner Dealer Audit Model (Separate Capacity Assumption)
- Fixed audit budget: **$K$ partners audited per month** (e.g. $K=2$ or $K=3$).
- On-site Field Audit Fee: **₹2,500.00** per audited partner (auditor inspection fee).
- Delay Friction per Clean Claim: **₹260.00** per legitimate claim held during dealer audit. Goodwill loss is avoided because customer is told the dealer is undergoing an administrative quality review.
- Avoided Loss: Actual sum of all fraudulent claims submitted by the audited partners.

$$\text{Net Benefit}_{\text{Audit}} = \sum_{i \in \text{TP}} \text{Amount}_i - (K \times ₹2,500) - (N_{\text{FP}} \times ₹260)$$

---

## 6. Strategic Conclusions & Next Steps

### 6.1 Core Verdict
1. **Partner monitoring adds immense operational value, but ONLY when deployed as dealer audits, NOT as a single-claim desk queue.**
   - Feeding partner signals into a 40-claim desk review queue still loses ₹38,967 over Q2 because ₹640/claim friction wipes out the low value of small claims.
   - Deploying partner signals as **Partner Dealer Audits (auditing top 2 burst partners)** turns an unrecoverable deficit into a **+₹6,665 profit** across May and June.
2. **Neither ML models nor partner monitoring is a standalone panacea:**
   - ML models (V2 LightGBM) are indispensable in decentralized, high-value regimes (April).
   - Partner monitoring is indispensable in centralized small-claim burst regimes (May & June).

### 6.2 Recommended Deployment Architecture (Ops Policy v4.2)
Implement a **Coordinated Dual-Track Operations Architecture**:
- **Track 1: High-Value Claim Desk Queue (ML Driven):**
  - Allocate 25 desk reviews per month strictly to claims $\ge ₹2,000$ ranked by the V2 LightGBM model. This captures high-value fraud profitably without being diluted by small claims.
- **Track 2: Dealer Audit Trigger (Telemetry Driven):**
  - Allocate a budget of **2 dealer field audits per month**.
  - Automatically place claims on payout hold and dispatch an audit when a partner's trailing 30-day small-claim volume exceeds 5 claims with a velocity ratio $>1.5$.
  - This directly traps collusive dealers like `SP3160` and `SP3232` at minimal overhead ($2 \times ₹2,500 = ₹5,000$).

---

## 7. Deliverables & Artifact Verification

- Report: [`outputs/partner_backtest/partner_backtest_report.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/partner_backtest_report.md)
- Monthly Results Data: [`outputs/partner_backtest/monthly_results.csv`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/monthly_results.csv)
- Claims & Partner Signals Data: [`outputs/partner_backtest/partner_risk_signals.csv`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/partner_risk_signals.csv)
- Limitations Analysis: [`outputs/partner_backtest/limitations.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/limitations.md)
- Backtest Test Suite: [`tests/test_partner_backtest.py`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/tests/test_partner_backtest.py) (3 passed; 12 total test suite passed)
