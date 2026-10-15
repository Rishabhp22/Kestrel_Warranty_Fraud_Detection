# Independent Verification Audit: Kestrel Partner-Level Risk Monitoring Backtest

**Audit Date:** October 14, 2026  
**Auditor:** Independent Technical Auditor & Fraud Risk Specialist  
**Audit Scope:** Verification of Source Data, Monthly Reconciliation, Model Metric Divergence, Policy Cost Traceability, & Partner Sensitivity  
**Production Protection Status:** Strictly read-only audit. No production models, candidate models, feature definitions, `outputs/predictions.csv`, or existing backtest results were altered.

---

## 1. Executive Audit Summary & Verification Matrix

Every check required by the audit specification was executed independently against the raw datasets (`train.csv`, `partners.csv`, `ops-policy.pdf`), previous evaluation reports, and the backtest output CSVs (`monthly_results.csv`, `partner_risk_signals.csv`).

| Audit Item | Verification Requirement | Auditor Finding & Reconciliation Status | Verification Result |
| :--- | :--- | :--- | :---: |
| **Check 1** | Metric Recomputation from Source | All monthly/quarterly metrics recomputed from raw transactions & predictions; match CSVs with 0.00 error | **PASS (Exact)** |
| **Check 2** | Reconciliation vs Previous Reports | Reconciled V1/V2 metrics between `v2_independent_verification.md` and `partner_backtest`; exact divergence caused by unresolved claim presence | **PASS (Identified & Reconciled)** |
| **Check 3** | Cost Parameter Traceability | Traced ₹380, ₹260, ₹640, ₹2,500 to `ops-policy.pdf`; ₹380 & ₹260 are documented; ₹640 is an operational combination; ₹2,500 is an unverified assumption | **PASS (Clarified & Documented)** |
| **Check 4** | Unresolved Claims Treatment | Verified that 67 unlabelled claims in Q2 are excluded from confirmed TP/FP counts and tracked in separate telemetry | **PASS (Exact)** |
| **Check 5** | Separate Capacity Assumptions | Confirmed that Claim Desk Reviews (40 claims/mo) and Partner Audits (2, 3, 5 partners/mo) use distinct capacity constraints | **PASS (Verified)** |
| **Check 6** | SP3160 & SP3232 Sensitivity Test | Recalculated results under 4 scenarios; verified that Top 2 Partner Audit benefit depends 100% on SP3160 and SP3232 | **PASS (100% Fragility Proven)** |
| **Check 7** | Exact CSV Reconciliation | Reconciled all 24 strategy-month rows in `monthly_results.csv` against `partner_risk_signals.csv` with zero discrepancy | **PASS (100% Reconciled)** |

---

## 2. Metric Reconciliation Against Previous Evaluation Reports

A vital audit objective is resolving why the V1 and V2 LightGBM numbers in `outputs/partner_backtest/monthly_results.csv` differ slightly from those in `outputs/v2_independent_verification.md` and `outputs/model_comparison_v2.md`.

### 2.1 Side-by-Side Comparison of V1 and V2 Metrics

| Model & Split | Evaluation Document | Validation Cohort | Confirmed Fraud (TP) | False Alarms (FP) | Unresolved Claims in Queue | Loss Avoided | Total Cost | Net Operating Benefit (Q2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 LightGBM** | `v2_independent_verification.md` | Labeled Only ($N=2,131$) | 5 | 115 | 0 (Pre-filtered) | ₹29,157.00 | ₹73,600.00 | **-₹44,443.00** |
| **V1 LightGBM** | `monthly_results.csv` (Backtest) | Full Stream ($N=2,198$) | 5 | 110 | 5 | ₹29,157.00 | ₹70,400.00 | **-₹41,243.00** |
| **V2 LightGBM** | `v2_independent_verification.md` | Labeled Only ($N=2,131$) | 6 | 114 | 0 (Pre-filtered) | ₹30,356.00 | ₹72,960.00 | **-₹42,604.00** |
| **V2 LightGBM** | `monthly_results.csv` (Backtest) | Full Stream ($N=2,198$) | 6 | 110 | 4 | ₹30,356.00 | ₹70,400.00 | **-₹40,044.00** |

### 2.2 Root Cause Analysis of Metric Divergence
1. **Validation Population Difference:**
   - In [`outputs/v2_independent_verification.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/v2_independent_verification.md), the evaluation was run exclusively on **pre-filtered labeled claims** ($N = 2,131$). In this closed set, every claim in the top 40 had a confirmed ground-truth label ($y \in \{0, 1\}$). Consequently, any non-fraud claim was a confirmed False Positive ($40 - \text{TP} = \text{FP}$), yielding 114 FPs for V2 ($114 \times ₹640 = ₹72,960$) and 115 FPs for V1 ($115 \times ₹640 = ₹73,600$).
   - In [`outputs/partner_backtest/`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/), the backtest simulated the **true operational stream** ($N = 2,198$), which includes incoming claims with unconfirmed/blank outcomes (the 67 unresolved claims).
2. **Impact on Top-40 Queue Selection:**
   - For **V2 LightGBM**: The model selected **4 unresolved claims** into the top 40 across Q2 (1 in April, 1 in May, 2 in June).
   - For **V1 LightGBM**: The model selected **5 unresolved claims** into the top 40 across Q2 (2 in April, 1 in May, 2 in June).
3. **Financial Accounting of Unresolved Claims:**
   - Under audit standards, an unresolved claim cannot be legally penalized as a confirmed false alarm because the investigation outcome is undecided.
   - Therefore, in `monthly_results.csv`, confirmed FP count dropped from 114 to 110 for V2, reducing FP costs by exactly $4 \times ₹640 = ₹2,560$ (from ₹72,960 to ₹70,400), which improved the net benefit from **-₹42,604** to **-₹40,044**.
   - For V1, confirmed FP count dropped from 115 to 110, reducing FP costs by exactly $5 \times ₹640 = ₹3,200$ (from ₹73,600 to ₹70,400), improving net benefit from **-₹44,443** to **-₹41,243**.
- **Auditor Conclusion:** The divergence is **100% mathematically accounted for**. The true loss-avoidance figures (₹29,157 for V1; ₹30,356 for V2) and true positive fraud counts (5 for V1; 6 for V2) are **identical in both reports**.

---

## 3. Policy & Cost Parameter Traceability Audit

Inspection of Kestrel's official policy documentation ([`ops-policy.pdf`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/ops-policy.pdf)) revealed the exact evidentiary status of each financial parameter:

```
From ops-policy.pdf (§4 Costs used for decisions & §5 Claim approval):
"A genuine claim held for review delays the customer's repair; Kestrel issues goodwill worth Rs 380 on average when that happens.
A fraudulent claim paid out costs the full claim amount. Blended cost of a service contact: Rs 260.
The claims investigation desk can review at most 40 claims per month."
```

### 3.1 Traceability Assessment Table

| Financial Parameter | Model Value | Documented in `ops-policy.pdf`? | Evidentiary Status | Audit Classification |
| :--- | :---: | :---: | :--- | :---: |
| **Customer Goodwill Compensation** | **₹380.00** | **Yes (§4)** | Text explicitly states average goodwill voucher is Rs 380 | **Established Policy Fact** |
| **Service Contact Cost** | **₹260.00** | **Yes (§4)** | Text explicitly states blended cost of service contact is Rs 260 | **Established Policy Fact** |
| **Loss Avoided per Fraud** | **Full Claim Amount** | **Yes (§4)** | Text states fraudulent claim paid out costs full claim amount | **Established Policy Fact** |
| **Desk Monthly Review Capacity** | **40 claims/month** | **Yes (§5)** | Text states desk can review at most 40 claims per month | **Established Policy Fact** |
| **Small-Claim Auto-Approval Threshold** | **< ₹2,000.00** | **Yes (§5)** | Auto-approval threshold effective 1 May 2026 | **Established Policy Fact** |
| **Combined False-Alarm Friction** | **₹640.00** | **Partial (§4)** | Additive combination ($₹380 + ₹260 = ₹640$). Policy lists items separately; assumes 100% of held claims incur both costs | **Operational Modeling Assumption** |
| **Partner Dealer Audit Cost** | **₹2,500.00** | **NO** | Zero mention of dealer audit or ₹2,500 fee in policy documentation | **Hypothetical Simulation Assumption** |
| **Clean-Claim Delay Friction under Audit** | **₹260.00** | **NO** | Assumes goodwill (₹380) is waived if audit is framed as quality check | **Hypothetical Simulation Assumption** |

### 3.2 Auditor Guidance for Business Memo
The business memo must **not** present the ₹2,500 dealer audit cost or the ₹640 combined false-positive cost as undisputed operational facts. They must be explicitly labeled as **operational modeling assumptions based on policy guidance**.

---

## 4. Treatment of Unresolved Claims (67 Claims)

1. **Population Breakdown:** There are exactly **67 unresolved claims** in the Q2 validation window:
   - April 2026: 25 unresolved claims
   - May 2026: 25 unresolved claims
   - June 2026: 17 unresolved claims
2. **Supervised Target Handling:** Confirmed that in `partner_risk_signals.csv`, all 67 unresolved claims have `fraud_label = NaN` and `is_unresolved = True`. They were **never converted to non-fraud (`0.0`)**.
3. **Queue Evaluation Treatment:**
   - In all queue evaluations, unresolved claims are reported in a distinct column (`unresolved_flagged`).
   - They are excluded from the numerator of fraud catch (`tp_fraud_caught`) and the numerator of false alarms (`fp_false_alarms`).
   - Precision is calculated conservatively over verified claims: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$.

---

## 5. Capacity Separation & Claim Counting Consistency

- **Claim Review Queue:** Constrained to exactly **40 claims per month** (120 quarterly), reflecting §5 of `ops-policy.pdf`.
- **Partner Dealer Audit Policy:** Assumes a **separate operational capacity** of $K$ dealer audits per month ($K \in \{2, 3, 5\}$).
- **Claim Counting Consistency:**
  - When partner $P$ is audited in month $M$, **every claim submitted by partner $P$ in month $M$ is placed on administrative hold and physically verified**.
  - Verified: In June 2026, under the Top 2 Partner Audit strategy, the two audited partners (`SP3160` and `SP3232`) submitted exactly 14 claims (7 from SP3160, 7 from SP3232). All 14 claims were reviewed: 10 were confirmed fraud, 3 were confirmed clean, and 1 was unresolved. The arithmetic matches the raw data exactly.

---

## 6. Sensitivity Analysis: Does Partner-Audit Benefit Depend on SP3160 and SP3232?

To evaluate the robustness of the Partner Dealer Audit Policy (Top 2 Partners/Month, ₹2,500 audit fee, ₹260 clean-claim friction), four independent sensitivity scenarios were executed across Q2:

### 6.1 Four-Scenario Sensitivity Results Matrix

| Scenario Evaluated | Audited Partners in Q2 | Total Claims Audited | Confirmed Fraud Caught (TP) | False Alarms (FP) | Avoided Fraud Loss | Total Operational Cost | Net Operating Benefit (Q2) | June Net Benefit (Post-Policy) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1: All Partners (Reported Baseline)** | SP3232, SP3220, SP3015, SP3160 | 37 | **14** | **22** | **₹19,785.00** | **₹20,720.00** | **-₹935.00** | **+₹8,004.00 (Profitable)** |
| **Scenario 2: Excluding SP3160 Only** | SP3220, SP3232, SP3015, SP3141, SP3060 | 34 | **4** | **30** | **₹6,316.00** | **₹22,800.00** | **-₹16,484.00** | **-₹244.00 (Deficit)** |
| **Scenario 3: Excluding SP3232 Only** | SP3220, SP3140, SP3015, SP3160, SP3060 | 32 | **10** | **21** | **₹13,469.00** | **₹20,460.00** | **-₹6,991.00** | **+₹1,688.00 (Modest)** |
| **Scenario 4: Excluding BOTH SP3160 & SP3232** | SP3220, SP3330, SP3015, SP3141, SP3060, SP3150 | 34 | **0** | **34** | **₹0.00** | **₹23,840.00** | **-₹23,840.00** | **-₹7,340.00 (Severe Loss)** |

### 6.2 Key Auditor Findings on Fragility
1. **100% Dependency on Two Outlets:**
   - In Scenario 4 (excluding both `SP3160` and `SP3232`), the Top 2 Partner Audit strategy caught **ZERO fraud across the entire quarter**.
   - Every single claim audited was clean (34 false alarms out of 34 claims), producing an operating deficit of **-₹23,840.00**.
2. **`SP3160` is the Primary Anchor:**
   - `SP3160` alone provided 10 of the 14 caught frauds (71.4% of all fraud caught by dealer audit).
   - Removing `SP3160` causes June net benefit to flip from **+₹8,004** profit into an operating deficit (**-₹244**).
3. **Operational Conclusion:**
   - The reported commercial viability of partner audits is **not a generalized network phenomenon**. It is a direct response to two specific rogue outlets that engaged in uninhibited small-claim burst fraud.

---

## 7. Exact Reconciliation of CSV Totals and Source Data

Every monthly and quarterly total in `outputs/partner_backtest/monthly_results.csv` was programmatically verified against `outputs/partner_backtest/partner_risk_signals.csv`:

$$\text{Reconciliation Equations Tested Across All 24 Rows:}$$
1. $\text{claims\_flagged} = \text{tp\_fraud\_caught} + \text{fp\_false\_alarms} + \text{unresolved\_flagged}$ $\implies$ **100% Exact Match**
2. $\text{fn\_fraud\_missed} = \text{total\_month\_fraud} - \text{tp\_fraud\_caught}$ $\implies$ **100% Exact Match**
3. $\text{net\_benefit\_inr} = \text{loss\_avoided\_inr} - \text{total\_cost\_inr}$ $\implies$ **100% Exact Match (0.00 rupee error)**
4. In Claim Desk Queues: $\text{total\_cost\_inr} = \text{fp\_false\_alarms} \times 640.00$ $\implies$ **100% Exact Match**
5. In Partner Audits: $\text{total\_cost\_inr} = (K \times 2500.00) + (\text{fp\_false\_alarms} \times 260.00)$ $\implies$ **100% Exact Match**

---

## 8. Required Corrections for the Business Memo

Before submitting the final business memo, the following factual corrections and caveats must be incorporated:

1. **Do NOT Claim Positive Overall Quarterly Net Benefit for Any Claim Queue:**
   - Every single claim desk queue strategy under Ops Policy v4.1 (Random, V1, V2, Pure Partner Signal, Hybrid) generated a substantial operating loss over Q2 (-₹39,000 to -₹71,000).
   - Negative net benefit must be reported honestly as an **operating loss**, not obscured behind precision percentages or gross loss-avoidance figures.
2. **Explicitly Qualify the Partner Audit Strategy:**
   - The business memo must clarify that the positive net recovery in June (+₹8,004) and post-policy May–June (+₹6,665) was achieved under a **separate operational capacity model (Dealer Audits)**, not the 40-claim desk queue.
   - It must explicitly state that this performance is **100% dependent on two partners (`SP3160` and `SP3232`)** and that excluding them reduces fraud catch to zero.
3. **Label Cost Figures as Assumptions:**
   - The memo must distinguish between policy-documented figures (₹380 goodwill, ₹260 service contact) and simulation assumptions (₹640 combined friction, ₹2,500 dealer audit fee).
4. **Propose Dual-Track Operations (Ops Policy v4.2):**
   - The business memo should recommend allocating 25 desk reviews to claims $\ge ₹2,000$ scored by V2 LightGBM, combined with a 2-partner monthly audit trigger for dealer bursts $< ₹2,000$.

---

## 9. Final Auditor Verdict

### **VERDICT: AUDIT PASSED WITH DOCUMENTED FRAGILITY CAVEATS**

- **Mathematical Integrity:** All reported figures in `monthly_results.csv` and `partner_risk_signals.csv` are 100% reproducible, mathematically reconciled, and free of chronological label leakage.
- **Evidentiary Integrity:** Cost parameters and partner dependencies are fully unmasked and documented.
- **Readiness:** Candidate model pipelines and backtest findings are verified and ready for business memo integration. No production artifacts were modified during this audit.
