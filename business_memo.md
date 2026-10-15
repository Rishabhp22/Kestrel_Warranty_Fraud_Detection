# Business Memorandum

**To:** Ritu Deshpande, Head of D2C Operations; Farhan Sheikh, Finance Controller; Meenal Joshi, Service Desk Manager  
**From:** Analytics & Engineering — Warranty Fraud Detection Project  
**Date:** October 2026  
**Re:** Kestrel Warranty Fraud Risk Model — Operational Audit, Monthly Reality, and Strategic Recommendations  
**Classification:** Internal — Operations Leadership  

---

## 1. Purpose

This memorandum provides an updated, evidence-grounded operational assessment of the machine learning fraud risk scoring system developed for Kestrel Home Appliances. It replaces earlier preliminary estimates with verified monthly evaluations on out-of-time Q2 2026 data, analyzes the structural impact of the May 1, 2026 auto-approval policy change, and proposes a controlled operational pilot to address small-claim fraud clustering.

---

## 2. Operational Reality & The Fixed Desk Capacity Constraint

Kestrel's claims investigation desk operates under a strict policy ceiling: a **maximum capacity of 40 claims per month** can be reviewed manually (Ops Policy v4.1, §5). Reviewing 40 claims each month represents the upper operational limit, not a requirement to review low-confidence claims. Over the three-month Q2 validation window (April, May, June 2026), evaluating a baseline queue that fills all 40 slots reviewed 120 claims out of 2,198 submitted claims.

- **Baseline Fraud Prevalence:** There are 45 confirmed fraud cases among the 2,131 labeled validation claims (2.11% base rate).
- **Cost Parameters (Ops Policy v4.1, §4):**
  - **Goodwill Delay Compensation:** ₹380 on average issued when holding a genuine customer's claim delays repair.
  - **Service Contact Overhead:** ₹260 blended cost per customer service contact.
  - **Combined False-Alarm Friction Cost:** **₹640 per false positive** is an operational modeling assumption that couples both costs under the assumption that 100% of delayed genuine customers contact customer support.
  - **Avoided Fraud Loss:** Recovers the full claim amount (`claim_amount_inr`).

---

## 3. Verified Out-of-Time Monthly Performance (V2 LightGBM Candidate)

Earlier draft memos reported an aggregate benefit of `+Rs. 4,122` by pooling 3 months of validation data into an artificial single 40-claim review. **That figure is obsolete and withdrawn.** 

When evaluated strictly month-by-month against an unconstrained queue that fills the **maximum capacity of 40 claims per month**, the V2 LightGBM model produced a **quarterly net operating loss of -₹40,044.00** (or -₹42,604.00 if unlabelled claims are treated as confirmed clean).

### Verified Month-by-Month Operational Triage Matrix (V2 LightGBM Candidate — Top 40 Queue)
*(Source: `outputs/partner_backtest/monthly_results.csv`, evaluated on the 2,198 operational stream)*

| Operational Window | Regime & Policy Status | Monthly Desk Reviews | Confirmed Fraud Caught (TP) | False Alarms (FP) | Unresolved Claims Held | Queue Precision (%) | Fraud Recall (%) | Gross Loss Avoided (INR) | False-Alarm Cost (₹640/FP) | Net Operating Benefit (INR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **April 2026** | Pre-Policy (Inspection Required) | 40 | **4** | 35 | 1 | 10.26% | 44.44% (4/9) | ₹26,383.00 | ₹22,400.00 | **+₹3,983.00 (Profitable)** |
| **May 2026** | Post-Policy (Auto-Approval < ₹2k) | 40 | **1** | 38 | 1 | 2.56% | 7.14% (1/14) | ₹1,995.00 | ₹24,320.00 | **-₹22,325.00 (Loss)** |
| **June 2026** | Post-Policy (Auto-Approval < ₹2k) | 40 | **1** | 37 | 2 | 2.63% | 4.55% (1/22) | ₹1,978.00 | ₹23,680.00 | **-₹21,702.00 (Loss)** |
| **Q2 2026 Total** | **120 Reviews over 3 Months** | **120** | **6** | **110** | **4** | **5.17%** | **13.33% (6/45)**| **₹30,356.00** | **₹70,400.00** | **-₹40,044.00 (Net Operating Loss)** |

*(Note: Precision is calculated on verified claims: $\text{TP}/(\text{TP}+\text{FP})$. The 4 unresolved claims are tracked separately).*

### Comparison Against V1 Baseline
- **V1 LightGBM Baseline:** Caught 5 frauds (11.11% recall), 110 false alarms, ₹29,157 saved, ₹70,400 cost $\to$ **Net Loss: -₹41,243.00**.
- **V2 LightGBM Candidate:** Caught 6 frauds (13.33% recall), 110 false alarms, ₹30,356 saved, ₹70,400 cost $\to$ **Net Loss: -₹40,044.00** (+₹1,199 loss avoided; +₹1,199 net improvement over baseline).
- **Core Business Takeaway:** While V2 outperforms V1 on identical claims, **running an unconstrained queue filling all 40 desk slots post-May produces an operating loss under both models.** Negative net benefit is an operating loss, not a positive business outcome.

The test dataset (`test_unlabelled.csv`) contains 2,252 claims with unobserved ground-truth labels. While V2 caught 6 fraud cases versus 5 for V1 on historical Q2 validation under the evaluated review-capacity setup, performance on the hidden test set remains unverified. Historical validation results are not hidden-test results.

---

## 4. Why Desk Review Loses Money Post-May: The Policy Loophole

The root cause of the May and June operating deficits is structural:

1. **The May 1, 2026 Inspection Policy Change:**  
   Under Ops Policy v4.1 (§5), all claims under ₹2,000 submitted on or after May 1, 2026 are **auto-approved without partner inspection** to accelerate customer turnaround.
2. **Fraud Vector Inversion:**  
   Rogue service partners immediately exploited this threshold:
   - In **April** (pre-policy), fraud was distributed across higher-value claims (mean fraud amount: ₹7,869). Model triage caught 4 frauds (WC709780, WC709418, WC709403, and WC709172), yielding **+₹3,983 net profit**.
   - In **May and June**, **35 out of 36 confirmed frauds (97.2%) shifted into sub-₹2,000 small claims** (average fraud amount: ₹1,480).
3. **The ₹640 vs. ₹1,500 Asymmetry:**  
   When a caught fraud recovers only ~₹1,500, but each false alarm costs ₹640:
   $$\text{Breakeven Precision} = \frac{₹640}{₹1,500} \approx 42.7\%$$
   In May and June, desk queue precision was only **2.56%** (1 hit, 38 false alarms). Spending ₹24,320 in false-alarm friction to recover ₹1,995 guarantees a ~₹22,000 loss each month. Claim-level desk review cannot profitably police sub-₹2,000 claims individually.

---

## 5. Unresolved Claims Disclosure (67 Claims)

In the Q2 validation dataset, exactly **67 claims** (25 in April, 25 in May, 17 in June) have blank `is_fraud` values. 
- In accordance with Tanmay Kulkarni's documentation, these represent claims whose physical investigations were still undecided at export.
- **Strict Accounting Standard:** These 67 claims are included in historical partner volume telemetry, but are **strictly excluded from supervised training labels and confirmed fraud/non-fraud counts**. They must **never be treated as confirmed non-fraud (0)**. 
- In Q2 desk triage, 4 unresolved claims were selected into V2's top 40. They were held without incurring confirmed false-alarm friction, but pending final investigation outcomes, actual desk recovery may vary.

---

## 6. Offline Partner-Level Risk Monitoring Backtest & Critical Caveats

To test whether monitoring partner behavioral surges adds operational value beyond claim-level models, an offline, point-in-time backtest was executed (`outputs/partner_backtest/partner_backtest_report.md`).

### 6.1 Partner Dealer Audit Findings
Simulating a **Partner Dealer Audit Strategy** (where operations audits high-burst outlets rather than individual claims):
- Auditing the **Top 2 burst partners per month** consolidated investigations: in June 2026, it caught **10 small frauds with only 3 false alarms**, recovering ₹13,784 for ₹5,780 in audit costs, yielding **+₹8,004.00 net benefit**.
- Across Q2, it caught 14 small frauds with 22 false alarms, cutting total costs from ₹70,400 to ₹20,720 and reducing quarterly loss to **-₹935.00** (nearly break-even).

### 6.2 Mandatory Operational Caveats & Fragility Disclosures
1. **Simulation Cost Assumptions (Not Policy Facts):**  
   The assumed ₹2,500 dealer audit fee and ₹260 clean-claim holding friction are **hypothetical operational modeling assumptions**, not established policy constants in `ops-policy.pdf`. If physical field audits cost ₹5,000+, quarterly net benefit drops to -₹15,935.
2. **100% Dependency on Two Outlets (`SP3160` and `SP3232`):**  
   Independent sensitivity testing (`outputs/partner_backtest/independent_audit.md`) established that **`SP3160` (10 frauds) and `SP3232` (4 frauds) account for 100% of the fraud caught by dealer audits**.  
   - When both partners are excluded, **the strategy caught zero fraud (34 false alarms out of 34 claims), producing an operating deficit of -₹23,840.00**.
   - Partner monitoring is an effective targeted response to collusive dealer bursts, but it is **not proven to generalize** across the wider network.
3. **Telemetry Triggers are Pilot Hypotheses:**  
   The backtest ranked outlets using a composite multi-factor telemetry score (`score_partner_signal`). Heuristic alert rules (such as velocity ratio $> 1.5$ and $> 5$ small claims) are **unvalidated pilot hypotheses, not proven operational rules**; when evaluated against the audited Q2 validation data, that specific heuristic flagged only two claims, both of which were clean (zero fraud caught).

---

## 7. Strategic Recommendations & Policy-Feasibility Analysis

Neither claim-level ML models nor partner volume monitoring is a standalone panacea. We evaluate two distinct claim-level operational tracks under the **maximum capacity of 40 claims per month**, followed by recommendations for executive leadership:

### 7.1 Track A: Strict Policy-Compliant Pre-Payment Desk Queue (Deployable under Ops Policy v4.1)
Under current Ops Policy v4.1 (§5), claims under ₹2,000 submitted on or after May 1, 2026 are **auto-approved without partner inspection**. Pre-payment holds cannot be placed on sub-₹2,000 claims without an authorized policy exception. 

Evaluating an **Adaptive Model-Score Quality Gate (`score_v2 >= 0.15`, ceiling 40/month)** strictly on policy-eligible claims (all claims in April; claims $\ge ₹2,000$ in May and June):
- **April 2026 (Pre-Policy):** All claims are policy-eligible for desk review. 18 claims reviewed (16 large, 2 small) $\implies$ **3 TP, 15 FP, 0 Unresolved**, ₹25,184 saved, ₹9,600 cost $\implies$ **+₹15,584.00 net benefit**.
- **May 2026 (Post-Policy, Claims $\ge ₹2\text{k}$):** 3 claims eligible and reviewed $\implies$ **0 TP, 3 FP, 0 Unresolved**, ₹0 saved, ₹1,920 cost $\implies$ **-₹1,920.00 net loss**.
  - *Auto-Approved Sub-₹2,000 Stream (Uninspected):* 9 claims cross `score_v2 >= 0.15`; all pass without pre-payment inspection (0 TP, 9 FP, 0 Unresolved, ₹0 uninspected fraud loss; zero customer friction cost).
- **June 2026 (Post-Policy, Claims $\ge ₹2\text{k}$):** 15 claims eligible and reviewed $\implies$ **0 TP, 14 FP, 1 Unresolved** (₹7,258 amount), ₹0 saved, ₹8,960 cost $\implies$ **-₹8,960.00 net loss** (Worst-Case Bound: -₹9,600.00; Best-Case Bound: -₹1,702.00).
  - *Auto-Approved Sub-₹2,000 Stream (Uninspected):* 10 claims cross `score_v2 >= 0.15`; all pass without pre-payment inspection (1 TP [WC710957, ₹1,978 loss], 9 FP, 0 Unresolved; uninspected fraud loss = ₹1,978.00; zero customer friction cost).
- **Q2 Total (Track A):** **36 claims reviewed** (well below the 40/month ceiling in every month), **3 confirmed frauds caught**, **32 confirmed false alarms**, **1 unresolved claim held** (unknown outcome), **₹25,184.00 avoided loss**, **₹20,480.00 cost** (@ ₹640/FP).
- **Net Operating Benefit:** **+₹4,704.00** (Worst-Case Bound with unresolved clean: **+₹4,064.00**; Best-Case Bound with unresolved fraud: **+₹11,962.00**).
- **Critical Policy Limitation of Track A:** While commercially positive on large claims (+₹4,704.00), this track **catches zero fraud in May and June** because 97.2% of post-May fraud migrated into sub-₹2,000 claims that bypass inspection under current policy.
- **Why Post-Payment Sub-₹2,000 Economics Cannot Be Computed from Historical Data:** Ops Policy v4.1 mandates that claims under ₹2,000 are auto-approved without inspection. The policy defines no post-payment audit clawback workflow, recovery fee structure, or legal recovery timeline. Furthermore, historical CRM logs lack post-payment investigation timestamps and recovery receipts. Any calculation assuming post-payment financial recovery on auto-approved claims would be an unverified simulation rather than a policy-compliant empirical calculation. Sub-₹2,000 claims are therefore reported strictly as uninspected auto-approvals (19 claims crossing `score_v2 >= 0.15`, 1 fraud of ₹1,978, zero pre-payment customer friction).

### 7.2 Track B: Historical Analytical Reference Scenario (55 Claims, Net -₹4,838.00)
If the model-score threshold `score_v2 >= 0.15` is applied to all claims regardless of amount:
- The desk reviews **55 claims** (18 in April, 12 in May, 25 in June; strictly $\le 40$ reviews each month).
- It includes 19 sub-₹2,000 auto-approved claims (9 in May, 10 in June), catching 1 small fraud (WC710957, ₹1,978 saved) but incurring 18 false alarms ($18 \times ₹640 = ₹11,520$ friction).
- **Economics:** 4 confirmed frauds caught, 50 false alarms, 1 unresolved, ₹27,162 saved, ₹32,000 cost $\implies$ **net benefit of -₹4,838.00** (Worst-Case Bound: **-₹5,478.00**; Best-Case Bound: **+₹2,420.00**).
- **Governance Finding:** This represents an **87.9% operating-loss reduction** compared to filling all 40 slots (-₹40,044), but it is **strictly an analytical reference scenario, NOT a deployable pre-payment policy** under current Ops Policy v4.1, because holding sub-₹2,000 claims before payment lacks policy authorization.

### 7.3 Proposed Pilot: Composite Partner Risk Monitoring (Hypothesis for Approval)
Because sub-₹2,000 fraud cannot be profitably or legally policed via pre-payment claim holds, we recommend leadership consider a **controlled 60-day operational pilot**:
- **Separation of Economics:** Partner-audit economics must be kept strictly separate from claim-level desk economics.
- **Assumptions Requiring Budget Approval:** The assumed ₹2,500 audit fee per dealer and any dedicated pilot budget are hypothetical modeling assumptions requiring formal executive authorization before implementation.
- **Pilot Architecture:** Audit rogue partner bursts on a **post-payment audit and clawback basis** (e.g. 2 partner audits/month ceiling), allowing legitimate customers to receive instant auto-approvals while recovering systemic dealer fraud.

---

## 8. Verified Technical Design & Leakage Prevention

The updated V2 pipeline (`src/features.py`) incorporates **23 engineered features** (verified via `FEATURE_COLS_V2`):
- **Post-Policy Telemetry:** Retrospective partner 30-day claim count, 30-day small-claim ratio, and post-policy interaction flags.
- **Strict Leakage Prevention:** Features query only prior claims ($t_{\text{prev}} < t_{\text{curr}}$). `inspector_note` is 100% excluded. Partner onboarding date is excluded to prevent negative transfer.
- **AI Investigation Assistant:** The Groq Llama-3.3-70b assistant acts strictly as a triage brief generator; it **cannot alter model scores** and operates with strict prompt injection containment.

---

## 9. Next Steps & Approvals Needed

1. **Operational Implementation of Track A Quality Gate:** Configure desk triage queues to prioritize policy-eligible claims crossing `score_v2 >= 0.15` up to the maximum capacity of 40 claims/month.
2. **Management Approval for Pilot:** Authorize the proposed dual-track partner audit pilot for Q3 2026 operations and clarify policy mechanisms for auditing sub-₹2,000 auto-approved claims.

*All figures in this memorandum are verified against `outputs/partner_backtest/monthly_results.csv`, `outputs/model_comparison_v2.md`, and `outputs/partner_backtest/independent_audit.md`.*
