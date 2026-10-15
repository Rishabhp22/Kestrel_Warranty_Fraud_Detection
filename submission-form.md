# Submission Form — Kestrel Home Appliances Warranty Fraud Detection (Variant C)

---

## Section 1 — Candidate Details

| Field | Value |
|---|---|
| **Name** | Rishabh Poddar |
| **Email** | rishabhpoddar22092005@gmail.com |
| **Submission Date** | October 2026 |
| **Assignment Variant** | Variant C |

---

## Section 2 — Repository and Project Structure

| Item | Path / Location |
|---|---|
| **Project Root** | `Kestrel-Fraud-Detection/` |
| **Source (ML pipeline)** | `src/train.py`, `src/features.py`, `src/predict.py`, `src/evaluate.py` |
| **API / Dashboard** | `api/app.py`, `api/ai_assistant.py`, `api/alerts.py`, `api/explainer.py` |
| **Dashboard Template** | `api/templates/index.html` |
| **Tests** | `tests/test_api.py`, `tests/test_features_v2.py`, `tests/test_evaluate_v2.py`, `tests/test_partner_backtest.py`, `tests/test_alerts.py`, `tests/test_ai_assistant.py` |
| **Model Artifacts (Production)** | `models/selected_model.pkl`, `models/engineer.pkl`, `models/lgbm.pkl`, `models/baseline.pkl` |
| **Candidate Artifacts (V2)** | `models/v2_candidate/lgbm_v2.pkl`, `models/v2_candidate/engineer_v2.pkl`, `models/v2_candidate/training_meta_v2.json` |
| **Prediction Outputs** | `outputs/predictions.csv` (2,252 test claims scored) |
| **Verification & Audit Reports** | `outputs/model_comparison_v2.md`, `outputs/v2_independent_verification.md`, `outputs/v2_evaluation_report.md` |
| **Partner Backtest Deliverables** | `outputs/partner_backtest/partner_backtest_report.md`, `outputs/partner_backtest/monthly_results.csv`, `outputs/partner_backtest/partner_risk_signals.csv`, `outputs/partner_backtest/limitations.md`, `outputs/partner_backtest/independent_audit.md` |
| **Submission Documents** | `business_memo.md`, `submission-form.md`, `TASK2_REAUDIT_REPORT.md` |

---

## Section 3 — Prediction CSV

| Field | Value |
|---|---|
| **File** | `outputs/predictions.csv` |
| **Schema** | `claim_id,score` |
| **Row Count** | **2,252 data rows** (matching `test_unlabelled.csv` and `sample_submission.csv` exactly) |
| **Score Range** | 0.0 – 1.0 (predicted fraud probability) |
| **Verified** | Yes — confirmed by `test_ai_assistant.py` test 11 and manual count |
| **Status** | Promoted V2 LightGBM model; outputs/predictions.csv generated with 23 features on 2,252 test claims |

Sample rows:
```csv
claim_id,score
WC711348,0.452474
WC711349,0.011776
WC711350,0.011500
```

The test dataset (`test_unlabelled.csv`) contains 2,252 claims with unobserved ground-truth labels. While V2 caught 6 fraud cases versus 5 for V1 on historical Q2 validation under the evaluated review-capacity setup, performance on the hidden test set remains unverified. Historical validation results are not hidden-test results.

---

## Section 4 — Model and Training Details

| Field | Value |
|---|---|
| **Selected Model Architecture** | LightGBM (`LGBMClassifier`), depth-constrained with balanced class weighting |
| **Feature Pipeline** | `FraudFeatureEngineer` (**23 engineered features**, verified via `FEATURE_COLS_V2`) |
| **Training Period** | April 2025 – March 2026 (out-of-time temporal split at April 1, 2026; 9,015 labeled claims, 96 confirmed fraud) |
| **Validation Period** | April 2026 – June 2026 (out-of-time temporal holdout; 2,131 labeled claims, 45 confirmed fraud; 67 unresolved claims) |
| **Operational Evaluation** | Maximum capacity of **40 claims per month** (review ceiling under Ops Policy v4.1, §5; 120 reviews over Q2 2026 under baseline queue) |
| **Baseline Model** | Logistic Regression Benchmark (17 features) & V1 LightGBM Baseline |
| **Model Selection Criterion** | Monthly Precision@40, Avoided Fraud Loss, and Operational Net Benefit under Ops Policy v4.1 |

### Leakage Prevention & Data Integrity Safeguards
- **`inspector_note` Excluded:** Post-investigation field is 100% excluded from all features, training, inference, and explanations.
- **Strict Point-in-Time Rolling Windows:** All telemetry features (`partner_claims_last_30d`, `partner_small_claim_ratio_30d`) strictly query historical transactions submitted prior to the current claim ($t_{\text{prev}} < t_{\text{curr}}$).
- **No Partner Historical Fraud Rates:** Because resolution timestamps are absent from raw data, historical fraud labels cannot be established at scoring time. No partner fraud-rate features were created, avoiding label leakage.
- **Unresolved Label Isolation:** All 67 validation claims with blank `is_fraud` labels are included in volume telemetry but strictly excluded from supervised training targets. They are never relabelled as non-fraud (0).
- **Deduplication Safeguard:** Duplicate `claim_id` submissions (which occur when partners resubmit bounced claims) are deduplicated by keeping the earliest submission timestamp (`keep='first'`), preventing split contamination.
- **Policy Threshold Indicator:** Policy features reflect the May 1, 2026 auto-approval threshold (`claim_amount_inr < 2000.0`).

---

## Section 5 — Evaluation Metrics (Verified, Out-of-Time Monthly Evaluation)

### 5.1 Month-by-Month Operational Triage (Ops Policy v4.1: 40 Claims/Month Budget)
*(Evaluated across the 2,198 operational validation claims; source: `outputs/partner_backtest/monthly_results.csv`)*

| Month | Policy Regime | Model | TP (Fraud) | FP (Clean) | Unresolved | Precision (%) | Recall (%) | Gross Loss Avoided | False-Alarm Cost (₹640/FP) | Net Operating Benefit |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **April 2026** | Pre-Policy | V1 LightGBM | 3 | 35 | 2 | 7.89% | 33.33% | ₹25,184.00 | ₹22,400.00 | **+₹2,784.00** |
|  |  | **V2 LightGBM** | **4** | **35** | **1** | **10.26%** | **44.44%** | **₹26,383.00** | **₹22,400.00** | **+₹3,983.00** |
| **May 2026** | Post-Policy (< ₹2k auto) | V1 LightGBM | 1 | 38 | 1 | 2.56% | 7.14% | ₹1,995.00 | ₹24,320.00 | **-₹22,325.00** |
|  |  | **V2 LightGBM** | **1** | **38** | **1** | **2.56%** | **7.14%** | **₹1,995.00** | **₹24,320.00** | **-₹22,325.00** |
| **June 2026** | Post-Policy (< ₹2k auto) | V1 LightGBM | 1 | 37 | 2 | 2.63% | 4.55% | ₹1,978.00 | ₹23,680.00 | **-₹21,702.00** |
|  |  | **V2 LightGBM** | **1** | **37** | **2** | **2.63%** | **4.55%** | **₹1,978.00** | **₹23,680.00** | **-₹21,702.00** |
| **Q2 Total** | **Quarterly Total (120)** | V1 LightGBM | 5 | 110 | 5 | 4.35% | 11.11% | ₹29,157.00 | ₹70,400.00 | **-₹41,243.00** |
|  |  | **V2 LightGBM** | **6** | **110** | **4** | **5.17%** | **13.33%** | **₹30,356.00** | **₹70,400.00** | **-₹40,044.00** |

*(Note: In the closed labeled-only validation set ($N=2,131$), where unresolved claims are pre-filtered, V2 FP was 114 and net benefit was -₹42,604.00, and V1 FP was 115 and net benefit was -₹44,443.00. Both evaluations confirm identical fraud catch and avoided loss).*

### 5.2 Economic Cost Basis & Disclosure of Financial Realities
- **Cost Parameters (Ops Policy v4.1, §4):** Goodwill compensation = ₹380 average for repair delay; Blended service contact cost = ₹260. The combined ₹640 friction is an operational modeling assumption that couples both costs under the assumption that 100% of delayed genuine customers contact customer support.
- **Financial Reality Statement:** Negative net benefit (-₹40,044.00) is an **operating loss**, NOT a positive business outcome. Under policy costs, individual claim triage cannot profitably police sub-₹2,000 claims post-May.
- **Operational Track A: Policy-Compliant Pre-Payment Desk Review (`score_v2 >= 0.15`, ceiling 40/month):** Note that `score_v2 >= 0.15` is an uncalibrated model-score threshold, not a calibrated 15% fraud probability. Applying Ops Policy v4.1 strictly separates eligible claims from auto-approved claims: in April 2026 (pre-policy), all claims are eligible (18 reviewed, 3 TP, 15 FP, 0 Unresolved, ₹25,184 saved, ₹9,600 cost $\to$ **+₹15,584.00 net benefit**); in May 2026, only claims $\ge ₹2,000$ are eligible for pre-payment desk review (3 reviewed, 0 TP, 3 FP, 0 Unresolved, ₹0 saved, ₹1,920 cost $\to$ **-₹1,920.00 net loss**; 9 sub-₹2,000 claims with `score_v2 >= 0.15` are auto-approved without inspection, with 0 TP, 9 FP, and ₹0 fraud loss); in June 2026, only claims $\ge ₹2,000$ are eligible (15 reviewed, 0 TP, 14 FP, 1 Unresolved [₹7,258 amount], ₹0 saved, ₹8,960 cost $\to$ **-₹8,960.00 net loss**; 10 sub-₹2,000 claims with `score_v2 >= 0.15` are auto-approved without inspection, with 1 TP [WC710957, ₹1,978 loss], 9 FP, and 0 Unresolved).
  - **Q2 Track A Totals:** **36 claims reviewed** (strictly $\le 40$ reviews each month), **3 confirmed frauds**, **32 confirmed false alarms**, and **1 unresolved claim held** (unknown outcome). Avoided fraud loss: ₹25,184.00, false-alarm friction: ₹20,480.00 $\to$ **Net Operating Benefit: +₹4,704.00** (Worst-Case Bound with unresolved clean: **+₹4,064.00**; Best-Case Bound with unresolved fraud: **+₹11,962.00**).
  - **Policy Feasibility Limitation:** While commercially positive on large claims (+₹4,704.00), Track A catches zero fraud in May and June because 97.2% of post-May fraud migrated into sub-₹2,000 claims that bypass inspection under current policy.
  - **Sub-₹2,000 Economics Limitation:** Ops Policy v4.1 mandates auto-approval without inspection for claims $< ₹2,000$ and does not define a post-payment audit clawback mechanism, recovery fee, or timeline. Furthermore, the dataset contains no post-payment audit records. Policy-compliant post-payment recovery cannot be computed from available data without inventing assumptions; sub-₹2,000 claims are therefore reported strictly as uninspected auto-approvals (19 claims crossing `score_v2 >= 0.15`, 1 fraud of ₹1,978, zero pre-payment desk friction).
- **Operational Track B: Historical Analytical Reference Scenario (55 Claims, Net -₹4,838.00):** If an unconstrained pre-payment hold were hypothetically assumed across all claims with `score_v2 >= 0.15` regardless of policy amount, the desk would review 55 claims (18 in April, 12 in May, 25 in June; strictly $\le 40$ reviews each month), holding 19 sub-₹2,000 auto-approved claims. This catches 4 confirmed frauds, flags 50 false alarms, and holds 1 unresolved claim, yielding **-₹4,838.00 net benefit** (Worst-Case Bound: **-₹5,478.00**; Best-Case Bound: **+₹2,420.00**). This represents an **87.9% operating-loss reduction** compared to the baseline 40-slot queue (-₹40,044.00), but is **strictly an analytical reference scenario, NOT a deployable result**, because holding sub-₹2,000 claims before payment lacks policy authorization under Ops Policy v4.1.

### 5.3 Partner Dealer Audit Backtest Findings & Caveats
- An offline backtest (`outputs/partner_backtest/partner_backtest_report.md`) showed that auditing the **Top 2 burst partners per month** consolidated investigations, catching 14 small frauds with 22 false alarms across Q2 (₹19,785 saved, ₹20,720 cost, net -₹935.00).
- **Mandatory Caveat 1 (Simulation Cost Assumptions):** The assumed ₹2,500 dealer audit fee and ₹260 clean-claim holding friction are **hypothetical operational modeling assumptions requiring budget approval**, not established policy constants in `ops-policy.pdf`. Partner-audit economics are kept strictly separate from claim-level desk economics.
- **Mandatory Caveat 2 (Two-Partner Dependency):** Independent audit (`outputs/partner_backtest/independent_audit.md`) proved that **100% of the fraud caught by dealer audits came from two partners (`SP3160` and `SP3232`)**. Excluding them reduces caught fraud to zero (net benefit -₹23,840.00).
- **Telemetry Triggers are Pilot Hypotheses:** Proposed telemetry triggers (such as velocity ratios and small-claim volume thresholds) are **unvalidated pilot hypotheses, not proven operational rules**; an empirical check of the heuristic `velocity > 1.5 and > 5 small claims` against audited Q2 validation data flagged only two claims, both of which were clean (zero fraud caught). True partner burst detection requires multi-factor composite telemetry.
- **Strategic Recommendation:** Maintain the **maximum capacity of 40 claims per month** as an operational ceiling (without requiring the desk to fill low-confidence slots) while leadership considers authorizing a **controlled dual-track operational pilot** (quality-gated desk reviews for claims $\ge ₹2,000$, plus a separate operational trigger for 2 dealer burst audits per month).

---

## Section 6 — API Endpoint Documentation

Base URL: `http://127.0.0.1:5000` (start with `python api/app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Operational triage dashboard (HTML) |
| `POST /predict` | POST (JSON) | Single claim risk scoring + TreeSHAP explanations + optional Zapier alert |
| `POST /predict_bulk` | POST (multipart CSV) | Batch CSV scoring; `?format=json` for dashboard AJAX use |
| `GET /analytics` | GET | Aggregate risk analytics over population |
| `GET /download_scored` | GET | Download latest bulk-scored CSV |
| `POST /ai/investigate` | POST (JSON) | Full ML scoring + Groq-powered grounded investigation brief |

### Single Predict Request Example
```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"claim_id\":\"WC712506\",\"submitted_at\":\"2026-08-16 12:29\",\"partner_id\":\"SP3260\",\"sku\":\"KH-MG-01\",\"product_serial\":\"KH450246940\",\"days_since_purchase\":19,\"claim_amount_inr\":2131.0,\"photo_attached\":\"Y\",\"partner_inspected\":\"Y\",\"claim_description\":\"water leaking\",\"customer_prior_claims\":2,\"source\":\"crm\"}"
```

---

## Section 7 — Groq AI Integration

| Field | Value |
|---|---|
| **Provider** | Groq Cloud (official `groq` Python SDK) |
| **Model** | `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL` env var) |
| **Endpoint** | `POST /ai/investigate` |
| **Configuration** | `GROQ_API_KEY` in `.env` (excluded from Git) |
| **Graceful Degradation** | Returns `status: NOT_CONFIGURED` if key absent; model scoring operates uninterrupted |
| **Grounding & Security** | AI is provided only backend TreeSHAP reasons and claim metadata. `inspector_note` is never passed. Untrusted `claim_description` is enclosed in XML containment tags and length-capped. |
| **Integrity Guard** | Client-side requests cannot override backend model scores. The assistant provides triage support and cannot be cited as evidence of improved accuracy. |

---

## Section 8 — Alert System

| Field | Value |
|---|---|
| **Trigger** | Automatically fires when `/predict` returns `risk_level: HIGH` |
| **Mechanism** | HTTP POST to Zapier Catch Hook (`ZAPIER_WEBHOOK_URL` in `.env`) |
| **Idempotency** | SQLite event store (`outputs/alerts.db`) prevents duplicate alerts per claim |
| **Failsafe** | Network timeouts and HTTP 500 errors do not break core prediction responses |
| **Payload** | Metadata + score + reasons; `inspector_note` excluded |

---

## Section 9 — Dashboard Features

- **Single Claim Tab:** 12-field intake form with test presets (High / Medium / Low risk), real-time ML scoring, TreeSHAP risk factor breakdown, Zapier alert status, and inline Groq AI investigation brief.
- **Bulk Prediction Tab:** Drag-and-drop CSV upload with risk-tier filter chips and one-click scored CSV download.
- **Operational Analytics Tab:** Population KPIs, validated model benchmark table, clickable risk-tier breakdown, and partner risk summary with clear data-source indicators.
- **Technology:** HTML5 + Vanilla CSS + Vanilla JavaScript (no external build step required). Fonts: Inter + JetBrains Mono.

---

## Section 10 — Test Coverage & Execution Summary

All test suites were independently executed and passed with zero failures:

| Test Suite | Runner | Test Count | Scope & Coverage | Status |
|---|---|:---:|---|:---:|
| `tests/test_api.py` | pytest | 1 | REST API routes (GET /, /predict happy/error, /predict_bulk, /analytics) | ✅ PASSED |
| `tests/test_features_v2.py` | pytest | 5 | Chronological leakage, 30d window bounds, cold start, unresolved telemetry, schema | ✅ PASSED |
| `tests/test_evaluate_v2.py` | pytest | 3 | Top-40 monthly economics, candidate schema, production artifact immutability | ✅ PASSED |
| `tests/test_partner_backtest.py`| pytest | 3 | Partner chronological leakage, unlabelled claim isolation, CSV arithmetic reconciliation | ✅ PASSED |
| `tests/test_alerts.py` | python | 7 | Webhook triggers, non-triggers, missing URL fallback, timeout resilience, idempotency | ✅ PASSED |
| `tests/test_ai_assistant.py` | python | 11 | Mock completion, missing key fallback, injection defense, score tampering, predictions.csv integrity | ✅ PASSED |

**Total Automated Tests:** **30 tests executed and passing**.

Run all test suites:
```bash
pytest tests/test_api.py tests/test_features_v2.py tests/test_evaluate_v2.py tests/test_partner_backtest.py -v
python tests/test_alerts.py
python tests/test_ai_assistant.py
```

---

## Section 11 — Security and Secrets Management

- `.env` is listed in `.gitignore` and protected from VCS commits.
- `.env.example` provides a documented template with zero credentials.
- `GROQ_API_KEY` and `ZAPIER_WEBHOOK_URL` are loaded at runtime from environment variables.
- Webhook URLs are masked in all log output (`url[:28] + "...***"`).
- Zero secrets or credentials exist in committed repository files.

---

## Section 12 — Evaluation Artifacts & Verification Files

| Artifact | Location | Description |
|---|---|---|
| **Model Comparison Report** | [`outputs/model_comparison_v2.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/model_comparison_v2.md) | V1 vs V2 LightGBM vs CatBoost monthly evaluation |
| **Independent Verification Audit**| [`outputs/v2_independent_verification.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/v2_independent_verification.md) | Verification of V2 candidate metrics and exact arithmetic |
| **Partner Backtest Report** | [`outputs/partner_backtest/partner_backtest_report.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/partner_backtest_report.md) | Chronological backtest of 9 partner-monitoring strategies |
| **Monthly Results CSV** | [`outputs/partner_backtest/monthly_results.csv`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/monthly_results.csv) | Exact month-by-month performance and cost breakdown |
| **Partner Signals CSV** | [`outputs/partner_backtest/partner_risk_signals.csv`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/partner_risk_signals.csv) | Full dataset of 2,198 validation claims with partner telemetry |
| **Backtest Limitations Doc** | [`outputs/partner_backtest/limitations.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/limitations.md) | Documentation of timestamp limits and partner concentration |
| **Backtest Independent Audit** | [`outputs/partner_backtest/independent_audit.md`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/partner_backtest/independent_audit.md) | Verification audit reconciling divergences and SP3160 dependency |
| **Feature Importance Plot** | [`outputs/v2_feature_importance.png`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/v2_feature_importance.png) | LightGBM feature gain chart |
| **ROC & PR Curves Plot** | [`outputs/v2_curves_lgbm.png`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/v2_curves_lgbm.png) | Discrimination curves for V2 model |
| **Predictions CSV** | [`outputs/predictions.csv`](file:///d:/AI_ML_JOURNEY/PYTHON_LEARNING/Projects/Kestrel-Fraud-Detection/outputs/predictions.csv) | 2,252 scored test claims |

---

## Section 13 — Startup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure .env
cp .env.example .env
# Edit .env to add GROQ_API_KEY and/or ZAPIER_WEBHOOK_URL

# 3. Start API server & Dashboard
python api/app.py
# Open http://127.0.0.1:5000

# 4. Run full automated test suite
pytest tests/test_api.py tests/test_features_v2.py tests/test_evaluate_v2.py tests/test_partner_backtest.py -v
python tests/test_alerts.py
python tests/test_ai_assistant.py
```
