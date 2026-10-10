# Kestrel Home Appliances — Warranty Fraud Detection System

An operational machine learning triage platform designed for Kestrel Home Appliances to detect and prioritize suspicious warranty claims under strict real-world operational constraints.

---

## 1. Project Overview

Kestrel Home Appliances manufactures and services home appliances (Air Fryers, Induction Cooktops, Mixer Grinders, Robot Vacuums, etc.) across India. Warranty claims are submitted by authorized service centers, franchises, and freelance technicians.

- **Objective:** Predict the likelihood that a submitted warranty claim is fraudulent (`is_fraud = 1`).
- **Core Operational Constraint:** The claims investigation desk can manually review **at most 40 claims per month** (operations policy §5; Farhan Sheikh: *"what I want to know is how much fraud we stop per claim we check, in rupees — not a percentage"*).
- **Financial Dynamics:** Each false positive investigation costs Kestrel **Rs. 640** (Rs. 380 customer goodwill delay voucher + Rs. 260 service contact cost). Each true positive caught recovers the full fraudulent payout amount.
- **Model Philosophy:** This system is an **operational risk-ranking tool** to maximize fraud capture per audit within the 40-claim/month budget. It is not an automated denial engine or a guarantee of fraud.

---

## 2. Folder Structure

```text
Kestrel-Fraud-Detection/
├── api/
│   ├── ai_assistant.py            # Groq-powered grounded AI fraud investigation assistant
│   ├── alerts.py                  # Zapier Catch Hook integration & SQLite idempotency
│   ├── app.py                     # Flask REST API & dashboard server
│   ├── explainer.py               # TreeSHAP plain-English explanation layer
│   └── templates/
│       └── index.html             # Responsive operational demo dashboard
├── models/
│   ├── baseline.pkl               # Logistic Regression benchmark pipeline
│   ├── engineer.pkl               # Fitted FraudFeatureEngineer (with cached references)
│   ├── lgbm.pkl                   # Tuned LightGBM model
│   ├── selected_model.pkl         # Selected production model (LightGBM)
│   └── training_meta.json         # Training parameters & validation metrics
├── outputs/
│   ├── alerts.db                  # SQLite persistent alert delivery & idempotency store
│   ├── curves_baseline.png        # Baseline ROC & PR curves
│   ├── curves_lgbm.png            # LightGBM ROC & PR curves
│   ├── evaluation_report.md       # Full out-of-time validation report
│   ├── feature_importance.png     # LightGBM split-gain feature importance plot
│   └── predictions.csv            # Final predictions on test_unlabelled.csv (2,253 lines)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── evaluate.py                # Standalone evaluation script
│   ├── features.py                # Leak-free FraudFeatureEngineer class
│   ├── predict.py                 # Final predictions generation script
│   └── train.py                   # Full training & validation pipeline
├── tests/
│   ├── test_ai_assistant.py       # Groq AI assistant, prompt-injection & resilience tests
│   ├── test_alerts.py             # Zapier integration, idempotency & resilience tests
│   └── test_api.py                # Core REST API endpoint & pipeline test suite
├── .env.example                   # Template for Zapier webhook environment variables
├── .gitignore                     # Git ignore rules protecting .env and database
├── email-thread.txt               # Stakeholder context & constraints
├── ops-policy.pdf                 # Kestrel Operations Policy v4.1
├── partners.csv                   # Partner reference dataset (380 partners)
├── products.csv                   # Product SKU catalog (21 SKUs)
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Project dependencies (Flask, LightGBM, requests, etc.)
├── sample_submission.csv          # Sample submission format (2,252 rows)
├── test_unlabelled.csv            # Unlabeled evaluation claims (2,252 rows)
└── train.csv                      # Historical claims with outcomes (12,029 rows)
```

---

## 3. Environment Setup & Installation

### Requirements
- Python 3.9+ (Windows / Linux / macOS)
- Standard scientific stack (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `flask`, `matplotlib`)

### Installation
Clone or navigate to the repository directory and install dependencies:

```bash
cd Kestrel-Fraud-Detection
pip install -r requirements.txt
```

---

## 4. How to Generate Predictions

To generate or re-verify `outputs/predictions.csv` using the selected model:

```bash
python src/predict.py
```

This script:
1. Loads `models/engineer.pkl` and `models/selected_model.pkl`.
2. Transforms `test_unlabelled.csv`.
3. Validates that output matches `sample_submission.csv` format and ordering exactly.
4. Generates `outputs/predictions.csv` with exactly 2 columns (`claim_id,score`) and exactly 2,253 lines (1 header + 2,252 data rows).
5. Logs score distribution and the top 10 highest-risk claims.

---

## 5. How to Start the REST API & Dashboard

To start the Flask service on port 5000:

```bash
python api/app.py
```

Once running, navigate your browser to:
```
http://127.0.0.1:5000/
```

To run on a custom port:
```bash
PORT=8000 python api/app.py
```

---

## 6. API Endpoints

### 1. `POST /predict` (Single Claim Evaluation)

Evaluates an individual claim, returning the raw fraud score, operational risk band, and 3–5 plain-English reasons based on TreeSHAP attributions.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "claim_id": "WC711348",
  "submitted_at": "2026-07-01 00:41",
  "partner_id": "SP3219",
  "sku": "KH-MG-03",
  "product_serial": "KH395147108",
  "days_since_purchase": 577,
  "claim_amount_inr": 3730.0,
  "photo_attached": "Y",
  "partner_inspected": "Y",
  "claim_description": "power button not working",
  "customer_prior_claims": 1,
  "source": "crm"
}
```
*(Note: `inspector_note` is never used by the model and should be omitted.)*

#### Example Response (`200 OK`)
```json
{
  "claim_id": "WC711348",
  "fraud_score": 0.4525,
  "risk_level": "HIGH",
  "reasons": [
    "Claim amount is unusually high relative to product list price (69% of retail price)",
    "Customer has a prior warranty claim history",
    "Submitted by a franchise partner (statistically elevated risk profile)",
    "High absolute claim payout amount requested (Rs. 3,730)"
  ]
}
```

---

### 2. `POST /predict_bulk` (Batch CSV Scoring)

Uploads a CSV file of warranty claims and returns scored predictions.

#### Option A: Direct CSV Download
```bash
curl -X POST -F "file=@test_unlabelled.csv" http://127.0.0.1:5000/predict_bulk -o scored_claims.csv
```
Returns a downloadable CSV containing all original columns plus `fraud_score` and `risk_level`.

#### Option B: JSON Summary for Web Dashboards
```bash
curl -X POST -F "file=@test_unlabelled.csv" "http://127.0.0.1:5000/predict_bulk?format=json"
```

#### Example Response (`200 OK`)
```json
{
  "average_score": 0.023,
  "download_url": "/download_scored",
  "high_risk_count": 11,
  "low_risk_count": 2171,
  "medium_risk_count": 70,
  "message": "Successfully scored 2252 claims.",
  "preview": [
    {
      "claim_amount_inr": 3730.0,
      "claim_id": "WC711348",
      "fraud_score": 0.4525,
      "partner_id": "SP3219",
      "risk_level": "HIGH",
      "submitted_at": "2026-07-01 00:41"
    }
  ],
  "total_rows": 2252
}
```

---

### 3. `GET /analytics`

Returns population-level metrics across the unlabeled test dataset (Q2 2026), partner risk breakdowns, and validation benchmarks.

```bash
curl http://127.0.0.1:5000/analytics
```

#### Example Response (`200 OK`)
```json
{
  "average_fraud_score": 0.023,
  "highest_claim_id": "WC712506",
  "highest_fraud_score": 0.6268,
  "product_family_summary": [ ... ],
  "risk_distribution": {
    "HIGH": { "count": 11, "percentage": 0.49 },
    "MEDIUM": { "count": 70, "percentage": 3.11 },
    "LOW": { "count": 2171, "percentage": 96.4 }
  },
  "top_risk_partners": [ ... ],
  "total_claims": 2252,
  "validation_performance": { ... }
}
```

---

## 7. Operational Risk Bands

To assist desk investigators in triage, raw probabilities are mapped to operational tiers:

| Tier | Raw Score Range | Recommended Operational Action |
| :--- | :---: | :--- |
| **HIGH** | `score >= 0.30` | **Priority manual audit**; require physical invoice verification and secondary inspection before authorizing payment. |
| **MEDIUM** | `0.10 <= score < 0.30` | **Desk review queue**; audited up to the monthly 40-claim desk capacity limit. |
| **LOW** | `score < 0.10` | **Automated workflow**; low-friction auto-approval and fast customer turnaround. |

*Note: These bands are operational guidelines and do not represent calibrated probability intervals.*

---

## 8. Model Evaluation & Validation Summary

Models were strictly evaluated on an **out-of-time temporal split**:
- **Training Period:** April 1, 2025 – March 31, 2026 (9,015 claims; 96 confirmed fraud cases, 1.06%)
- **Validation Period:** April 1, 2026 – June 30, 2026 (2,131 claims; 45 confirmed fraud cases, 2.11%)

### Performance Benchmark Comparison

| Metric | Logistic Regression (Baseline) | LightGBM (Selected Production Model) |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.6238 | **0.6136** |
| **PR-AUC** | 0.0653 | **0.0569** |
| **Precision@40** (Monthly Desk Limit) | 5.00% (2 caught / 38 false alarms) | **10.00% (4 caught / 36 false alarms)** |
| **Recall@40** | 4.44% | **8.89%** |
| **INR Saved @ 40** | Rs. 15,075 | **Rs. 27,162** |
| **INR Wasted (FP Cost @ Rs. 640)** | Rs. 24,320 | **Rs. 23,040** |
| **Net Financial Benefit @ 40** | **-Rs. 9,245 (Net Loss)** | **+Rs. 4,122 (Net Profit)** |
| **Precision@120** (Total 3-mo Budget) | 4.17% (5 caught) | **4.17% (5 caught)** |
| **F1-Score (@0.50 Threshold)** | 0.0578 | **0.0385** |

### Why LightGBM Was Selected
While overall raw accuracy is ~98% across both models due to severe class imbalance, accuracy is uninformative for fraud triage. Within the actual operational constraint of **40 reviews per month**:
- **LightGBM delivers +Rs. 4,122 net positive return** and **10.00% precision** (stopping 4 fraudulent claims).
- **Logistic Regression is net negative (-Rs. 9,245)** and achieves only 5.00% precision (stopping 2 fraudulent claims).

---

## 9. Important Feature Engineering Decisions & Target Leakage Prevention

1. **Duplicate Claims Handled via First-Submission Retention:**
   - 1,362 rows in `train.csv` represent 681 duplicated `claim_id` values.
   - Analysis verified that 100% of feature values (`claim_amount_inr`, `sku`, `partner_id`, `product_serial`, `claim_description`, `source`, and `is_fraud`) are identical across duplicate pairs, exactly matching Tanmay Kulkarni's explanation (*"Partners re-submit claims when they're bounced"*).
   - 5 duplicate pairs straddled the March/April 2026 split boundary.
   - **Decision:** Deduplicated by keeping the initial submission (`keep='first'` ordered by `submitted_at`). This eliminates cross-split leakage, prevents double-counting retries, and matches `test_unlabelled.csv` (which has 0 duplicate claim IDs).
2. **215 Blank `is_fraud` Labels Excluded:**
   - Blank labels represent cases still under investigation at export time. They were strictly excluded from training and validation sets and **never treated as 0**.
3. **`inspector_note` Fully Excluded:**
   - Notes are authored during or after in-person investigations. Using note text or presence at intake constitutes direct post-hoc leakage.
4. **Partner Age / Onboarding Date Excluded:**
   - In training (Apr 2025 – Mar 2026), newly onboarded partners had near-zero fraud (0.2%). Models given partner age learned that new partners are "safe".
   - In validation, new partners accounted for 10.0% fraud. Including partner age collapsed validation ROC-AUC to 0.30 due to catastrophic negative transfer. Excluding it allowed the model to generalize effectively.
5. **May 1, 2026 Policy Awareness:**
   - Feature `unexpected_no_inspection` recognizes that post-May 1, 2026, claims under Rs. 2,000 do not require inspection, whereas claims >= Rs. 2,000 without inspection represent policy violations.

---

## 10. Known Limitations & Data Quality Constraints

1. **Extremely Few Positive Labeled Cases:**
   - Only 145 confirmed fraud cases exist in the historical training dataset across 12,029 claims (1.2% base rate). Overfitting is an inherent hazard; the model is heavily regularized with shallow depth (`max_depth=3`, `num_leaves=4`) to prioritize generalization over memorization.
2. **Legacy Zoho Zero Imputation:**
   - Legacy Zoho Desk could not store blank labels, so unresolved claims migrated prior to October 2025 were automatically defaulted to `0`. A small fraction of historical 0s may have been undecided cases.
3. **Investigation Capacity Ceiling:**
   - Expanding reviews blindly beyond the top 40 claims per month rapidly accumulates Rs. 640 false positive costs. Operations should adhere strictly to score-ranked prioritization rather than arbitrary thresholding.

---

## 11. Zapier Email Alert Integration

The system includes a zero-overhead, failsafe automation integration that sends email alerts to operations leads when a claim is predicted as **HIGH risk** (`score >= 0.30`).

### 11.1 Architecture & Operational Flow

```text
[Claim Intake / POST /predict]
       │
       ▼
 [LightGBM Scoring & TreeSHAP]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Score < 0.30: LOW / MEDIUM]     [Score >= 0.30: HIGH RISK]
       │                                 │
       ▼                                 ▼
   No Webhook                   Check outputs/alerts.db (Idempotency)
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
                   Already Sent                       Not Sent Yet
                  (Skip duplicate)                        │
                                                          ▼
                                              POST to ZAPIER_WEBHOOK_URL
                                                          │
                                         ┌────────────────┴────────────────┐
                                         ▼                                 ▼
                                    HTTP 2xx (Success)               HTTP Error / Timeout
                                    Mark status='SENT'               Mark status='FAILED'
                                    in outputs/alerts.db             (Safe to retry on next intake)
```

- **Strict Risk Gating:** Alerts only fire for `risk_level == "HIGH"`. `LOW` and `MEDIUM` claims **never** trigger outbound calls.
- **Fail-Safe Resilience:** Outbound HTTP timeouts or network failures never crash or invalidate successful claim predictions.
- **Zero-Leakage Privacy:** Free-text `inspector_note` and raw internal customer identifiers are strictly omitted from the webhook payload.
- **Idempotency Guarantee:** Backed by SQLite (`outputs/alerts.db`) using key `f"{claim_id}:high_risk_claim"`. Duplicate submissions of the same claim will not generate duplicate emails.

### 11.2 Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your Zapier Catch Hook URL in `.env`:
   ```bash
   ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/123456/abcdef/
   ```
3. `.env` is automatically gitignored to prevent secret exposure. If `ZAPIER_WEBHOOK_URL` is omitted, the API continues operating normally and reports `alert.status: "NOT_CONFIGURED"`.

### 11.3 Webhook Payload Format

Outbound JSON sent to Zapier:

```json
{
  "event": "high_risk_claim",
  "claim_id": "WC712506",
  "fraud_score": 0.6268,
  "risk_level": "HIGH",
  "reasons": [
    "Claim amount is unusually high relative to product list price (67% of retail price)",
    "Customer has a prior warranty claim history",
    "Submitted by a franchise partner (statistically elevated risk profile)",
    "Failure reported within 30 days of purchase (high risk early-failure pattern)"
  ],
  "claim_amount_inr": 2131.0,
  "submitted_at": "2026-08-16 12:29",
  "alert_created_at": "2026-10-09T06:30:15.123456+00:00"
}
```

### 11.4 Step-by-Step Zapier Catch Hook & Email Action Setup

Follow these exact steps in your Zapier account to activate email notifications:

1. **Create a New Zap:**
   - Log in to [zapier.com](https://zapier.com) and click **Create Zap**.
2. **Configure Trigger (Catch Hook):**
   - App: Select **Webhooks by Zapier**.
   - Event: Select **Catch Hook**.
   - Click **Continue**.
   - Copy the provided **Webhook URL** (e.g., `https://hooks.zapier.com/hooks/catch/xxxxxx/yyyyyy/`).
   - Paste this URL into your `.env` file as `ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/xxxxxx/yyyyyy/`.
3. **Test the Trigger:**
   - Submit a test high-risk claim to your local server (or click **🔴 High Risk Example** in the dashboard at `http://127.0.0.1:5000/`).
   - In Zapier, click **Test Trigger**. Zapier will capture the incoming payload with `claim_id`, `fraud_score`, `reasons`, and `claim_amount_inr`.
4. **Configure Action (Send Email):**
   - App: Select **Email by Zapier** (or **Gmail** / **Outlook**).
   - Event: Select **Send Outbound Email**.
   - Configure fields:
     - **To:** Operations team email (e.g. `desk-triage@kestrelhome.com`).
     - **Subject:** `⚠️ HIGH RISK FRAUD ALERT: Claim {{claim_id}} (Score: {{fraud_score}})`
     - **Body:**
       ```text
       HIGH RISK WARRANTY CLAIM ESCALATION
       -----------------------------------
       Claim Number   : {{claim_id}}
       Fraud Score    : {{fraud_score}} (High Risk Band)
       Claim Amount   : Rs. {{claim_amount_inr}}
       Submitted At   : {{submitted_at}}

       Primary Risk Drivers (TreeSHAP Attributions):
       {{reasons}}

       Recommended Action:
       Hold payment authorization and route for mandatory physical invoice verification within the 40-claim monthly audit budget.
       ```
5. **Publish the Zap:**
   - Click **Publish**. Your automated escalation pipeline is now live!

### 11.5 Exact Manual Verification Steps

To test a real alert after configuring your own Zapier Catch Hook URL:

1. Ensure your `.env` file contains your active Zapier Catch Hook URL:
   ```bash
   ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/YOUR_HOOK_ID/
   ```
2. Start the Flask application:
   ```bash
   python api/app.py
   ```
3. Open `http://127.0.0.1:5000/` in your browser.
4. Click the **🔴 High Risk Example** button (`WC712506`, score: 62.68%).
5. Observe the result panel:
   - Risk level badge displays `🔴 HIGH RISK`.
   - Alert indicator displays `⚡ Zapier Alert Dispatched: Escalation email webhook triggered.`
6. Click **🔴 High Risk Example** a second time:
   - Alert indicator displays `ℹ️ Zapier Alert: Previously dispatched (idempotent duplicate protected).`
   - Check `outputs/alerts.db` — the claim will only have been dispatched once.
7. Click **🟢 Low Risk Routine** button (`WC711748`, score: 0.51%):
   - Notice that no Zapier alert is dispatched.
8. Check your inbox to confirm receipt of the formatted email from Zapier.

### 11.6 Running Automated Tests

To run the complete test suite:

```bash
# Core REST API and ML pipeline verification
python tests/test_api.py

# Zapier integration, idempotency, privacy and error resilience verification
python tests/test_alerts.py
```

### 11.7 Troubleshooting & Limitations

- **"Zapier Alert: Webhook not configured"**: Ensure `.env` is in the project root directory and `ZAPIER_WEBHOOK_URL` is populated without quotes. Restart `api/app.py` after editing `.env`.
- **"Zapier Alert: Webhook delivery failed"**: Verify that the Zapier Catch Hook is toggled **ON** and that your local machine has outbound internet connectivity. If delivery fails, the status will be marked `FAILED`, allowing automatic retry on subsequent requests.
- **Idempotency Reset**: To re-test a previously alerted claim, delete the record from `outputs/alerts.db` or remove `outputs/alerts.db`.

---

## 12. Groq-Powered AI Fraud Investigation Assistant

The platform integrates a professional, evidence-grounded AI Fraud Investigation Assistant powered by the official Groq API (`groq>=0.9.0`). It acts as a specialized copilot for human warranty investigators—not an autonomous decision engine or generic chatbot.

### 12.1 Core Grounding & Security Principles

```text
[Incoming Claim Data]
        │
        ▼
[Backend Feature Engineering & LightGBM Model] ───► [Single Source of Truth: score & risk band]
        │
        ▼
[TreeSHAP Attribution Generator] ───────────────► [Verified Feature Impact List]
        │
        ▼
[Prompt Sanitizer & Boundary Isolation] ────────► [Neutralize untrusted user text]
        │
        ▼
[Groq API Client (llama-3.3-70b-versatile)]
        │
        ▼
[Structured 5-Part Investigation Brief (JSON)] ───► [Rendered in Dashboard]
```

1. **Model Primacy:** The local LightGBM model remains the sole authority for fraud scores and triage ranking. The AI **cannot** override, recalculate, or alter backend scores.
2. **Strict Evidence Grounding:** The assistant only reasons from verified structured claim telemetry and TreeSHAP feature attributions. It never fabricates product defect history, past investigations, or technician findings.
3. **Prompt-Injection Defense:** Untrusted user input (`claim_description`) is isolated inside XML containment tags (`<claim_description>`) and explicitly ordered to be treated as raw symptom text, neutralizing attempts to override system instructions.
4. **Leakage & Privacy Protection:** Confidential post-investigation notes (`inspector_note`) and unnecessary customer PII are strictly excluded.
5. **Graceful Degradation:** The AI assistant is entirely non-mandatory. If `GROQ_API_KEY` is omitted, rate limits are reached, or the provider is unreachable, core model scoring and triage dashboards function without disruption.

### 12.2 Configuration

Add your Groq API credentials to `.env`:

```bash
# Groq API Key (obtain free key from https://console.groq.com)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Selected Model Identifier (Default: llama-3.3-70b-versatile)
GROQ_MODEL=llama-3.3-70b-versatile
```

#### Verified Model Options on Groq:
- **`llama-3.3-70b-versatile` (Default / Recommended):** State-of-the-art open reasoning, 128k context window, low latency. Ideal for complex multi-factor claim triage.
- **`llama-3.1-8b-instant`:** Ultra-fast inference with generous free-tier throughput limits.

### 12.3 The 5-Part Structured Investigation Output

When invoked via `POST /ai/investigate` or the dashboard's **⚡ Analyze Claim with AI** button, the assistant produces five distinct, scannable sections:

1. **Risk Summary:** Plain-English summary of the model's score and risk band in the context of Kestrel's 40-claim monthly audit ceiling.
2. **Why This Claim Was Flagged:** Transparent explanation grounded directly in the verified TreeSHAP drivers (e.g. claim-to-list ratio, days since purchase, customer prior claim history).
3. **What Should Be Verified:** Practical, actionable verification checks for the desk investigator (e.g. cross-referencing invoice purchase date with serial number prefix, inspecting physical part photos).
4. **Missing Information & Limitations:** Explicit statement of facts not determinable from intake telemetry (e.g. physical part metallurgy, customer dispute records).
5. **Recommended Next Step:** Proportionate operational recommendation (e.g., hold payout for mandatory invoice audit vs. proceed with routine auto-approval).

### 12.4 API Endpoint: `POST /ai/investigate`

Evaluates claim parameters, executes local ML scoring & TreeSHAP reasons on the backend, and queries Groq with structured JSON output enforcement.

#### Request Example
```bash
curl -X POST http://127.0.0.1:5000/ai/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "WC712506",
    "submitted_at": "2026-08-16 12:29",
    "partner_id": "SP3260",
    "sku": "KH-MG-01",
    "product_serial": "KH450246940",
    "days_since_purchase": 19,
    "claim_amount_inr": 2131.0,
    "photo_attached": "Y",
    "partner_inspected": "Y",
    "claim_description": "water leaking",
    "customer_prior_claims": 2,
    "source": "crm"
  }'
```

#### Response Example (`200 OK`)
```json
{
  "claim_id": "WC712506",
  "fraud_score": 0.6268,
  "risk_level": "HIGH",
  "reasons": [
    "Claim amount is unusually high relative to product list price (67% of retail price)",
    "Customer has a prior warranty claim history",
    "Submitted by a franchise partner (statistically elevated risk profile)",
    "Failure reported within 30 days of purchase (high risk early-failure pattern)"
  ],
  "ai_analysis": {
    "status": "SUCCESS",
    "model_used": "llama-3.3-70b-versatile",
    "risk_summary": "Claim WC712506 is classified as HIGH risk with a model risk score of 0.6268, placing it in the top priority review tier under the 40-claim monthly audit budget.",
    "why_flagged": "The primary risk drivers are an unusually high claim payout (67% of retail list price), repeat claim activity by the customer, and an early failure just 19 days post-purchase via a franchise service partner.",
    "verification_steps": [
      "Request original dealer purchase invoice to confirm the 19-day purchase timeline against serial KH450246940.",
      "Verify clear photographs of the failed component showing physical water ingress.",
      "Confirm that technician labor and replacement part pricing match Kestrel master service rate cards."
    ],
    "missing_information_and_limitations": "Physical part inspection records, dealer invoice scans, and customer contact notes are not available in initial CRM telemetry.",
    "recommended_next_step": "Hold automatic payout authorization and assign claim to the desk review queue for mandatory invoice verification."
  }
}
```

### 12.5 Running Automated AI Assistant Tests

To execute the comprehensive AI assistant test suite (fully mocked, zero paid API credit usage):

```bash
python tests/test_ai_assistant.py
```

Covering all 11 scenarios:
- Successful analysis with mocked Groq response
- Graceful degradation when `GROQ_API_KEY` is missing (`NOT_CONFIGURED`)
- Validation of missing & invalid claim fields (HTTP 400)
- Malformed inputs & oversized text truncation defense
- Provider rate-limiting (`RATE_LIMITED`), timeouts (`TIMEOUT`), and service errors (`PROVIDER_ERROR`)
- Prompt injection containment and defense
- Fallback for empty TreeSHAP explanations
- Safe handling of non-JSON provider completions
- Immunity of backend model scores to client-side override attempts
- Regression checks for existing API routes and alert functionality
- Verification that `outputs/predictions.csv` schema and 2,252 rows remain unchanged

### 12.6 Troubleshooting & Provider Quotas

- **`status: "NOT_CONFIGURED"`**: Add `GROQ_API_KEY=gsk_...` to `.env` in the project root and restart the Flask app.
- **`status: "RATE_LIMITED"`**: Groq free-tier rate limits (RPM/TPM) have been reached. The UI will advise the user to wait a moment. For higher throughput, switch to `GROQ_MODEL=llama-3.1-8b-instant` in `.env`.
- **`status: "TIMEOUT"`**: Network latency exceeded the 15-second request threshold. The local LightGBM score remains valid on the screen.

