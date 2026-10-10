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
| **Tests** | `tests/test_api.py`, `tests/test_alerts.py`, `tests/test_ai_assistant.py` |
| **Model Artifacts** | `models/lgbm.pkl`, `models/engineer.pkl`, `models/selected_model.pkl` |
| **Outputs** | `outputs/predictions.csv`, `outputs/evaluation_report.md`, `outputs/curves_lgbm.png`, `outputs/feature_importance.png` |
| **Submission Documents** | `business_memo.md`, `submission-form.md` |

---

## Section 3 — Prediction CSV

| Field | Value |
|---|---|
| **File** | `outputs/predictions.csv` |
| **Schema** | `claim_id,score` |
| **Row Count** | **2,252 data rows** (matching `test_unlabelled.csv` exactly) |
| **Score Range** | 0.0 – 1.0 (raw LightGBM predicted fraud probability) |
| **Verified** | Yes — confirmed by `test_ai_assistant.py` test 11 and manual count |

Sample rows:
```
claim_id,score
WC711348,0.452474
WC711349,0.011776
WC711350,0.011500
```

---

## Section 4 — Model and Training Details

| Field | Value |
|---|---|
| **Selected Model** | LightGBM (`LGBMClassifier`) |
| **Feature Pipeline** | `FraudFeatureEngineer` (17 engineered features) |
| **Training Period** | April 2025 – March 2026 (out-of-time split at April 1 2026) |
| **Validation Period** | April 2026 – June 2026 (temporal holdout) |
| **Validation Strategy** | Strict out-of-time; no random splits; no target encoding across full dataset |
| **Baseline Model** | Logistic Regression |
| **Model Selection Criterion** | Precision@40 (desk review budget) and Net Financial Benefit @ 40 reviews |

### Leakage Prevention

- `inspector_note` is fully excluded from all features, inference, and explanation (post-hoc target leakage).
- Partner onboarding date excluded (caused catastrophic negative transfer in out-of-time validation; see evaluation report Section 2.3).
- Blank `is_fraud` labels excluded from training (never imputed as 0).
- Duplicate `claim_id` rows de-duplicated by keeping first submission (5 pairs straddled the split boundary).
- Global target encoding not used.
- `unexpected_no_inspection` feature is policy-aware: accounts for the May 1 2026 auto-approval threshold change.

---

## Section 5 — Evaluation Metrics (Verified, Out-of-Time)

| Metric | LightGBM (Selected) | Logistic Regression (Baseline) |
|---|---|---|
| **Precision@40** | **10.00% (4/40)** | 5.00% (2/40) |
| **Net Financial Benefit @ 40** | **+Rs. 4,122** | -Rs. 9,245 |
| ROC-AUC | 0.6136 | 0.6238 |
| PR-AUC | 0.0569 | 0.0653 |
| Recall@40 | 8.89% | 4.44% |
| Base Fraud Rate (validation) | 2.11% (45 confirmed frauds in 2,130 labelled val rows) | — |
| Precision@120 (3-month budget) | 4.17% | 4.17% |

Financial model: 40 reviews/month; Rs. 380 goodwill + Rs. 260 service contact = Rs. 640 per false positive.  
Full workings documented in `outputs/evaluation_report.md`.

---

## Section 6 — API Endpoint Documentation

Base URL: `http://127.0.0.1:5000` (local; start with `python api/app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Operational triage dashboard (HTML) |
| `POST /predict` | POST (JSON) | Single claim risk scoring + TreeSHAP explanations + optional Zapier alert |
| `POST /predict_bulk` | POST (multipart CSV) | Batch CSV scoring; `?format=json` for dashboard AJAX use |
| `GET /analytics` | GET | Aggregate risk analytics over `outputs/predictions.csv` population |
| `GET /download_scored` | GET | Download latest bulk-scored CSV |
| `POST /ai/investigate` | POST (JSON) | Full ML scoring + Groq-powered grounded investigation brief |

### Single Predict Request Example

```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"claim_id\":\"WC712506\",\"submitted_at\":\"2026-08-16 12:29\",\"partner_id\":\"SP3260\",\"sku\":\"KH-MG-01\",\"product_serial\":\"KH450246940\",\"days_since_purchase\":19,\"claim_amount_inr\":2131.0,\"photo_attached\":\"Y\",\"partner_inspected\":\"Y\",\"claim_description\":\"water leaking\",\"customer_prior_claims\":2,\"source\":\"crm\"}"
```

### Single Predict Response Schema

```json
{
  "claim_id": "WC712506",
  "fraud_score": 0.6268,
  "risk_level": "HIGH",
  "reasons": ["...", "..."],
  "alert": {"triggered": true, "status": "SENT | NOT_CONFIGURED | ALREADY_SENT | FAILED"}
}
```

---

## Section 7 — Groq AI Integration

| Field | Value |
|---|---|
| **Provider** | Groq Cloud (official `groq` Python SDK v1.7.0) |
| **Model** | `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL` env var) |
| **Endpoint** | `POST /ai/investigate` |
| **Configuration** | `GROQ_API_KEY` in `.env` (excluded from Git) |
| **Graceful Degradation** | Returns `status: NOT_CONFIGURED` if key absent; prediction still works |
| **Grounding** | AI is given only backend-generated TreeSHAP reasons and claim metadata. `inspector_note` is never passed. |
| **Prompt injection defence** | User-controlled `claim_description` field is wrapped in XML containment tags and capped at 400 characters. |
| **Response schema** | 5-part JSON: `risk_summary`, `why_flagged`, `verification_steps`, `missing_information_and_limitations`, `recommended_next_step` |

---

## Section 8 — Alert System

| Field | Value |
|---|---|
| **Trigger** | Automatically fires when `/predict` returns `risk_level: HIGH` |
| **Mechanism** | HTTP POST to Zapier Catch Hook (`ZAPIER_WEBHOOK_URL` in `.env`) |
| **Idempotency** | SQLite event store (`outputs/alerts.db`) prevents duplicate successful alerts per claim |
| **Failsafe** | Timeout, HTTP error, and network failure do not break prediction response |
| **Payload** | Claim metadata + score + reasons; `inspector_note` excluded |

---

## Section 9 — Dashboard Features

The single-page web dashboard (`GET /`) provides:

- **Single Claim Tab:** 12-field intake form with test presets (High / Medium / Low risk), real-time ML scoring, TreeSHAP risk factor breakdown, Zapier alert status, and inline Groq AI investigation brief.
- **Bulk Prediction Tab:** Drag-and-drop CSV upload with risk-tier filter chips (All / High / Medium / Low) and one-click scored CSV download.
- **Operational Analytics Tab:** Pre-scored Q2 2026 population KPIs, validated model benchmark table, clickable risk-tier breakdown (filters partner table), and partner risk summary with clear data-source indicator.

Technology: HTML + Vanilla CSS + Vanilla JavaScript (no build step required). Fonts: Inter + JetBrains Mono via Google Fonts.

---

## Section 10 — Test Coverage

| Test File | Runner | Tests | Status |
|---|---|---|---|
| `tests/test_api.py` | pytest | 1 pytest test (covers GET /, /predict happy path, error cases, /predict_bulk direct & JSON, /analytics) | ✅ PASSED |
| `tests/test_alerts.py` | python (7 sub-tests) | HIGH alert trigger, LOW/MED non-trigger, no-URL fallback, timeout/500 failsafe, idempotency, retry, route regression | ✅ ALL PASSED |
| `tests/test_ai_assistant.py` | python (11 sub-tests) | Mock success, NOT_CONFIGURED, field validation, sanitisation, timeout, rate limit, provider error, injection defence, score tamper, route regression, predictions.csv integrity | ✅ ALL PASSED |

Run all tests:
```bash
python -m pytest tests/test_api.py -v
python tests/test_alerts.py
python tests/test_ai_assistant.py
```

---

## Section 11 — Security and Secrets Management

- `.env` is listed in `.gitignore` and must not be committed.
- `.env.example` provides a documented template with no real keys.
- `GROQ_API_KEY` and `ZAPIER_WEBHOOK_URL` are loaded at runtime from environment variables.
- Webhook URLs are masked in all log output (`url[:28] + "...***"`).
- No API keys appear in any committed file.

---

## Section 12 — Evaluation Artifacts

| Artifact | Location |
|---|---|
| Evaluation report | `outputs/evaluation_report.md` |
| ROC + PR curves (LightGBM) | `outputs/curves_lgbm.png` |
| ROC + PR curves (Baseline) | `outputs/curves_baseline.png` |
| Feature importance chart | `outputs/feature_importance.png` |
| Training configuration & metrics | `models/training_meta.json` |
| Prediction CSV (2,252 rows) | `outputs/predictions.csv` |

---

## Section 13 — Startup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure .env
cp .env.example .env
# Edit .env to add GROQ_API_KEY and/or ZAPIER_WEBHOOK_URL

# 3. Train model (artifacts already present in models/)
python src/train.py

# 4. Start API server
python api/app.py

# 5. Open dashboard
# http://127.0.0.1:5000

# 6. Run tests
python -m pytest tests/test_api.py -v
python tests/test_alerts.py
python tests/test_ai_assistant.py
```

---

*Prediction CSV (`outputs/predictions.csv`) must not be modified. It was generated by `src/predict.py` using the fitted `models/selected_model.pkl` and `models/engineer.pkl` artifacts on `test_unlabelled.csv`.*
