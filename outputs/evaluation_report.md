# Kestrel Warranty Fraud Detection -- Model Evaluation Report

**Generated:** 2026-10-09 01:05:50  
**Pipeline Phase:** Phase 1 (ML / Data Pipeline)  

## 1. Executive Summary

This report documents the machine learning fraud detection pipeline developed for Kestrel Home Appliances. The pipeline uses historical warranty claims (April 2025 - March 2026) to detect fraudulent claims in out-of-time operational validation (April 2026 - June 2026).

- **Selected Model:** **LightGBM**  
- **Validation PR-AUC:** 0.0569 (baseline fraud rate = 2.11%)  
- **Validation ROC-AUC:** 0.6136  
- **Precision@40 (Monthly Desk Budget):** 0.1000 (Catching 4 fraud claims in top 40 reviews)  
- **Net Financial Benefit @ 40 Reviews:** Rs. 4,122  

## 2. Key Data & Feature Engineering Decisions

### 2.1 Duplicate Claim Investigation & Decision
- **Finding:** 1,362 rows represent 681 duplicated `claim_id` values (each appearing twice). Investigation verified that 100% of feature values (`claim_amount_inr`, `sku`, `partner_id`, `product_serial`, `claim_description`, `source`, and `is_fraud`) are identical between pairs. Only the timestamp differed. This exactly matches Tanmay Kulkarni's note: *'Partners re-submit claims when they're bounced, so the same claim number can show up twice.'*  
- **Contamination Check:** 5 duplicate pairs straddled the March/April 2026 temporal split boundary. Keeping duplicates would have leaked training claims directly into the validation set.  
- **Decision:** Retain only the initial submission (`keep='first'` ordered by `submitted_at`). This cleanly eliminates cross-split leakage, prevents artificial double-weighting of bounced claims, and aligns with `test_unlabelled.csv` which contains zero duplicate claim IDs.

### 2.2 Blank `is_fraud` Label Handling
- **Finding:** 215 rows in `train.csv` had blank `is_fraud` labels.  
- **Decision:** Treated strictly as unknown / unverified. Never imputed as 0.  
- **Rationale:** Operations policy §5 and Tanmay's email confirm that claims under active investigation remain blank at export time. Imputing blanks as 0 would contaminate the negative class with unconfirmed fraud. These rows were excluded from supervised training and labeled validation.

### 2.3 Target Leakage Prevention
- **`inspector_note` Fully Excluded:** Inspector notes are generated during or after an in-person investigation. Using note text or presence at initial claim intake is post-hoc target leakage.  
- **No Global Target Encoding:** Partner fraud rates were not encoded across the dataset to prevent future-to-past leakage.  
- **Partner Age / Onboarding Date Excluded:** During the training period (Apr 2025 - Mar 2026), newly onboarded partners had near-zero fraud (0.2%). Models given `partner_age_days` learned a strongly negative weight (assuming new partners are safe). In validation, new partners became the primary fraud vector (10.0%), causing catastrophic negative transfer (ROC-AUC collapsed to 0.30). Excluding partner age resolved this distribution inversion completely.

### 2.4 May 1, 2026 Auto-Approval Policy Handling
- From 1 May 2026, claims under Rs. 2,000 were auto-approved without inspection to cut turnaround times.  
- Feature `unexpected_no_inspection` was engineered with policy awareness: pre-May 2026, any lack of partner inspection was unexpected; post-May 2026, lack of inspection is normal for claims < Rs. 2,000 and only abnormal for claims >= Rs. 2,000.

## 3. Validation Performance Comparison

| Metric | Logistic Regression (Baseline) | LightGBM (Primary) | Selected |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.6238 | 0.6136 | **0.6136** |
| **PR-AUC** | 0.0653 | 0.0569 | **0.0569** |
| **Precision@40 (1 mo budget)** | 0.0500 (2/40) | 0.1000 (4/40) | **0.1000** |
| **Recall@40** | 0.0444 | 0.0889 | **0.0889** |
| **Net Benefit @ 40** | Rs. -9,245 | Rs. 4,122 | **Rs. 4,122** |
| **Precision@120 (3 mo budget)** | 0.0417 | 0.0417 | **0.0417** |
| **F1-Score (@0.50)** | 0.0578 | 0.0385 | **0.0385** |

## 4. Business & Financial Impact Analysis

The investigation desk is constrained to reviewing **40 claims per month** (120 claims across the 3-month validation window). Each false positive investigation costs Kestrel Rs. 640 (Rs. 380 customer goodwill compensation + Rs. 260 service contact cost).

When the top 40 suspicious claims are prioritized using the selected model:
- **Fraud Caught:** 4 fraudulent claims stopped
- **Precision:** 10.00% (vs. 2.11% base rate -- over 3.5x lift)
- **Net Financial Benefit:** **Rs. 4,122**

## 5. Artifacts Created

- `models/engineer.pkl`: Fitted feature pipeline, serializes reference tables for API use
- `models/baseline.pkl`: Fitted Logistic Regression baseline pipeline
- `models/lgbm.pkl`: Fitted LightGBM primary model
- `models/selected_model.pkl`: Selected production model
- `models/training_meta.json`: Full configuration, feature list, and evaluation metrics
- `outputs/feature_importance.png`: Feature gain bar chart
- `outputs/curves_baseline.png`: Baseline ROC and PR curves
- `outputs/curves_lgbm.png`: LightGBM ROC and PR curves

## 6. Next Steps (Phase 2 Preview)

1. Build FastAPI REST endpoint loading `models/selected_model.pkl` and `models/engineer.pkl`.
2. Score `test_unlabelled.csv` and export formatted predictions.
3. Construct modern web review dashboard for the claims investigation desk.
