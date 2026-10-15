# Kestrel Warranty Fraud Detection — Version 2 Operational Evaluation Report

**Generated:** 2026-10-14 21:16:03  
**Evaluation Type:** Strict Monthly Operational Capacity (40 Claims/Month) Holdout  

## 1. Executive Summary

This report documents the rigorous re-evaluation of the Kestrel Warranty Fraud Detection models under the operational review desk constraint of **40 claims per month** (120 reviews over the April–June 2026 validation window). Candidate Version 2 models incorporate point-in-time outlet telemetry and May 1 policy regime features without introducing target or temporal leakage.

## 2. Algorithm Comparison (Operational Monthly Capacity: 120 Total Reviews)

| Model | Features | Precision@Capacity (120) | Total Frauds Caught | Recall@Capacity | Loss Avoided (INR) | Waste (FP Cost) | Net Operating Benefit |
|---|---|---|---|---|---|---|---|
| **Baseline LightGBM (17 Features)** | 17 | **4.17%** (5/120) | **5** / 45 | 11.11% | Rs. 29,157 | Rs. 73,600 | **Rs. -44,443** |
| **LightGBM V2 Candidate (23 Features)** | 23 | **5.00%** (6/120) | **6** / 45 | 13.33% | Rs. 30,356 | Rs. 72,960 | **Rs. -42,604** |
| **Logistic Regression V2 (23 Features)** | 23 | **4.17%** (5/120) | **5** / 45 | 11.11% | Rs. 29,157 | Rs. 73,600 | **Rs. -44,443** |
| **Random Forest V2 (23 Features)** | 23 | **3.33%** (4/120) | **4** / 45 | 8.89% | Rs. 27,162 | Rs. 74,240 | **Rs. -47,078** |

## 3. Month-by-Month Operational Breakdown (40 Reviews/Month)

### 3.1 Candidate Model (LightGBM V2 — 23 Features)

| Month | Regime | Eligible Claims | Fraud Total | Base Rate | Precision@40 | Recall@40 | TP (Small/Large) | FP | Saved | Waste (@Rs.640) | Net Benefit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | Pre-Policy | 709 | 9 | 1.27% | **10.00%** (4/40) | 44.44% | 4 (1/3) | 36 | Rs. 26,383 | Rs. 23,040 | **Rs. 3,343** |
| 2026-05 | Post-Policy | 709 | 14 | 1.97% | **2.50%** (1/40) | 7.14% | 1 (1/0) | 39 | Rs. 1,995 | Rs. 24,960 | **Rs. -22,965** |
| 2026-06 | Post-Policy | 713 | 22 | 3.09% | **2.50%** (1/40) | 4.55% | 1 (1/0) | 39 | Rs. 1,978 | Rs. 24,960 | **Rs. -22,982** |

### 3.2 Baseline Model (Production LightGBM — 17 Features)

| Month | Regime | Eligible Claims | Fraud Total | Base Rate | Precision@40 | Recall@40 | TP (Small/Large) | FP | Saved | Waste (@Rs.640) | Net Benefit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | Pre-Policy | 709 | 9 | 1.27% | **7.50%** (3/40) | 33.33% | 3 (0/3) | 37 | Rs. 25,184 | Rs. 23,680 | **Rs. 1,504** |
| 2026-05 | Post-Policy | 709 | 14 | 1.97% | **2.50%** (1/40) | 7.14% | 1 (1/0) | 39 | Rs. 1,995 | Rs. 24,960 | **Rs. -22,965** |
| 2026-06 | Post-Policy | 713 | 22 | 3.09% | **2.50%** (1/40) | 4.55% | 1 (1/0) | 39 | Rs. 1,978 | Rs. 24,960 | **Rs. -22,982** |

## 4. Key Empirical & Operational Conclusions

1. **Pre-Policy Performance (April 2026):** LightGBM V2 improved Precision@40 from 7.50% (3/40) to **10.00% (4/40)** and Recall@40 from 33.33% to **44.44%**, boosting monthly Net Benefit from +Rs. 1,504 to **+Rs. 3,343**.
2. **Post-Policy Economics (May & June 2026):** Under the fixed 40-claim desk capacity, both models generated deficits in May and June because 97.2% of post-May fraud migrated to sub-Rs.2,000 claims. At a penalty of Rs. 640 per false alarm, each false positive heavily degrades the financial return when catching small claims worth only ~Rs. 1,500.
3. **Overall Operational Advantage:** LightGBM V2 caught **6 frauds (13.33% recall)** vs 5 frauds (11.11% recall) for baseline, producing the highest net operating return among all tested models.
