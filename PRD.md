# Product Requirements Document
## Pakistan Property Price Prediction Model

**Version:** 1.2  
**Date:** 2026-05-17  
**Status:** Deployed

---

## 1. Problem Statement

Pakistan's residential property market suffers from severe information asymmetry. Buyers and sellers rely on brokers for pricing guidance, and brokers benefit financially from keeping valuations opaque. There is no publicly accessible, data-driven reference for fair market value.

This project builds a machine learning model that predicts the fair sale price of a residential property in Pakistan given its physical and locational attributes. The model transforms raw Zameen.com listing data into a reliable pricing signal that is transparent, reproducible, and fast.

---

## 2. Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | Predict residential sale price within acceptable error | MAPE ≤ 25% on held-out test set |
| G2 | Establish a reproducible baseline to iterate on | RMSE (log-price) documented and versioned |
| G3 | Understand which features drive price predictions | SHAP analysis identifying top 5 features |
| G4 | Produce a clean, analysis-ready dataset | `data_cleaned.csv` with < 2% data loss from cleaning |

---

## 3. Scope

### In Scope

- Residential **For Sale and For Rent** listings (both purposes kept; `purpose` is an input feature)
- Cities covered by the dataset: Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad
- Property types: House, Flat, Upper Portion, Lower Portion, Penthouse, Room (Farm House excluded — see cleaning rules)
- Price range: post-outlier removal (IQR on log-price); sale and rental ranges differ
- Prediction output: price range — low estimate (P10), best estimate (P50), high estimate (P90)

### Out of Scope

- Commercial or industrial properties
- Properties outside the five cities in the dataset
- Real-time price predictions (no live data pipeline in this phase)

---

## 4. Dataset

| Attribute | Detail |
|-----------|--------|
| Source | Zameen.com (Pakistan's largest property portal) |
| File | `Pakistan House Prices and Property Listings.csv` |
| Raw size | 168,446 rows × 17 columns, 43 MB |
| For Sale rows | 120,655 (71.6%) · For Rent rows: 47,791 (28.4%) — both kept |
| Date range | August 2018 – July 2019 |
| Area units | Marla and Kanal only (1 Marla = 272.25 sqft, 1 Kanal = 5,445 sqft) |
| Geographic coverage | Latitude / longitude coordinates available for 99.996% of rows |

### Input Features (model)

| Feature | Type | Description |
|---------|------|-------------|
| `city` | Categorical | One of 5 cities (`province_name` excluded — redundant with city) |
| `property_type` | Categorical | One of 6 property types (Farm House excluded) |
| `purpose` | Categorical | For Sale or For Rent |
| `bedrooms` | Numeric | Number of bedrooms (NaN if unfilled) |
| `baths` | Numeric | Number of bathrooms (NaN if unfilled) |
| `log_area_sqft` | Numeric | `log1p(area_sqft)` — raw area is heavily right-skewed |
| `latitude` | Numeric | Geographic coordinate |
| `longitude` | Numeric | Geographic coordinate |
| `is_premium_location` | Binary | 1 = location name contains DHA/Bahria/Gulberg/Clifton/F-series Islamabad sectors etc. |
| `dist_to_center` | Numeric | Haversine km from property GPS to city's commercial centre |
| `location_median_price` | Numeric | Target-encoded: median log-price of all training listings in same neighbourhood. Fallback: city median → global median. |
| `location_price_per_sqft` | Numeric | Target-encoded: median PKR/sqft of training listings in same neighbourhood. Fallback: city median → global median. |
| `lat_lon_cluster` | Categorical | KMeans cluster ID (50 clusters fit on training lat/lon). Captures spatial price zones that axis-aligned lat/lon splits approximate poorly. |

> **Date features excluded by design:** `year_added` and `month_added` are not model inputs. The training data covers only 2018–2019; tree-based models cannot extrapolate to future years (a 2026 input would be silently treated as 2019, adding noise). `date_added` is retained solely to define the train/test split boundary.

### Target Variable

`log_price = log1p(price)` — predicted in log-space, reported back in PKR.

---

## 5. Data Cleaning Rules

| Rule | Rationale |
|------|-----------|
| Keep both `purpose` values (For Sale + For Rent) | Both are valid price prediction targets; `purpose` becomes an input feature so the model learns different price levels for each |
| Drop `property_id`, `location_id`, `page_url`, `agency`, `agent` | Identifiers with no predictive value |
| Keep `location` column in cleaned data | Used to build `location_lookup.csv` for the website; not a model input feature |
| Convert area strings to `area_sqft` | Standardise mixed Marla/Kanal strings into a single numeric scale |
| Drop rows with zero or null price | Rows with price = 0 are data entry errors |
| IQR outlier removal on `log1p(price)` | Applied **per purpose** separately (For Sale and For Rent have different price scales — combined IQR misses within-purpose outliers) |
| Drop Farm House listings | Sparse category with high MAPE and high area variance. Model cannot learn reliable patterns. Website does not support Farm House valuations. |
| Drop rows outside Pakistan lat/lon bounds (20–40°N, 55–80°E) | 5 rows with clearly invalid/swapped coordinates |
| Replace 0 in `bedrooms` and `baths` with NaN | Zero values represent unfilled fields, not true zero-room properties |
| Cap `bedrooms` and `baths` at 20 | Listings claiming 50+ bedrooms are data errors |
| Exclude `year_added` and `month_added` from model features | Training data covers 2018–2019 only; tree models cannot extrapolate to 2026+. Supplying a future year as input would be silently treated as the nearest training year, producing misleading results. `date_added` is retained only to define the train/test split. |

---

## 6. Model Requirements

### Algorithm
Five models trained and benchmarked (LightGBM selected as primary):

| Model | Family | Notes |
|-------|--------|-------|
| Linear Regression | Linear | Performance floor baseline |
| Random Forest | Bagging | Strong but slower than boosting |
| XGBoost | Boosting | Direct peer of LightGBM; level-wise tree growth |
| LightGBM | Boosting | **Primary model.** Leaf-wise growth, native NaN + categorical support, fastest |
| CatBoost | Boosting | Strong with categoricals; ordered boosting reduces overfitting |

LightGBM selected because: fastest training on 100k+ rows, native NaN handling eliminates imputation for bedrooms/baths, native categorical support, consistently competitive MAPE.

**Four LightGBM models are trained:** one RMSE model (for evaluation metrics) and three quantile models (P10, P50, P90) to produce a price range output rather than a single point estimate.

### Train / Test Split
- **Strategy:** Time-based (not random)
- **Train:** Listings added before 2019-07-01 (~72,700 rows, 60%)
- **Test:** Listings added in July 2019 (~47,900 rows, 40%)
- **Rationale:** Simulates real deployment; model must generalize to unseen future listings

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| RMSE (log-price) | Root mean squared error on log1p(price) | Baseline to beat |
| MAPE (PKR) | Mean absolute percentage error on actual PKR price | ≤ 25% |
| MAPE by city | Per-city breakdown to detect geographic weaknesses | Documented |
| MAPE by property type | Per-type breakdown to detect structural weaknesses | Documented |

---

## 7. Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| `pakistan_property_analysis.ipynb` | Single notebook: EDA → Cleaning → Baseline model | Planned |
| `data_cleaned.csv` | Analysis-ready dataset saved as cleaning checkpoint | Planned |
| Baseline RMSE / MAPE scores | Documented "par scores" for future iterations | Planned |
| Feature importance chart | Top features by LightGBM split count | Planned |
| Residuals analysis | Predicted vs actual, residual distribution, MAPE by segment | Planned |

---

## 8. Implementation Phases

### Phase 1 — Ground Truth (Current)
Build the single analysis notebook covering:
1. **Part 1:** Full EDA — price distributions, city/type breakdowns, geographic heatmap, temporal analysis, null analysis, correlations
2. **Part 2:** Cleaning pipeline — filter, parse area, remove outliers, clean bedrooms/baths, extract date features, save checkpoint
3. **Part 3:** Baseline LightGBM — encode features, time-based split, train with early stopping, evaluate RMSE + MAPE, feature importance, residuals

### Phase 2 — Feature Engineering (Complete)
- ✅ Target encoding for `location` (neighbourhood median log-price, train-only)
- ✅ Haversine distance to city commercial centre (`dist_to_center`)
- ✅ `is_premium_location` binary flag (DHA/Bahria/Gulberg/Clifton/Islamabad F-series)
- ✅ `log_area_sqft` replacing raw `area_sqft` (area is right-skewed)
- ✅ IQR outlier removal applied per-purpose (fixes bimodal distribution problem)
- ✅ `province_name_code` dropped (100% redundant with `city_code`)
- ✅ Lakh/Crore price formatting in all user-facing outputs via `format_pkr()`
- ✅ Confidence indicator: interval width relative to best estimate (High/Moderate/Low)
- ✅ MAPE cross-tab heatmap: city × property type (exposes hidden segment weaknesses)
- ✅ Rental yield sanity check: implied yield should fall in 3-5% for Pakistani market
- ✅ SHAP values analysis: direction and magnitude of each feature's effect

### Phase 3 — Model Tuning (Complete)
- ✅ Benchmark: Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost compared on RMSE / MAE / R² / MAPE
- ✅ KMeans geographic clustering (50 clusters on lat/lon, fit on training data only — `lat_lon_cluster` feature)
- ~~Stacking ensemble (LightGBM + XGBoost + Linear)~~ — dropped; MAPE target (≤ 25%) already achieved at ~18%

### Phase 4 — Deployment (Complete)
- ✅ Inference wrapper: accepts raw property attributes → returns predicted PKR price
- ✅ Model serialization (`joblib`)
- ✅ Confidence interval estimation (quantile regression)
- ✅ FastAPI backend with `/predict`, `/locations`, `/options` endpoints
- ✅ Single-page frontend served from `backend/static/` via FastAPI
- ✅ Deployed as a single Render Web Service → `https://pakistan-property-price.onrender.com`
- ✅ Inflation toggle (×2.9 CPI adjustment) — all displayed values update simultaneously

---

## 9. Known Risks and Constraints

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Listed price ≠ actual transaction price | High | Medium | Document caveat; model predicts listed price, not transacted price |
| Model predicts 2019-era prices, not 2026 prices | High | High | Date features intentionally excluded (they cannot extrapolate to future years). Outputs should be presented as relative/calibrated estimates, not absolute current market values. Retraining on current data is the correct long-term fix. |
| PKR devaluation since training data (2018–2019) | High | High | Model output is in nominal 2019 PKR. Disclaimer shown on all results. Inflation toggle (×2.9 CPI) implemented in Phase 4 — updates all displayed values simultaneously. |
| Sparse data for rare cities / property types | Medium | Medium | Per-segment MAPE monitoring; flag high-error segments |
| Zero-bedroom/bath listings as NaN may bias predictions | Low | Low | LightGBM learns from NaN direction at each split; monitor residuals for this group |

---

## 10. Non-Goals

- This model does **not** replace a professional property valuation
- This model does **not** account for property condition, interior quality, or renovation status
- This model does **not** predict current (2026) market prices — it is calibrated to 2018–2019 listing prices
- This model does **not** forecast future appreciation
