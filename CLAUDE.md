# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

All four phases are complete. The project is a working end-to-end property price estimator:
- ML model trained on 120k+ Zameen.com listings (2018–2019)
- FastAPI backend serving predictions
- Single-page frontend with full UI

## Running the Backend

```powershell
cd backend
python -m uvicorn main:app --reload
```

`pip` and `uvicorn` are not on PATH on this machine — always use `python -m` prefix. Backend runs at `http://127.0.0.1:8000`. The frontend (`frontend/index.html`) is opened directly in the browser; it hardcodes `http://127.0.0.1:8000` as the API base.

## Backend Architecture

Three files in `backend/`:

| File | Role |
|------|------|
| `main.py` | FastAPI app. Three endpoints: `GET /options`, `GET /locations?q=&city=`, `POST /predict` |
| `inference.py` | Loads all artifacts at import time, exposes `predict()` and `get_locations()` |
| `model_artifacts.pkl` | Serialized model + encoders + lookup dicts (joblib, compress=3) |
| `location_lookup.csv` | Neighbourhood → mean lat/lon + listing count + assigned city |
| `requirements.txt` | fastapi, uvicorn, lightgbm, pandas, numpy, scikit-learn, joblib, rapidfuzz |

**Request flow:** `POST /predict` → Pydantic validates → `inference.predict()` → fuzzy-matches location name → builds 13-feature row → runs 3 quantile models → returns p10/p50/p90 + formatted strings + confidence + lat/lon.

**Key inference details:**
- `_fuzzy_match()` uses rapidfuzz with `score_cutoff=60`. Falls back to exact match if rapidfuzz not installed.
- Location fallback chain: neighbourhood median → city median → global median (for both `location_median_price` and `location_price_per_sqft`).
- `None` bedrooms/baths → `np.nan` (not `None`) before building the DataFrame — pandas creates object dtype from None, which LightGBM rejects.
- City assignment on `location_lookup.csv` is computed at startup via nearest city centre (Haversine). Used to filter autocomplete results by city.
- `format_pkr()` in inference.py: ≥1 Crore → "X.XX Crore", ≥1 Lakh → "X.X Lakh", else plain `{n:,}`.

## Frontend Architecture

Single file: `frontend/index.html`. No build step, no dependencies, open directly in browser.

**CSS design system — do not bypass these:**
- Single font: Inter (weights 400/500/600/700/800 only). Do not add Montserrat, IBM Plex Mono, or any other family.
- All colours are CSS tokens in `:root`. Never add hardcoded hex values — always use or extend the token set:
  - `--accent` / `--accent-dark` / `--accent-dim` — brand green and its tints
  - `--text-1` / `--text-2` / `--text-3` — primary, secondary, placeholder text
  - `--border` — all dividers and input borders
  - `--surface` — card backgrounds
  - `--radius-card` / `--radius-input` — border radii
- Page background: `body::before` (fixed dark property photo) + `body::after` (fixed dark gradient overlay). Cards float above via `box-shadow`. No `backdrop-filter` — it was removed for GPU performance.
- Collapsible panels (EMI calculator, Rental Yield): toggled via JS `display` on `.emi-wrap` / `.rb-wrap`. Both share `.panel-header` for shared styles; individual class names (`.emi-header`, `.rb-header`) are kept because JS targets them by class.

**Key JS design decisions:**
- `fmtPkr(amount)` mirrors backend `format_pkr` — same thresholds. Keep them in sync.
- Sqft unit is frontend-only: converts to Marla before API call (`val / 272.25`, unit = `'Marla'`). Backend only accepts Marla/Kanal.
- Location autocomplete: on focus fetches top 200 locations (filtered by selected city via `?city=`), on input debounces 280ms. Changing city clears the location field.
- `toTitleCase()` applies smart casing: roman numerals (I–XX) and abbreviations (DHA, PECHS, KDA, LDA, NHA, PAF, NFC, PIA, PTCL) stay ALL CAPS; hyphenated tokens like F-7 are treated as a single word.
- Price count-up animation (`animatePrice`) runs 700ms ease-out cubic on results reveal.
- Stale result banner appears when any form field changes after a result is showing; clears on next successful submit.
- EMI calculator and Rental Yield Analysis panels are only shown when purpose = "For Sale".
- Rental Yield Analysis fetches a second prediction with `purpose = 'For Rent'` on demand (lazy, cached in `rbLoaded`).

## Project Structure

```
Pakistan Property Price/
├── CLAUDE.md                  ← Claude Code instructions (this file)
├── PRD.md                     ← Product requirements document
├── data/                      ← All datasets
│   ├── Pakistan House Prices and Property Listings.csv  (raw Zameen.com data)
│   ├── data_cleaned.csv       (cleaned output from the notebook)
│   └── location_lookup.csv    (neighbourhood → lat/lon + city; source copy)
├── model/                     ← Trained model output (source copy from notebook)
│   └── model_artifacts.pkl    (copy this to backend/ after retraining)
├── notebooks/                 ← Jupyter training notebook
│   └── Pakistan Price.ipynb
├── backend/                   ← FastAPI server (self-contained; has its own working copies)
│   ├── main.py
│   ├── inference.py
│   ├── model_artifacts.pkl    (working copy used by the server)
│   ├── location_lookup.csv    (working copy used by the server)
│   └── requirements.txt
└── frontend/                  ← Single-page web app
    ├── index.html
    ├── hero.jpeg              (background photo used by the app)
    └── pakistan house.jpeg    (alternate/unused background photo)
```

## Running the Notebook

**Google Colab (preferred):** Upload `notebooks/Pakistan Price.ipynb` and `data/Pakistan House Prices and Property Listings.csv`. Cell 1 installs all dependencies. Run all cells top to bottom.

**Local:** `conda install -c conda-forge lightgbm xgboost catboost` then launch Jupyter.

**scikit-learn compatibility:** Use `np.sqrt(mean_squared_error(...))` — the `squared=False` param was removed in sklearn 1.4.

## Model Architecture

Four LightGBM models: one RMSE model (evaluation only) + three quantile models (P10, P50, P90). Target is `log1p(price)`; convert back with `np.expm1()` for user-facing output.

**13 features:**
```python
FEATURE_COLS = [
    'city_code', 'property_type_code', 'purpose_code',
    'baths', 'bedrooms',          # NaN allowed — LightGBM handles natively
    'log_area_sqft',              # log1p(area_sqft)
    'latitude', 'longitude',
    'is_premium_location',        # binary: DHA/Bahria/Gulberg/Clifton etc.
    'dist_to_center',             # Haversine km to city commercial centre
    'location_median_price',      # target-encoded: median log-price per neighbourhood
    'location_price_per_sqft',    # target-encoded: median PKR/sqft per neighbourhood
    'lat_lon_cluster',            # KMeans zone (50 clusters, fit on training lat/lon only)
]
```

**Critical decisions:**
- Train/test split is **time-based** (train < 2019-07-01, test = July 2019). Never random-split.
- IQR outlier removal runs **per purpose** separately — For Sale and For Rent have different price scales.
- `price_per_sqft` is EDA only — contains the target (leakage). Use target-encoded neighbourhood versions instead.
- Farm House excluded: sparse, high area variance, unreliable MAPE. Not in the frontend dropdown.
- `year_added`/`month_added` excluded: model cannot extrapolate to 2026+.

## Serializing a New Model

Run the serialization cell at the bottom of `notebooks/Pakistan Price.ipynb`. It saves `model_artifacts.pkl` — move it to `model/` (source copy) and also copy it to `backend/` before restarting the server. The artifact bundle contains: `model_p10/p50/p90/rmse`, `kmeans`, `loc_median`, `city_median`, `global_median`, `loc_ppq`, `city_ppq`, `global_ppq`, `city_enc`, `property_type_enc`, `purpose_enc`, `FEATURE_COLS`, `CAT_FEATURES`, `CITY_CENTERS`, `PREMIUM_KEYWORDS`.
