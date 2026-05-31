# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

End-to-end property price estimator — fully deployed as a single Render Web Service:
- ML model trained on 120k+ Zameen.com listings (2018–2019)
- Single service (API + frontend) → `https://pakistan-property-price.onrender.com`
- GitHub → `https://github.com/mustafanoman128/pakistan-property-price`

The FastAPI backend serves the frontend from `backend/static/`. The root route (`GET /`) returns `backend/static/index.html`; assets are served under `/static/`.

## Running the Backend

```powershell
cd backend
python -m uvicorn main:app --reload
```

`pip` and `uvicorn` are not on PATH on this machine — always use `python -m` prefix. Backend runs at `http://127.0.0.1:8000`.

**Local dependency install:** `python -m pip install -r requirements.txt` (run from `backend/`).

**Local dev API switch:** `frontend/index.html` line ~1066 has `const API = 'https://pakistan-property-price.onrender.com'`. Change to `http://127.0.0.1:8000` for local testing; revert before pushing.

**Deployment workflow:** `frontend/index.html` is the source file — edit this one. Before pushing to Render, copy it to `backend/static/index.html`. Render serves the copy; the original in `frontend/` is for local development.

## Backend Architecture

| File | Role |
|------|------|
| `main.py` | FastAPI app. Three endpoints: `GET /options`, `GET /locations?q=&city=`, `POST /predict` |
| `inference.py` | Loads all artifacts at import time, exposes `predict()` and `get_locations()` |
| `model_artifacts.pkl` | Serialized model + encoders + lookup dicts (joblib, compress=3) |
| `location_lookup.csv` | Neighbourhood → mean lat/lon + listing count + assigned city |
| `requirements.txt` | fastapi, uvicorn, lightgbm, pandas, numpy, scikit-learn, joblib, rapidfuzz |

**Request flow:** `POST /predict` → Pydantic validates → `inference.predict()` → fuzzy-matches location name → builds 13-feature row → runs 3 quantile models → returns p10/p50/p90 + formatted strings + confidence + lat/lon. The `city_median_pkr` field in the response is the city-level median in raw PKR; the frontend uses it to render an "X% above/below city median" badge in the results panel.

**API docs:** FastAPI auto-generates interactive docs at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc`.

**Key inference details:**
- `_fuzzy_match()` uses rapidfuzz with `score_cutoff=60`. Falls back to exact match if rapidfuzz not installed.
- Location fallback chain: neighbourhood median → city median → global median (for both `location_median_price` and `location_price_per_sqft`).
- `None` bedrooms/baths → `np.nan` (not `None`) before building the DataFrame — pandas creates object dtype from None, which LightGBM rejects.
- City assignment on `location_lookup.csv` is computed at startup via nearest city centre (Haversine). Used to filter autocomplete results by city.
- `format_pkr()` in inference.py: ≥1 Crore → "X.XX Crore", ≥1 Lakh → "X.X Lakh", else plain `{n:,}`.

## Frontend Architecture

Single file: `frontend/index.html`. No build step, no dependencies, open directly in browser.

**CSS design system — do not bypass these:**
- Two fonts: `Playfair Display` (display/headings, `--font-display`) for prices and the page title; `Outfit` (body/UI, `--font-body`) for all other text. Do not add any other family.
- All colours are CSS tokens in `:root`. Never add hardcoded hex values — always use or extend the token set:
  - `--accent` / `--accent-dark` / `--accent-dim` — brand green and its tints
  - `--gold` / `--gold-dim` — warm copper accent (used for the 2026 inflation pill active state)
  - `--text-1` / `--text-2` / `--text-3` — primary, secondary, placeholder text
  - `--border` — all dividers and input borders
  - `--surface` — card backgrounds
  - `--radius-card: 20px` / `--radius-input: 10px` — border radii
- Page background: `body::before` (fixed dark property photo, base64-inlined) + `body::after` (fixed dark gradient overlay). Cards float above via deep `box-shadow`.
- Cards use `backdrop-filter: blur(32px) saturate(200%)` with `background: rgba(255,255,255,0.76)` and a `5px` green `border-top` plus an inset white highlight. Do not remove the backdrop-filter — it was deliberately re-added for visual quality.
- Collapsible panels (EMI calculator, Rental Yield): both share `.panel-header` / `.panel-title` / `.panel-chevron` / `.panel-body` CSS classes. JS targets them by ID (`emi-body`, `rb-body`, `emi-chevron`, `rb-chevron`).

**Key JS state variables:**
- `mainResult` — raw API response from the last `/predict` call
- `lastPayload` — last form payload, used to fetch comparison/rental predictions
- `lastAreaSqft` — area in sqft at time of last prediction, used by `renderPrices()`
- `inflationAdjusted` — boolean; drives the 2026 toggle
- `INFLATION_FACTOR = 2.9` — cumulative Pakistan CPI 2019→mid-2026
- `rentRawP50` — raw rent p50 from the rental yield fetch; 0 until loaded
- `rbLoaded` — boolean; true once the rental yield fetch has completed for the current result (prevents duplicate fetches)
- `compareItems` — raw array of `{ label, p50, formatted }` from last comparison run; null until compared

**Key JS functions:**
- `fmtPkr(amount)` — mirrors backend `format_pkr`. Keep thresholds in sync.
- `renderPrices(skipP50)` — single source of truth for all displayed price values. Applies `INFLATION_FACTOR` when `inflationAdjusted` is true. Also re-renders the comparison chart and recalculates EMI.
- `setInflation(adjusted)` — toggles `inflationAdjusted`, syncs pill button UI, calls `renderPrices()`.
- `animatePrice(el, targetRaw, targetFormatted)` — 700ms ease-out count-up on results reveal. Called once in `showResults`; `renderPrices` handles all subsequent updates without animation.
- `renderHBarChart(items)` — reads `inflationAdjusted` internally to apply the factor to displayed values. Bar widths are ratio-based so they're unaffected by the multiplier.
- `syncSelectColor(sel)` — adds `.has-value` class to `<select>` elements when a value is chosen, turning selected text green via CSS.
- `toTitleCase()` — smart casing: roman numerals (I–XX) and abbreviations (DHA, PECHS, KDA, LDA, NHA, PAF, NFC, PIA, PTCL) stay ALL CAPS.

**Other JS patterns:**
- Sqft unit is frontend-only: converts to Marla before API call (`val / 272.25`, unit = `'Marla'`). Backend only accepts Marla/Kanal.
- Location autocomplete: on focus fetches top 200 locations (filtered by city), on input debounces 280ms. Changing city clears location field.
- Stale result banner appears on any form change after a result is shown; clears on next successful submit.
- EMI and Rental Yield panels are only shown when purpose = "For Sale". Rental yield fetches a second `/predict` with `purpose = 'For Rent'` on demand (lazy, cached in `rbLoaded`). Annual yield % is unaffected by the inflation toggle because both sale and rent prices scale by the same factor.
- `showResults()` always resets `inflationAdjusted = false`, `rentRawP50 = 0`, and `compareItems = null` — every new prediction starts from the 2019 baseline.

## Project Structure

```
Pakistan Property Price/
├── CLAUDE.md                  ← Claude Code instructions (this file)
├── PRD.md                     ← Product requirements document
├── data/                      ← Source datasets (large CSVs excluded from git via .gitignore)
│   ├── Pakistan House Prices and Property Listings.csv  (raw; gitignored)
│   ├── data_cleaned.csv       (processed; gitignored)
│   └── location_lookup.csv    (source copy; committed)
├── model/                     ← Trained model output (source copy from notebook)
│   └── model_artifacts.pkl    (copy this to backend/ after retraining)
├── notebooks/                 ← Jupyter training notebook
│   └── Pakistan Price.ipynb
├── backend/                   ← FastAPI server (self-contained working copies)
│   ├── main.py
│   ├── inference.py
│   ├── model_artifacts.pkl
│   ├── location_lookup.csv
│   ├── requirements.txt
│   └── static/                ← Deployed frontend (copy of frontend/index.html)
│       └── index.html
└── frontend/                  ← Source frontend — edit here, then copy to backend/static/
    ├── index.html
    └── hero.jpeg              (background photo, also base64-inlined in CSS)
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
