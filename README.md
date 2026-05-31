# Pakistan Property Price Estimator

**AI-powered property price estimates for Pakistan's residential market.**  
Built with LightGBM, FastAPI, and a zero-dependency single-page frontend.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-00644D?style=for-the-badge&logo=render&logoColor=white)](https://pakistan-property-price.onrender.com)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://pakistan-property-price.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io)

---

## Live Demo

> **[pakistan-property-price.onrender.com](https://pakistan-property-price.onrender.com)**

> Note: The app runs on Render's free tier and may take ~50 seconds to wake up after a period of inactivity. Subsequent requests are instant.

---

## What It Does

Enter a city, property type, area, and neighbourhood — get an instant AI-powered price estimate with a confidence range, per-sqft breakdown, and comparison against the city median.

**Features:**
- Price estimate with P10 / P50 / P90 confidence range
- Area input in Marla, Kanal, or Sqft
- Location autocomplete with 10,000+ neighbourhoods across 5 cities
- **2026 inflation-adjusted prices** — toggle between 2019 baseline and 2026 estimate with one tap (see below)
- Monthly payment (EMI) calculator for sale properties — updates with the toggle
- Rental yield analysis — fetches a live rent estimate and computes gross yield
- Compare up to 3 locations side-by-side — prices update with the toggle
- WhatsApp share button
- Fully responsive — works on mobile and desktop

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM (3 quantile models: P10, P50, P90) |
| Training Data | 120,000+ Zameen.com listings (2018–2019) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Deployment | Render (single Web Service — serves API and frontend) |

---

## Model Details

Four LightGBM models trained on log-transformed prices:

- **P10 / P50 / P90** — quantile regression for a calibrated price range
- **13 features** including area, bedrooms, baths, location target-encoding, KMeans geo-cluster, and distance to city centre
- **Time-based train/test split** — trained on pre-July 2019, tested on July 2019
- **Fuzzy location matching** via rapidfuzz — handles typos and partial names

Cities covered: **Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad**

---

## Run Locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

**Frontend**

Just open `frontend/index.html` directly in your browser. No server needed.

---

## Project Structure

```
├── backend/          FastAPI server + model artifacts (self-contained)
├── frontend/         Single-page web app (index.html + hero image)
├── notebooks/        Jupyter training notebook
├── data/             Source datasets and location lookup table
├── model/            Trained model artifacts (source copy)
└── CLAUDE.md         AI assistant instructions for this codebase
```

---

## 2026 Inflation Adjustment — Why ×2.9?

The model was trained on 2018–2019 data, so all raw predictions are in **2019 PKR values**. Pakistan experienced severe inflation in the years that followed. The ×2.9 multiplier is derived from Pakistan's cumulative CPI (Consumer Price Index) from 2019 to mid-2026:

| Year | Annual CPI Inflation |
|------|---------------------|
| 2019 | ~7% |
| 2020 | ~11% |
| 2021 | ~12% |
| 2022 | ~21% |
| 2023 | ~29% (peaked ~38%) |
| 2024 | ~23% (declining) |
| 2025 | ~10% (declining) |
| **Cumulative 2019 → mid-2026** | **~2.9×** |

Multiplying a 2019 prediction by 2.9 gives a rough estimate of what that property would be listed at in **today's PKR**. When the toggle is switched to **2026 Est.**, all prices update simultaneously — the main estimate, low/high range, per-sqft, per-Marla, EMI monthly payment, rental yield monthly rent, and the location comparison chart.

**Important caveats:**
- This is a CPI-based adjustment, not a property price index. Real estate in premium areas (DHA, Bahria) often tracks USD/PKR exchange rates more closely than CPI, meaning actual appreciation in those areas may be higher than 2.9×.
- The rental yield percentage is unaffected by the toggle because both sale price and rent scale by the same factor — the ratio stays constant.
- This adjustment is indicative only. Always verify with a local property agent for current market prices.

---

## Disclaimer

Estimates are based on 2018–2019 Zameen.com listing prices and reflect listed prices, not actual transaction prices. Current market values may differ significantly due to PKR devaluation and market changes since 2019.
