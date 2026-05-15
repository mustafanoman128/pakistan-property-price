# Pakistan Property Price Estimator

**AI-powered property price estimates for Pakistan's residential market.**  
Built with LightGBM, FastAPI, and a zero-dependency single-page frontend.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-00644D?style=for-the-badge&logo=render&logoColor=white)](https://pakistan-property-price-1.onrender.com)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://pakistan-property-price.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io)

---

## Live Demo

> **[pakistan-property-price-1.onrender.com](https://pakistan-property-price-1.onrender.com)**

> Note: The backend runs on Render's free tier and may take ~50 seconds to wake up after a period of inactivity. Subsequent requests are instant.

---

## What It Does

Enter a city, property type, area, and neighbourhood — get an instant AI-powered price estimate with a confidence range, per-sqft breakdown, and comparison against the city median.

**Features:**
- Price estimate with P10 / P50 / P90 confidence range
- Area input in Marla, Kanal, or Sqft
- Location autocomplete with 10,000+ neighbourhoods across 5 cities
- Monthly payment (EMI) calculator for sale properties
- Rental yield analysis — fetches a live rent estimate and computes gross yield
- Compare up to 3 locations side-by-side
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
| Deployment | Render (Web Service + Static Site) |

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

## Disclaimer

Estimates are based on 2018–2019 Zameen.com listing prices and reflect listed prices, not actual transaction prices. Current market values may differ significantly due to PKR devaluation and market changes since 2019.
