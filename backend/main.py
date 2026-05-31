import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import Optional
import os
import inference

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pakistan Property Price API",
    description=(
        "Predicts residential property prices using a LightGBM model "
        "trained on 2018–2019 Zameen.com listings."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

VALID_CITIES          = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad']
VALID_PROPERTY_TYPES  = ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'Penthouse', 'Room']
VALID_PURPOSES        = ['For Sale', 'For Rent']
VALID_AREA_UNITS      = ['Marla', 'Kanal']

DISCLAIMER = (
    "Estimates are based on 2018–2019 Zameen.com listing prices and reflect listed "
    "prices, not actual transaction prices. Current market values may differ "
    "significantly due to PKR devaluation and market changes since 2019."
)


# ── Request / Response schemas ────────────────────────────────────────────

class PredictRequest(BaseModel):
    city:          str
    property_type: str
    purpose:       str
    area_value:    float
    area_unit:     str            # 'Marla' or 'Kanal'
    location:      str
    bedrooms:      Optional[float] = None
    baths:         Optional[float] = None

    @field_validator('city')
    @classmethod
    def validate_city(cls, v):
        v = v.strip().title()
        if v not in VALID_CITIES:
            raise ValueError(f'Must be one of: {VALID_CITIES}')
        return v

    @field_validator('property_type')
    @classmethod
    def validate_property_type(cls, v):
        v = v.strip().title()
        if v not in VALID_PROPERTY_TYPES:
            raise ValueError(f'Must be one of: {VALID_PROPERTY_TYPES}')
        return v

    @field_validator('purpose')
    @classmethod
    def validate_purpose(cls, v):
        v = v.strip().title()
        if v not in VALID_PURPOSES:
            raise ValueError(f'Must be one of: {VALID_PURPOSES}')
        return v

    @field_validator('area_unit')
    @classmethod
    def validate_area_unit(cls, v):
        v = v.strip().title()
        if v not in VALID_AREA_UNITS:
            raise ValueError('Must be Marla or Kanal')
        return v

    @field_validator('area_value')
    @classmethod
    def validate_area_value(cls, v):
        if v <= 0:
            raise ValueError('Must be a positive number')
        if v > 10_000:
            raise ValueError('Exceeds maximum area (10,000 Marla / Kanal)')
        return v

    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        if len(v.strip()) > 200:
            raise ValueError('Location name too long (max 200 characters)')
        return v

    @field_validator('bedrooms', 'baths', mode='before')
    @classmethod
    def validate_rooms(cls, v):
        if v is not None:
            v = float(v)
            if not (1 <= v <= 20):
                raise ValueError('Must be between 1 and 20')
        return v


class PredictResponse(BaseModel):
    p10:              float
    p50:              float
    p90:              float
    p10_formatted:    str
    p50_formatted:    str
    p90_formatted:    str
    confidence:       str
    location_matched: str
    latitude:         float
    longitude:        float
    disclaimer:       str
    city_median_pkr:  float


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get('/options')
def options():
    """Return valid dropdown values for the frontend form."""
    return {
        'cities':         VALID_CITIES,
        'property_types': VALID_PROPERTY_TYPES,
        'purposes':       VALID_PURPOSES,
        'area_units':     VALID_AREA_UNITS,
    }


@app.get('/locations')
def locations(
    q:    str = Query(default='', max_length=200),
    city: str = Query(default='', max_length=50),
):
    """
    Return locations for autocomplete, optionally filtered by city.
    With ?q=<query>: substring-filtered, max 50 results.
    Without query: top 200 locations by listing count.
    """
    locs = inference.get_locations(city=city)
    if q:
        q_lower = q.strip().lower()
        locs = [l for l in locs if q_lower in l['location']][:50]
    else:
        locs = locs[:200]
    return locs


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict property price range (P10 / P50 / P90) given property attributes.
    Returns estimates in raw PKR and human-readable Lakh/Crore format.
    """
    try:
        result = inference.predict(
            city          = req.city,
            property_type = req.property_type,
            purpose       = req.purpose,
            area_value    = req.area_value,
            area_unit     = req.area_unit,
            location      = req.location,
            bedrooms      = req.bedrooms,
            baths         = req.baths,
        )
        result['disclaimer'] = DISCLAIMER
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in /predict for city=%s location=%s", req.city, req.location)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")
