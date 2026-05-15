from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
import inference

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

@app.get('/')
def root():
    return {
        'name':      'Pakistan Property Price API',
        'version':   '1.0.0',
        'endpoints': ['/predict', '/locations', '/options'],
    }


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
def locations(q: str = '', city: str = ''):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
