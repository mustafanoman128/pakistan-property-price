import logging
import joblib
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Load all artifacts once at module import ──────────────────────────────
# The backend drops model_artifacts.pkl and location_lookup.csv into this
# same directory. Path(__file__).parent always resolves correctly regardless
# of where uvicorn is launched from.

_DIR = Path(__file__).parent

try:
    # Trusted artifact: produced by our own training notebook (notebooks/Pakistan Price.ipynb).
    # Never replace this file with one from an untrusted source.
    _art = joblib.load(_DIR / 'model_artifacts.pkl')
except FileNotFoundError:
    raise RuntimeError(
        f"model_artifacts.pkl not found in {_DIR}. "
        "Copy it from model/ into backend/ before starting the server."
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model_artifacts.pkl: {e}") from e

_REQUIRED_KEYS = [
    'model_p10', 'model_p50', 'model_p90', 'kmeans',
    'loc_median', 'city_median', 'global_median',
    'loc_ppq', 'city_ppq', 'global_ppq',
    'city_enc', 'property_type_enc', 'purpose_enc',
    'FEATURE_COLS', 'CITY_CENTERS', 'PREMIUM_KEYWORDS',
]
_missing = [k for k in _REQUIRED_KEYS if k not in _art]
if _missing:
    raise RuntimeError(
        f"model_artifacts.pkl is missing keys: {_missing}. "
        "Re-run the serialization cell in the training notebook."
    )

model_p10  = _art['model_p10']
model_p50  = _art['model_p50']
model_p90  = _art['model_p90']

kmeans = _art['kmeans']

loc_median    = _art['loc_median']     # Series: location → median log-price
city_median   = _art['city_median']    # Series: city → median log-price
global_median = _art['global_median']  # float: last-resort fallback

loc_ppq    = _art['loc_ppq']           # Series: location → median PKR/sqft
city_ppq   = _art['city_ppq']          # Series: city → median PKR/sqft
global_ppq = _art['global_ppq']        # float

city_enc          = _art['city_enc']
property_type_enc = _art['property_type_enc']
purpose_enc       = _art['purpose_enc']

FEATURE_COLS     = _art['FEATURE_COLS']
CITY_CENTERS     = _art['CITY_CENTERS']
PREMIUM_KEYWORDS = _art['PREMIUM_KEYWORDS']

# Location lookup table (for lat/lon resolution and autocomplete)
try:
    _loc_df = pd.read_csv(_DIR / 'location_lookup.csv')
except FileNotFoundError:
    raise RuntimeError(
        f"location_lookup.csv not found in {_DIR}. "
        "Copy it from data/ into backend/ before starting the server."
    )

_bad_mask = _loc_df['mean_lat'].isna() | _loc_df['mean_lon'].isna()
if _bad_mask.any():
    raise RuntimeError(
        f"location_lookup.csv has {_bad_mask.sum()} rows with missing lat/lon: "
        f"{_loc_df.loc[_bad_mask, 'location'].tolist()[:5]}"
    )

_loc_df['location'] = _loc_df['location'].str.lower().str.strip()
_location_names = _loc_df['location'].tolist()

MARLA_TO_SQFT = 272.25
KANAL_TO_SQFT = 5445.0


# ── Helpers ───────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _nearest_city(lat: float, lon: float) -> str:
    return min(CITY_CENTERS.keys(), key=lambda c: _haversine_km(lat, lon, *CITY_CENTERS[c]))

# Assign each location to its nearest city centre (computed once at startup)
_loc_df['city'] = _loc_df.apply(
    lambda r: _nearest_city(r['mean_lat'], r['mean_lon']), axis=1
)


def _fuzzy_match(query: str) -> Optional[str]:
    """Return best-matching canonical location name, or None if score < 60."""
    try:
        from rapidfuzz import process
    except ImportError:
        return query if query in _location_names else None
    result = process.extractOne(query, _location_names, score_cutoff=60)
    return result[0] if result else None


def _resolve_lat_lon(canonical: Optional[str], city: str):
    """Return (lat, lon) from location_lookup, falling back to city centre."""
    if canonical:
        row = _loc_df[_loc_df['location'] == canonical]
        if not row.empty:
            return float(row.iloc[0]['mean_lat']), float(row.iloc[0]['mean_lon'])
    city_center = CITY_CENTERS.get(city)
    if city_center is None:
        raise ValueError(
            f"No city centre coordinates configured for '{city}'. "
            "Add it to CITY_CENTERS in the model artifacts."
        )
    return city_center


def _encode(enc: dict, label: str, field_name: str) -> int:
    """Encode a categorical label; raises ValueError on unknown values."""
    code = enc.get(label)
    if code is None:
        raise ValueError(
            f"Unknown {field_name} '{label}'. Known values: {list(enc.keys())}"
        )
    return code


def format_pkr(amount: float) -> str:
    """Convert raw PKR to 'X.XX Crore', 'X.X Lakh', or plain number."""
    crore = amount / 1e7
    lakh  = amount / 1e5
    if crore >= 1: return f'{crore:.2f} Crore'
    if lakh  >= 1: return f'{lakh:.1f} Lakh'
    return f'{round(amount):,}'


def get_locations(city: str = '') -> list:
    """Return location names with listing counts — used for autocomplete."""
    df = _loc_df
    if city:
        df = df[df['city'] == city.strip().title()]
    return (
        df[['location', 'listing_count']]
        .sort_values('listing_count', ascending=False)
        .to_dict(orient='records')
    )


# ── Main prediction function ──────────────────────────────────────────────

def predict(
    city: str,
    property_type: str,
    purpose: str,
    area_value: float,
    area_unit: str,          # 'Marla' or 'Kanal'
    location: str,
    bedrooms: Optional[float] = None,
    baths: Optional[float] = None,
) -> dict:
    # ── Normalize text inputs ─────────────────────────────────────────────
    city          = city.strip().title()
    property_type = property_type.strip().title()
    purpose       = purpose.strip().title()
    location_norm = location.strip().lower()
    area_unit     = area_unit.strip().title()

    # ── Area → sqft → log-transform ───────────────────────────────────────
    sqft          = area_value * (MARLA_TO_SQFT if area_unit == 'Marla' else KANAL_TO_SQFT)
    log_area_sqft = np.log1p(sqft)

    # ── Location → lat/lon ────────────────────────────────────────────────
    canonical  = _fuzzy_match(location_norm)
    lat, lon   = _resolve_lat_lon(canonical, city)

    # ── Premium flag ──────────────────────────────────────────────────────
    is_premium = int(any(kw in location_norm for kw in PREMIUM_KEYWORDS))

    # ── Distance to city commercial centre ────────────────────────────────
    # CITY_CENTERS[city] is safe here: _resolve_lat_lon already raised for unknown cities.
    dist = _haversine_km(lat, lon, *CITY_CENTERS[city])

    # ── Target encoding — 3-tier fallback ────────────────────────────────
    lookup_key  = canonical if canonical else location_norm
    loc_price   = loc_median.get(lookup_key, city_median.get(city, global_median))
    loc_ppq_val = loc_ppq.get(lookup_key,    city_ppq.get(city,   global_ppq))

    # ── KMeans cluster ────────────────────────────────────────────────────
    cluster = int(kmeans.predict(np.array([[lat, lon]]))[0])

    # ── Categorical encoding ──────────────────────────────────────────────
    row = {
        'city_code':               _encode(city_enc,          city,          'city'),
        'property_type_code':      _encode(property_type_enc, property_type, 'property_type'),
        'purpose_code':            _encode(purpose_enc,       purpose,       'purpose'),
        'baths':                   float(baths) if baths is not None else np.nan,
        'bedrooms':                float(bedrooms) if bedrooms is not None else np.nan,
        'log_area_sqft':           log_area_sqft,
        'latitude':                lat,
        'longitude':               lon,
        'is_premium_location':     is_premium,
        'dist_to_center':          dist,
        'location_median_price':   loc_price,
        'location_price_per_sqft': loc_ppq_val,
        'lat_lon_cluster':         cluster,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]

    # ── Run quantile models ───────────────────────────────────────────────
    p10 = float(np.expm1(model_p10.predict(X)[0]))
    p50 = float(np.expm1(model_p50.predict(X)[0]))
    p90 = float(np.expm1(model_p90.predict(X)[0]))

    # Ensure ordering (quantile models can occasionally cross)
    if p10 > p50 or p90 < p50:
        logger.warning(
            "Quantile crossing for location='%s' city='%s': p10=%.0f p50=%.0f p90=%.0f — clamping.",
            canonical, city, p10, p50, p90,
        )
    p10, p90 = min(p10, p50), max(p90, p50)

    # ── Confidence indicator ──────────────────────────────────────────────
    ratio      = (p90 - p10) / max(p50, 1)
    confidence = 'High' if ratio < 0.5 else ('Moderate' if ratio < 1.0 else 'Low')

    return {
        'p10':             p10,
        'p50':             p50,
        'p90':             p90,
        'p10_formatted':   format_pkr(p10),
        'p50_formatted':   format_pkr(p50),
        'p90_formatted':   format_pkr(p90),
        'confidence':      confidence,
        'location_matched': canonical or location_norm,
        'latitude':        lat,
        'longitude':       lon,
        'city_median_pkr': float(np.expm1(city_median.get(city, global_median))),
    }
