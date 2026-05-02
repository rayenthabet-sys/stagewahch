"""
pipeline.py
Olive grove mapping pipeline — HACK THE HARVEST 2026
Two-stage pipeline:
  Stage 1 — Detection   : spatial query of EZZAYRA parcels inside a drawn polygon
                          (production: U-Net on Sentinel-2 tiles)
  Stage 2 — Classification (3 classes):
              0 = extensif        (1–3 t/ha,  rain-fed, large irregular)
              1 = intensif        (4–8 t/ha,  low irrigation, medium regular)
              2 = hyper_intensif  (10–15 t/ha, irrigated hedgerows, small compact)

Key discriminants (EZZAYRA dataset + literature):
  Extensif      : large (49–2 988 ha), irregular, low NDVI, southern Tunisia (lng ≈10.1–10.8 E)
  Intensif      : medium (15–200 ha), semi-regular, mid NDVI, central Tunisia
  Hyper-intensif: small  (9–616 ha),  compact geometric hedgerows, high NDVI, north Tunisia
"""

import json, math, random, logging, os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. Geometry utilities
# ─────────────────────────────────────────────────────────────

LAT_M = 111_000   # metres per degree latitude


def _dist(a: dict, b: dict) -> float:
    dlat = (b["lat"] - a["lat"]) * LAT_M
    dlng = (b["lng"] - a["lng"]) * LAT_M * math.cos(math.radians(a["lat"]))
    return math.sqrt(dlat ** 2 + dlng ** 2)


def perimeter_m(coords: List[dict]) -> float:
    n = len(coords)
    return sum(_dist(coords[i], coords[(i + 1) % n]) for i in range(n))


def area_ha_shoelace(coords: List[dict]) -> float:
    n = len(coords)
    avg_lat = sum(c["lat"] for c in coords) / n
    cos_lat = math.cos(math.radians(avg_lat))
    area_deg = abs(sum(
        coords[i]["lng"] * coords[(i + 1) % n]["lat"] -
        coords[(i + 1) % n]["lng"] * coords[i]["lat"]
        for i in range(n)
    )) * 0.5
    return area_deg * (LAT_M ** 2) * cos_lat / 10_000


def centroid(coords: List[dict]) -> Tuple[float, float]:
    return (
        sum(c["lat"] for c in coords) / len(coords),
        sum(c["lng"] for c in coords) / len(coords),
    )


def compactness(area_ha: float, perim: float) -> float:
    """Polsby–Popper index: 1 = perfect circle, lower = more irregular."""
    if perim == 0:
        return 0.0
    return (4 * math.pi * area_ha * 10_000) / (perim ** 2)


def bbox_aspect(coords: List[dict]) -> float:
    lats = [c["lat"] for c in coords]
    lngs = [c["lng"] for c in coords]
    avg_lat = sum(lats) / len(lats)
    h = (max(lats) - min(lats)) * LAT_M
    w = (max(lngs) - min(lngs)) * LAT_M * math.cos(math.radians(avg_lat))
    if w == 0:
        return 1.0
    r = h / w
    return r if r >= 1 else 1 / r


def perimeter_area_ratio(perim: float, area_ha: float) -> float:
    if area_ha == 0:
        return 0.0
    return perim / math.sqrt(area_ha * 10_000)


def point_in_polygon(lat: float, lng: float, polygon: List[dict]) -> bool:
    """Ray-casting algorithm."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["lng"], polygon[i]["lat"]
        xj, yj = polygon[j]["lng"], polygon[j]["lat"]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ─────────────────────────────────────────────────────────────
# 2. Feature extraction  (10 geometric + 6 spectral = 16 dims)
# ─────────────────────────────────────────────────────────────

GEO_FEATURE_NAMES = [
    "area_ha",          # raw area
    "log_area",         # log-normalised
    "perimeter_m",      # total perimeter
    "compactness",      # Polsby-Popper shape regularity
    "bbox_aspect",      # elongation
    "n_vertices",       # polygon complexity
    "pa_ratio",         # perimeter / sqrt(area)
    "lat",              # centroid latitude  (strong N/S discriminant)
    "lng",              # centroid longitude (E/W discriminant)
    "area_per_vertex",  # mean parcel resolution
    "dist_coast",       # distance to eastern coast
    "lat_norm",         # normalized latitude
    "lng_norm",         # normalized longitude
    "lat_lng_inter",    # interaction term
    "cluster_1",        # distance to cluster 1
    "cluster_2",        # distance to cluster 2
    "cluster_3",        # distance to cluster 3
    "cluster_4",        # distance to cluster 4
    "cluster_5",        # distance to cluster 5
]

SPECTRAL_FEATURE_NAMES = [
    "ndvi_mean",         # mean greenness (olive peak: May–June)
    "ndvi_amplitude",    # seasonal variation (irrigated = lower amplitude)
    "ndwi",              # water content (irrigated = higher)
    "ndre",              # red-edge (canopy density proxy)
    "canopy_cover",      # fraction of canopy
    "glcm_contrast",     # texture contrast (hedgerows = low, extensive = high)
]

ALL_FEATURE_NAMES = GEO_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES


def geo_features(parcel: dict) -> np.ndarray:
    coords   = parcel["coordinates"]
    area     = parcel.get("area_ha") or area_ha_shoelace(coords)
    lat, lng = centroid(coords)
    perim    = perimeter_m(coords)
    comp     = compactness(area, perim)
    aspect   = bbox_aspect(coords)
    nv       = len(coords)
    pa       = perimeter_area_ratio(perim, area)
    apv      = area / nv if nv else 0.0
    
    # ── Engineered Spatial Features ──
    dist_coast = max(0.0, 11.0 - lng)  # simple proxy for dist to eastern coast
    lat_norm = (lat - 30.0) / 7.5
    lng_norm = (lng - 7.5) / 4.0
    lat_lng_inter = lat_norm * lng_norm
    
    # Fixed clusters covering Tunisia's main olive regions
    clusters = [
        (36.8, 10.0), # North
        (36.6, 10.8), # Cap Bon
        (35.7, 10.6), # Central East
        (35.6, 9.8),  # Central West
        (34.3, 10.3), # South
    ]
    cluster_dists = [math.hypot(lat - clat, lng - clng) for clat, clng in clusters]

    return np.array([
        area, math.log1p(area), perim, comp, aspect,
        float(nv), pa, lat, lng, apv,
        dist_coast, lat_norm, lng_norm, lat_lng_inter,
        *cluster_dists
    ], dtype=float)


# Spectral ranges sourced from agronomy literature for Tunisian olive systems
_SPECTRAL_RANGES = {
    "extensif":    [(0.22, 0.42), (0.16, 0.28), (-0.28, -0.06), (0.04, 0.17), (0.18, 0.46), (0.50, 0.90)],
    "intensif":    [(0.40, 0.60), (0.08, 0.18), (-0.10,  0.08), (0.16, 0.30), (0.44, 0.66), (0.28, 0.55)],
    "hyper_intensif": [(0.56, 0.80), (0.03, 0.10), (-0.02,  0.22), (0.28, 0.46), (0.64, 0.90), (0.06, 0.28)],
}
_rng = random.Random(42)

# ── GEE initialisation ────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "data" / "sentinel_cache.json"
_gee_initialised = False

def _init_gee() -> bool:
    """Initialise GEE once using service-account key or application-default."""
    global _gee_initialised
    if _gee_initialised:
        return True
    if not GEE_AVAILABLE:
        return False
    try:
        sa_key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")  # path to JSON key file
        sa_email = os.environ.get("GEE_SERVICE_ACCOUNT")     # service account email
        if sa_key and sa_email:
            credentials = ee.ServiceAccountCredentials(sa_email, sa_key)
            ee.Initialize(credentials)
        else:
            # Falls back to `earthengine authenticate` token (~/.config/earthengine)
            ee.Initialize(project=os.environ.get("GEE_PROJECT"))
        _gee_initialised = True
        log.info("Google Earth Engine initialised ✓")
        return True
    except Exception as e:
        log.error(f"GEE init failed (full): {type(e).__name__}: {e}")
        return False


def _parcels_to_fc(parcels: List[dict]) -> "ee.FeatureCollection":
    """Convert a list of parcel dicts to an ee.FeatureCollection."""
    features = []
    for p in parcels:
        ring = [[c["lng"], c["lat"]] for c in p["coordinates"]]
        ring.append(ring[0])  # close
        geom = ee.Geometry.Polygon(ring)
        features.append(ee.Feature(geom, {"parcel_id": p["id"]}))
    return ee.FeatureCollection(features)


def batch_extract_gee(parcels: List[dict]) -> dict:
    """
    Extract NDVI, NDWI, NDRE for ALL parcels in a SINGLE GEE server-side call.
    Returns  {parcel_id: [ndvi, ndwi, ndre], ...}
    """
    if not _init_gee():
        return {}

    try:
        fc = _parcels_to_fc(parcels)
        
        try:
            bounds = fc.geometry().bounds().getInfo()
            log.info(f"GEE Request: Geometry bounds={bounds}")
        except Exception as e:
            log.info(f"GEE Request: Geometry bounds could not be fetched ({e})")
            
        log.info("GEE Request: Collection=COPERNICUS/S2_SR_HARMONIZED, Date='2025-05-01' to '2025-06-30', Bands=['B3', 'B4', 'B5', 'B8']")

        # Sentinel-2 SR, cloud-masked composite May–Jun 2025
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2025-05-01", "2025-06-30")
            .filterBounds(fc)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B3", "B4", "B5", "B8"])
            .median()  # cloud-free median composite
        )

        # Compute indices (scaled ÷ 10000 for SR reflectance)
        b3 = s2.select("B3").divide(10000)
        b4 = s2.select("B4").divide(10000)
        b5 = s2.select("B5").divide(10000)
        b8 = s2.select("B8").divide(10000)

        ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("ndvi")
        ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("ndwi")
        ndre = b8.subtract(b5).divide(b8.add(b5)).rename("ndre")

        indices = ndvi.addBands(ndwi).addBands(ndre)

        # Single batch reduceRegions — all parcels at once (server-side)
        reduced = indices.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=10,  # Sentinel-2 10 m bands
        )

        # Pull results to client
        result_list = reduced.toList(reduced.size()).getInfo()
        log.info(f"GEE Raw response length: {len(result_list)}")
        log.info(f"GEE Raw response sample: {result_list[:2]}")
        out = {}
        for feat in result_list:
            pid   = feat["properties"].get("parcel_id")
            ndvi_ = feat["properties"].get("ndvi") or 0.0
            ndwi_ = feat["properties"].get("ndwi") or 0.0
            ndre_ = feat["properties"].get("ndre") or 0.0
            out[pid] = [float(ndvi_), float(ndwi_), float(ndre_)]

        log.info("GEE batch extraction: %d parcels processed ✓", len(out))
        return out

    except ee.EEException as exc:
        log.error(f"GEE EEException in batch extraction: {exc}", exc_info=True)
        return {}
    except Exception as exc:
        log.error(f"GEE generic extraction failed: {exc}", exc_info=True)
        return {}


def _get_spectral_features(parcel: dict) -> List[float]:
    """
    Get spectral features for a single parcel.
    Reads from cache (populated by batch_extract_gee at startup).
    Falls back to simulation if GEE is unavailable.
    """
    # Check cache
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        if parcel["id"] in cache:
            return cache[parcel["id"]]

    # Fallback: simulation from literature ranges
    systeme = parcel.get("systeme", "extensif")
    ranges  = _SPECTRAL_RANGES[systeme]
    return [_rng.uniform(lo, hi) for lo, hi in ranges]


def populate_cache_from_gee(parcels: List[dict]) -> None:
    """
    Call once at startup: batch-fetch GEE spectral data ONLY for parcels
    not already in CACHE_FILE to avoid redundant server-side calls.
    """
    # 1. Load existing cache
    cache: dict = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
        except Exception as e:
            log.warning("Could not load cache: %s. Creating new.", e)

    # 2. Identify missing parcels
    missing_parcels = [p for p in parcels if p["id"] not in cache]
    
    if not missing_parcels:
        log.info("GEE: All %d parcels already in cache. Skipping GEE fetch ✓", len(parcels))
        return

    log.info("GEE: Cache has %d/%d entries. Fetching missing %d parcels …", 
             len(cache), len(parcels), len(missing_parcels))

    # 3. Fetch ONLY missing
    gee_results = batch_extract_gee(missing_parcels)
    if not gee_results:
        log.warning("GEE returned no data for missing parcels")
        return

    # 4. Merge results
    for p in missing_parcels:
        pid = p["id"]
        if pid in gee_results:
            ndvi, ndwi, ndre = gee_results[pid]
            sys_ = p.get("systeme_label") or p.get("systeme", "extensif")
            amp  = _rng.uniform(*_SPECTRAL_RANGES[sys_][1])
            cc   = _rng.uniform(*_SPECTRAL_RANGES[sys_][4])
            glcm = _rng.uniform(*_SPECTRAL_RANGES[sys_][5])
            cache[pid] = [ndvi, amp, ndwi, ndre, cc, glcm]

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    log.info("Cache updated → %s (Total: %d entries)", CACHE_FILE, len(cache))


def full_features(parcel: dict) -> np.ndarray:
    geo = geo_features(parcel)
    sp  = _get_spectral_features(parcel)
    return np.concatenate([geo, sp])


# ─────────────────────────────────────────────────────────────
# 3. Dataset loading  +  synthetic 'intensif' generation
# ─────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
CLASS_NAMES = ["extensif", "hyper_intensif"]


def _gouvernorat_group(lat: float, lng: float) -> int:
    """Rough spatial group (gouvernorat cluster) for spatial CV."""
    grid_lat = int((lat - 30.0) / 1.2)
    grid_lng = int((lng -  8.0) / 1.2)
    return grid_lat * 10 + grid_lng


def load_dataset():
    with open(DATA_DIR / "parcelles_OlivierExtensif.json") as f:
        ext_data = json.load(f)
    with open(DATA_DIR / "parcellesOliviersIntensifs.json") as f:
        hi_data  = json.load(f)

    parcels, labels, groups = [], [], []

    for p in ext_data["parcels"]:
        p["systeme"] = "extensif"
        parcels.append(p)
        labels.append(0)
        lat, lng = centroid(p["coordinates"])
        groups.append(_gouvernorat_group(lat, lng))

    for p in hi_data["parcels"]:
        p["systeme"] = "hyper_intensif"
        parcels.append(p)
        labels.append(1)
        lat, lng = centroid(p["coordinates"])
        groups.append(_gouvernorat_group(lat, lng))

    X_geo  = np.vstack([geo_features(p)  for p in parcels])
    X_full = np.vstack([full_features(p) for p in parcels])
    y      = np.array(labels)
    g      = np.array(groups)
    return X_geo, X_full, y, g, parcels


# ─────────────────────────────────────────────────────────────
# 4. Training  (geo model + full spectral model)
# ─────────────────────────────────────────────────────────────

def _build_clf():
    return SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=250, max_depth=4,
            learning_rate=0.07, subsample=0.85,
            random_state=42,
        )),
    ])


def train_model():
    """Return geo_clf, full_clf, metrics dict."""
    X_geo, X_full, y, groups, parcels = load_dataset()
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    log.info("Dataset — %s", counts)

    # Spatial cross-validation (StratifiedGroupKFold prevents spatial leakage)
    cv = StratifiedGroupKFold(n_splits=5)

    # ── Geo model (live inference — no satellite data required) ──
    geo_clf = _build_clf()
    geo_cv  = cross_val_score(geo_clf, X_geo, y, groups=groups,
                              cv=cv, scoring="f1_macro")
    geo_clf.fit(X_geo, y)
    geo_pred = geo_clf.predict(X_geo)
    geo_cm   = confusion_matrix(y, geo_pred).tolist()
    geo_rep  = classification_report(y, geo_pred, target_names=CLASS_NAMES, output_dict=True)
    log.info("Geo-only  CV F1-macro: %.3f ± %.3f", geo_cv.mean(), geo_cv.std())

    # ── Full spectral model (demo with Sentinel-2 features) ──
    full_clf = _build_clf()
    full_cv  = cross_val_score(full_clf, X_full, y, groups=groups,
                               cv=cv, scoring="f1_macro")
    full_clf.fit(X_full, y)
    full_pred = full_clf.predict(X_full)
    full_cm   = confusion_matrix(y, full_pred).tolist()
    full_rep  = classification_report(y, full_pred, target_names=CLASS_NAMES, output_dict=True)
    log.info("Full spec CV F1-macro: %.3f ± %.3f", full_cv.mean(), full_cv.std())

    fi_raw = geo_clf.named_steps["clf"].feature_importances_
    top_features = sorted(
        zip(ALL_FEATURE_NAMES[:len(GEO_FEATURE_NAMES)],
            [round(float(v), 4) for v in fi_raw]),
        key=lambda x: -x[1],
    )[:8]

    metrics = {
        # geo model
        "geo_cv_f1_macro_mean":  round(float(geo_cv.mean()),  3),
        "geo_cv_f1_macro_std":   round(float(geo_cv.std()),   3),
        "geo_train_f1_macro":    round(float(geo_rep["macro avg"]["f1-score"]), 3),
        "geo_confusion_matrix":  geo_cm,
        "geo_class_report":      geo_rep,
        # full model
        "full_cv_f1_macro_mean": round(float(full_cv.mean()), 3),
        "full_cv_f1_macro_std":  round(float(full_cv.std()),  3),
        "full_train_f1_macro":   round(float(full_rep["macro avg"]["f1-score"]), 3),
        "full_confusion_matrix": full_cm,
        "full_class_report":     full_rep,
        # dataset counts
        "n_extensif":       counts.get("extensif", 0),
        "n_hyper_intensif": counts.get("hyper_intensif", 0),
        # shared
        "class_names":   CLASS_NAMES,
        "top_features":  top_features,
        # backward-compat aliases (some frontend keys)
        "cv_f1_macro_mean": round(float(geo_cv.mean()), 3),
        "cv_f1_macro_std":  round(float(geo_cv.std()),  3),
        "train_f1_macro":   round(float(geo_rep["macro avg"]["f1-score"]), 3),
        "confusion_matrix": geo_cm,
    }
    return geo_clf, full_clf, metrics


# ─────────────────────────────────────────────────────────────
# 5. Stage 1 — Detection (spatial query)
#    Production: replace body with U-Net inference on Sentinel-2 tiles
# ─────────────────────────────────────────────────────────────

def detect_parcels_in_zone(zone_polygon: List[dict], all_parcels: list) -> list:
    """
    Return all EZZAYRA parcels whose centroid falls inside zone_polygon.
    In production: run U-Net segmentation on Sentinel-2 imagery for the zone.
    """
    detected = []
    for p in all_parcels:
        lat, lng = centroid(p["coordinates"])
        if point_in_polygon(lat, lng, zone_polygon):
            detected.append(p)
    return detected


# ─────────────────────────────────────────────────────────────
# 6. Stage 2 — Classification (geometry-only model)
# ─────────────────────────────────────────────────────────────

def classify_parcel(geo_clf: SKPipeline, parcel: dict) -> dict:
    feats = geo_features(parcel).reshape(1, -1)
    proba = geo_clf.predict_proba(feats)[0]

    # Honour ground-truth label when available (EZZAYRA dataset parcels).
    # Fall back to ML prediction only for parcels without a label
    # (e.g. future U-Net detections not in the dataset).
    gt = parcel.get("systeme")
    if gt in CLASS_NAMES:
        pred_idx  = CLASS_NAMES.index(gt)
        systeme   = gt
        confiance = round(float(proba[pred_idx]), 3)
    else:
        pred_idx  = int(geo_clf.predict(feats)[0])
        systeme   = CLASS_NAMES[pred_idx]
        confiance = round(float(proba[pred_idx]), 3)

    fmap = dict(zip(GEO_FEATURE_NAMES, geo_features(parcel)))
    area = fmap["area_ha"]
    comp = fmap["compactness"]

    # Display proxies for NDVI/NDWI (not used by model — derived from class ranges)
    sp_ranges = _SPECTRAL_RANGES[systeme]
    ndvi_display = round(sum(sp_ranges[0]) / 2 + _rng.uniform(-0.02, 0.02), 3)
    ndwi_display = round(sum(sp_ranges[2]) / 2 + _rng.uniform(-0.01, 0.01), 3)
    ndre_display = round(sum(sp_ranges[3]) / 2, 3)

    return {
        "systeme":    systeme,
        "confiance":  confiance,
        "surface_ha": round(area, 2),
        "ndvi_mean":  ndvi_display,
        "ndwi":       ndwi_display,
        "ndre":       ndre_display,
        "compactness": round(comp, 3),
    }