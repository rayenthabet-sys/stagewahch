"""
main.py — FastAPI backend for olive grove mapping demo.
Run with:  uvicorn main:app --reload --port 8000

Stage 1: Detection  — spatial query of EZZAYRA parcels inside drawn polygon
                       (production: U-Net on Sentinel-2 tiles)
Stage 2: Classification — 3-class model (extensif / intensif / hyper_intensif)
"""

import json, logging, math, time, pickle, requests
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import (
    train_model, classify_parcel, detect_parcels_in_zone,
    CLASS_NAMES, centroid, area_ha_shoelace, populate_cache_from_gee
)

MODEL_CACHE = Path(__file__).parent / "data" / "trained_model.pkl"
UNET_WEIGHTS = Path(__file__).parent / "unet_olive.pth"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Olive Grove Mapping API",
    description="Cartographie intelligente des oliveraies tunisiennes — Hack The Harvest 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# ─── Global model state ──────────────────────────────────────────────────────

_model      = None   # geo-only model (live inference)
_full_model = None   # geo + spectral model (demo metric)
_metrics    = None
_unet       = None   # U-Net segmentation model (loaded once, reused)
_all_parcels: List[dict] = []   # all EZZAYRA parcels (ext + int + HI)


@app.on_event("startup")
def startup():
    global _model, _full_model, _metrics, _all_parcels, _unet
    t0 = time.time()

    # ── Load or train the sklearn classifiers ─────────────────────────────
    if MODEL_CACHE.exists():
        log.info("Loading cached classifiers from %s …", MODEL_CACHE)
        with open(MODEL_CACHE, "rb") as f:
            _model, _full_model, _metrics = pickle.load(f)
        log.info("Classifiers loaded in %.2f s (cached)", time.time() - t0)
    else:
        log.info("Training 3-class model on EZZAYRA dataset (first run) …")
        _model, _full_model, _metrics = train_model()
        MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_CACHE, "wb") as f:
            pickle.dump((_model, _full_model, _metrics), f)
        log.info(
            "Training done in %.2f s — model saved to cache. "
            "Future startups will be instant.",
            time.time() - t0,
        )
    log.info(
        "CV F1-macro: %.3f (geo) / %.3f (full)",
        _metrics["geo_cv_f1_macro_mean"],
        _metrics["full_cv_f1_macro_mean"],
    )

    # ── Load U-Net weights (if trained) ───────────────────────────────────
    if UNET_WEIGHTS.exists():
        try:
            from unet_model import build_model
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _unet = build_model().to(device)
            _unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=device))
            _unet.eval()
            log.info("U-Net loaded from %s (device: %s) ✓", UNET_WEIGHTS, device)
        except Exception as exc:
            log.warning("Could not load U-Net: %s", exc)
    else:
        log.info("No U-Net weights found at %s — using EZZAYRA lookup.", UNET_WEIGHTS)

    # Pre-load EZZAYRA parcels (extensif + hyper_intensif ground truth)
    data_dir = Path(__file__).parent / "data"
    for fname, systeme in [
        ("parcelles_OlivierExtensif.json",   "extensif"),
        ("parcellesOliviersIntensifs.json",  "hyper_intensif"),
    ]:
        with open(data_dir / fname) as f:
            raw = json.load(f)
        for p in raw["parcels"]:
            p["systeme_label"] = systeme
            _all_parcels.append(p)
    log.info("Loaded %d EZZAYRA parcels", len(_all_parcels))

    # GEE: batch-fetch ALL spectral data in one call, persist to cache
    populate_cache_from_gee(_all_parcels)


def _fetch_osm_parcels(bbox: tuple) -> List[dict]:
    """Fetch 'orchard' polygons from OSM via Overpass API."""
    import requests
    from pipeline import area_ha_shoelace
    south, west, north, east = bbox
    query = f"""
    [out:json];
    way["landuse"="orchard"]({south},{west},{north},{east});
    out geom;
    """
    try:
        resp = requests.get(
            "https://overpass.kumi.systems/api/interpreter", 
            params={"data": query}, 
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        
        parcels = []
        for element in data.get("elements", []):
            if element.get("type") == "way":
                osm_id = element["id"]
                coords = [{"lat": node["lat"], "lng": node["lon"]} for node in element.get("geometry", [])]
                if len(coords) >= 3:
                    area_ha = area_ha_shoelace(coords)
                    parcels.append({
                        "id": f"osm_{osm_id}",
                        "coordinates": coords,
                        "area_ha": area_ha,
                        "systeme": None
                    })
        return parcels
    except Exception as exc:
        log.warning("OSM fetch failed: %s", exc)
        return []


def _run_unet_on_zone(zone_coords: List[dict]) -> List[dict]:
    """Check overlap with cached tiles, run U-Net if overlapping."""
    global _unet, _all_parcels
    if _unet is None:
        return []
        
    import numpy as np
    import torch
    
    try:
        import cv2
    except ImportError:
        return []
        
    from unet_model import _parcel_to_bbox, PATCH_SIZE, DEVICE
    from pipeline import area_ha_shoelace
    
    z_lats = [c["lat"] for c in zone_coords]
    z_lngs = [c["lng"] for c in zone_coords]
    z_min_lat, z_max_lat = min(z_lats), max(z_lats)
    z_min_lng, z_max_lng = min(z_lngs), max(z_lngs)
    
    parcels_detected = []
    
    for p in _all_parcels:
        try:
            min_lng, min_lat, max_lng, max_lat = _parcel_to_bbox(p["coordinates"])
            
            # Check for bounding box overlap
            if (z_max_lat < min_lat or z_min_lat > max_lat or 
                z_max_lng < min_lng or z_min_lng > max_lng):
                continue
                
            tile_path = Path(__file__).parent / "data" / "tiles" / f"{p['id']}.npy"
            if not tile_path.exists():
                continue
                
            tile = np.load(tile_path)  # (H, W, C)
            tile_t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logit = _unet(tile_t)[0, 0].cpu().numpy()
                
            prob_mask = 1 / (1 + np.exp(-logit))
            binary = (prob_mask > 0.5).astype(np.uint8)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lng_range = max_lng - min_lng
            lat_range = max_lat - min_lat
            
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 50:
                    continue
                    
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                coords = []
                for pt in approx:
                    px, py = int(pt[0][0]), int(pt[0][1])
                    lng = min_lng + (px / (PATCH_SIZE - 1)) * lng_range
                    lat = max_lat - (py / (PATCH_SIZE - 1)) * lat_range  # flip Y
                    coords.append({"lat": round(lat, 6), "lng": round(lng, 6)})
                    
                if len(coords) >= 3:
                    area_ha = area_ha_shoelace(coords)
                    parcels_detected.append({
                        "id": f"unet_{p['id']}_{i}",
                        "coordinates": coords,
                        "area_ha": area_ha,
                        "systeme": None
                    })
        except Exception as exc:
            log.warning("U-Net inference failed for tile %s: %s", p.get("id"), exc)
            continue
            
    return parcels_detected


def _merge_parcels(ezzayra: List[dict], osm: List[dict], unet: List[dict]) -> List[dict]:
    """Merge sources, removing duplicates by centroid proximity > 0.001 deg."""
    from pipeline import centroid
    import math
    merged = []
    
    for src_list in [ezzayra, osm, unet]:
        for p in src_list:
            c_lat, c_lng = centroid(p["coordinates"])
            is_duplicate = False
            for m in merged:
                m_lat, m_lng = centroid(m["coordinates"])
                dist = math.hypot(c_lat - m_lat, c_lng - m_lng)
                if dist <= 0.001:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(p)
                
    return merged


# ─── Schemas ────────────────────────────────────────────────────────────────

class Coordinate(BaseModel):
    lat: float
    lng: float


# ── Stage 2 only (legacy / direct classification) ──
class ParcelInput(BaseModel):
    id:          Optional[str]  = None
    coordinates: List[Coordinate]
    area_ha:     Optional[float] = None


class DirectClassifyRequest(BaseModel):
    parcelles: List[ParcelInput] = Field(..., min_items=1)


# ── Stage 1 + 2 (main flow: user draws a zone) ──
class CartographierRequest(BaseModel):
    """
    Main endpoint payload.
    polygone_perimetre: GeoJSON-like polygon drawn by the user (zone to analyse).
    date: acquisition date hint for Sentinel-2 (YYYY-MM-DD). Optional for demo.
    """
    polygone_perimetre: List[Coordinate] = Field(
        ..., description="Polygon vertices drawn on the map (lat/lng).", min_items=3
    )
    date: Optional[str] = Field(None, description="Target acquisition date YYYY-MM-DD.")


class OliveraieResult(BaseModel):
    id:           str
    systeme:      str
    confiance:    float
    surface_ha:   float
    ndvi_mean:    float
    ndwi:         float
    ndre:         float
    compactness:  float
    centroid_lat: float
    centroid_lng: float
    coordinates:  List[Coordinate]
    name:         Optional[str] = None


class StatsResult(BaseModel):
    total:              int
    surface_totale_ha:  float
    surface_moyenne_ha: float
    repartition:        dict


class CartographierResponse(BaseModel):
    oliveraies: List[OliveraieResult]
    stats:      StatsResult
    latence_ms: int
    date:       Optional[str] = None
    zone_area_km2: Optional[float] = None


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    html = FRONTEND / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return {"message": "Olive Grove Mapping API — see /docs"}


@app.get("/api/health")
def health():
    return {
        "status":      "ok",
        "model_ready": _model is not None,
        "classes":     CLASS_NAMES,
        "n_parcels":   len(_all_parcels),
        "geo_f1_macro": _metrics["geo_cv_f1_macro_mean"] if _metrics else None,
        "full_f1_macro": _metrics["full_cv_f1_macro_mean"] if _metrics else None,
    }


@app.get("/api/metrics")
def metrics():
    """Return model training metrics and feature importances."""
    if not _metrics:
        raise HTTPException(503, "Model not ready")
    return _metrics


@app.get("/api/parcelles")
def list_parcelles(systeme: Optional[str] = None):
    """Return EZZAYRA parcels as GeoJSON FeatureCollection."""
    parcels = _all_parcels
    if systeme:
        parcels = [p for p in parcels if p["systeme_label"] == systeme]

    features = []
    for p in parcels:
        lat, lng = centroid(p["coordinates"])
        features.append({
            "type": "Feature",
            "properties": {
                "id":      p["id"],
                "name":    p["name"],
                "systeme": p["systeme_label"],
                "area_ha": round(p["area_ha"], 2),
                "centroid_lat": round(lat, 6),
                "centroid_lng": round(lng, 6),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[c["lng"], c["lat"]] for c in p["coordinates"]]
                    + [[p["coordinates"][0]["lng"], p["coordinates"][0]["lat"]]]
                ],
            },
        })
    return {
        "type":     "FeatureCollection",
        "features": features,
        "total":    len(features),
    }


@app.post("/api/cartographier", response_model=CartographierResponse)
def cartographier(req: CartographierRequest):
    """
    Main pipeline endpoint.

    Stage 1 — Detection: find all olive grove parcels inside the drawn polygon.
    Stage 2 — Classification: classify each into extensif / intensif / hyper_intensif.

    POST body:
        {
          "polygone_perimetre": [ {"lat": …, "lng": …}, … ],
          "date": "2026-06-15"           // optional
        }
    """
    if _model is None:
        raise HTTPException(503, "Model not ready — retry in a few seconds.")

    t0 = time.time()

    # ── Stage 1: Detection ───────────────────────────────────────────────────
    zone_coords = [{"lat": c.lat, "lng": c.lng} for c in req.polygone_perimetre]

    # Compute approximate zone area (km²)
    zone_area_ha = area_ha_shoelace(zone_coords)
    zone_area_km2 = round(zone_area_ha / 100, 1)

    ezzayra_detected = detect_parcels_in_zone(zone_coords, _all_parcels)
    
    # Compute bbox for OSM query
    z_lats = [c["lat"] for c in zone_coords]
    z_lngs = [c["lng"] for c in zone_coords]
    bbox = (min(z_lats), min(z_lngs), max(z_lats), max(z_lngs))
    
    osm_detected = _fetch_osm_parcels(bbox)
    unet_detected = _run_unet_on_zone(zone_coords)
    
    detected = _merge_parcels(ezzayra_detected, osm_detected, unet_detected)

    if not detected:
        return CartographierResponse(
            oliveraies=[],
            stats=StatsResult(
                total=0,
                surface_totale_ha=0.0,
                surface_moyenne_ha=0.0,
                repartition={c: 0 for c in CLASS_NAMES},
            ),
            latence_ms=int((time.time() - t0) * 1000),
            date=req.date,
            zone_area_km2=zone_area_km2,
        )

    # ── Stage 2: Classification ───────────────────────────────────────────────
    results = []
    for p in detected:
        parcel_dict = {
            "id":          p["id"],
            "coordinates": p["coordinates"],
            "area_ha":     p["area_ha"],
            # Ground-truth label from EZZAYRA dataset → classify_parcel will
            # use this directly and skip ML inference for these known parcels.
            # Parcels detected by U-Net that are NOT in the dataset will not
            # have this key and will go through the classifier normally.
            "systeme":     p.get("systeme_label") or p.get("systeme"),
        }
        prediction = classify_parcel(_model, parcel_dict)
        lat, lng   = centroid(p["coordinates"])

        results.append(OliveraieResult(
            id=p["id"],
            name=p.get("name"),
            systeme=prediction["systeme"],
            confiance=prediction["confiance"],
            surface_ha=prediction["surface_ha"],
            ndvi_mean=prediction["ndvi_mean"],
            ndwi=prediction["ndwi"],
            ndre=prediction["ndre"],
            compactness=prediction["compactness"],
            centroid_lat=round(lat, 6),
            centroid_lng=round(lng, 6),
            coordinates=[Coordinate(lat=c["lat"], lng=c["lng"]) for c in p["coordinates"]],
        ))

    # ── Stats ─────────────────────────────────────────────────────────────────
    repartition   = {c: 0 for c in CLASS_NAMES}
    surface_totale = 0.0
    for r in results:
        repartition[r.systeme] += 1
        surface_totale += r.surface_ha

    latence_ms = int((time.time() - t0) * 1000)
    log.info(
        "Zone %.1f km² — detected %d parcels, classified in %d ms",
        zone_area_km2, len(results), latence_ms,
    )

    return CartographierResponse(
        oliveraies=results,
        stats=StatsResult(
            total=len(results),
            surface_totale_ha=round(surface_totale, 2),
            surface_moyenne_ha=round(surface_totale / len(results), 2) if results else 0.0,
            repartition=repartition,
        ),
        latence_ms=latence_ms,
        date=req.date,
        zone_area_km2=zone_area_km2,
    )


@app.post("/api/cartographier/geojson")
def cartographier_geojson(req: CartographierRequest):
    """Same as /api/cartographier but returns a GeoJSON FeatureCollection."""
    result = cartographier(req)
    COLORS = {
        "extensif":       "#1D9E75",
        "intensif":       "#EF9F27",
        "hyper_intensif": "#E24B4A",
    }
    features = []
    for r in result.oliveraies:
        features.append({
            "type": "Feature",
            "properties": {
                "id":         r.id,
                "name":       r.name,
                "systeme":    r.systeme,
                "confiance":  r.confiance,
                "surface_ha": r.surface_ha,
                "ndvi_mean":  r.ndvi_mean,
                "ndwi":       r.ndwi,
                "ndre":       r.ndre,
                "color":      COLORS.get(r.systeme, "#888"),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[c.lng, c.lat] for c in r.coordinates]
                    + [[r.coordinates[0].lng, r.coordinates[0].lat]]
                ],
            },
        })
    return {
        "type":     "FeatureCollection",
        "features": features,
        "stats":    result.stats.model_dump(),
        "date":     result.date,
        "zone_area_km2": result.zone_area_km2,
    }