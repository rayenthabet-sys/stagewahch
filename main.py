"""
Agronomic Twin API — Sujet 03: Détection précoce d'anomalies sur oliveraies
FastAPI backend avec support GeoJSON Polygon
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import asyncio
import traceback
import json

from model import run_full_detection
from weather import fetch_weather_for_dates, extract_weather_series

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Agronomic Twin — Détection Stress Oliveraies",
    description="API de détection précoce d'anomalies NDVI sur oliveraies tunisiennes (Sujet 03)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class GeoPoint(BaseModel):
    lat: float = Field(..., ge=30.0, le=38.0)
    lon: float = Field(..., ge=7.5, le=12.0)


class GeoJSONPolygon(BaseModel):
    """GeoJSON Polygon conforme standard RFC 7946"""
    type: str = Field(default="Polygon", description="GeoJSON type")
    coordinates: List[List[List[float]]] = Field(..., description="Polygon coordinates [[[lon, lat], ...]]")

    @validator("type")
    def validate_type(cls, v):
        if v != "Polygon":
            raise ValueError("Only Polygon type is supported")
        return v


class OliveOrchard(BaseModel):
    id: str = Field(..., description="Identifiant parcelle EZZAYRA")
    polygone: Optional[GeoJSONPolygon] = Field(None, description="GeoJSON Polygon de la parcelle")
    centroide: Optional[GeoPoint] = Field(None, description="Centroid (fallback si pas de polygone)")
    systeme: str = Field("intensif", description="extensif | intensif | hyper-intensif")

    @validator("systeme")
    def validate_systeme(cls, v):
        valid = ["extensif", "intensif", "hyper-intensif"]
        if v not in valid:
            raise ValueError(f"systeme must be one of {valid}")
        return v
    
    def get_centroid(self) -> GeoPoint:
        """Retourne le centroïde à partir du polygone ou du champ centroide"""
        if self.centroide:
            return self.centroide
        if self.polygone:
            # Calcul simple du centroïde à partir du polygone
            coords = self.polygone.coordinates[0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            return GeoPoint(lat=sum(lats)/len(lats), lon=sum(lons)/len(lons))
        raise ValueError("No centroid or polygon provided")


class DiagnosticRequest(BaseModel):
    oliveraie: OliveOrchard
    dates: List[str] = Field(..., min_items=3, description="ISO dates")
    ndvi: List[float] = Field(..., min_items=3, description="NDVI values per date")
    rainfall: Optional[List[float]] = Field(None)
    lst: Optional[List[float]] = Field(None)
    fetch_weather: bool = Field(True)

    @validator("ndvi")
    def validate_ndvi(cls, v, values):
        if "dates" in values and len(v) != len(values["dates"]):
            raise ValueError("ndvi and dates must have the same length")
        for val in v:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"NDVI value {val} out of range [0, 1]")
        return v
    
    @validator("lst")
    def validate_lst(cls, v, values):
        if v and "dates" in values and len(v) != len(values["dates"]):
            raise ValueError("lst and dates must have the same length")
        return v


class BatchDiagnosticRequest(BaseModel):
    parcelles: List[DiagnosticRequest] = Field(..., max_items=50)


# ─────────────────────────────────────────────
# Helper: calculer surface d'un polygone (ha)
# ─────────────────────────────────────────────

def calculate_polygon_area(coordinates: List[List[List[float]]]) -> float:
    """Calcule la surface en hectares d'un polygone GeoJSON"""
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        coords = coordinates[0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        area, _ = geod.polygon_area(lons, lats)
        return abs(area) / 10000  # Convertir m² en hectares
    except ImportError:
        # Fallback approximatif si pyproj non dispo
        return 0.0


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Agronomic Twin — Sujet 03",
        "version": "3.0.0",
        "endpoints": [
            "POST /api/diagnostic-anomalie",
            "POST /api/batch-diagnostic",
            "GET /api/demo",
            "GET /api/demo-fleet"
        ]
    }


@app.post("/api/diagnostic-anomalie")
async def diagnostic_anomalie(request: DiagnosticRequest):
    """
    Main endpoint — full diagnostic for one olive orchard.
    Conforme au cahier des charges Sujet 03.
    
    Accepte soit un polygone GeoJSON soit un centroïde.
    """
    try:
        # Récupérer le centroïde (depuis polygone ou direct)
        centroid = request.oliveraie.get_centroid()
        
        rainfall = request.rainfall or []
        temperature = []
        lst = request.lst or []

        # Auto-fetch weather from Open-Meteo
        if request.fetch_weather:
            weather_data = await fetch_weather_for_dates(
                centroid.lat, centroid.lon, request.dates
            )
            weather_series = extract_weather_series(weather_data, request.dates)

            if not rainfall:
                rainfall = weather_series["rainfall"]
            temperature = weather_series["temperature"]
            if not lst:
                lst = weather_series["lst_proxy"]

        # Run full detection pipeline
        result = run_full_detection(
            orchard_id=request.oliveraie.id,
            dates=request.dates,
            ndvi=request.ndvi,
            systeme=request.oliveraie.systeme,
            rainfall=rainfall if rainfall else None,
            temperature=temperature if temperature else None,
            lst=lst if lst else None,
        )

        # Ajouter le polygone pour la carte
        if request.oliveraie.polygone:
            result["polygone"] = request.oliveraie.polygone.dict()
            result["surface_ha"] = calculate_polygon_area(request.oliveraie.polygone.coordinates)
        else:
            result["polygone"] = None
            result["surface_ha"] = None
        
        result["centroide"] = {"lat": centroid.lat, "lon": centroid.lon}

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection pipeline error: {str(e)}")


@app.post("/api/batch-diagnostic")
async def batch_diagnostic(request: BatchDiagnosticRequest):
    """Batch endpoint — run diagnostics on multiple orchards."""
    tasks = [diagnostic_anomalie(p) for p in request.parcelles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    errors = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append({"index": i, "error": str(r)})
        else:
            processed.append(r)

    statuts = [r["statut"] for r in processed]
    summary = {
        "total": len(request.parcelles),
        "traites": len(processed),
        "erreurs": len(errors),
        "repartition": {
            "vert": statuts.count("vert"),
            "orange": statuts.count("orange"),
            "rouge": statuts.count("rouge"),
        },
        "alertes_critiques": [r["id"] for r in processed if r["statut"] == "rouge"],
        "score_moyen": round(sum(r["anomaly_score"] for r in processed) / len(processed), 3) if processed else 0,
    }

    return {
        "summary": summary,
        "resultats": processed,
        "erreurs": errors,
    }


@app.get("/api/demo")
async def get_demo():
    """
    Demo parcel for jury live testing.
    Conforme PDF: exemple de parcelle avec stress hydrique.
    """
    demo_request = DiagnosticRequest(
        oliveraie=OliveOrchard(
            id="O_2026_307",
            systeme="intensif",
            centroide=GeoPoint(lat=36.5, lon=9.8),
        ),
        dates=[
            "2026-01-10", "2026-01-26", "2026-02-11", "2026-02-27",
            "2026-03-15", "2026-03-31", "2026-04-16", "2026-05-02",
        ],
        ndvi=[0.56, 0.58, 0.57, 0.55, 0.50, 0.44, 0.38, 0.32],
        rainfall=[18.2, 22.1, 15.4, 12.8, 4.2, 1.1, 0.5, 0.0],
        lst=[22.1, 21.5, 23.8, 24.2, 28.5, 32.1, 36.8, 39.2],
        fetch_weather=False,
    )
    return await diagnostic_anomalie(demo_request)


@app.get("/api/demo-fleet")
async def get_demo_fleet():
    """
    Demo fleet of 5 orchards with varied stress levels.
    """
    fleet = [
        {"id": "O_2026_101", "systeme": "intensif", "lat": 36.8, "lon": 9.5,
         "ndvi": [0.62, 0.63, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60],
         "lst": [21.0, 21.5, 22.0, 23.5, 26.0, 27.0, 29.0, 30.0]},
        {"id": "O_2026_205", "systeme": "extensif", "lat": 36.4, "lon": 10.1,
         "ndvi": [0.32, 0.33, 0.34, 0.32, 0.29, 0.25, 0.22, 0.20],
         "lst": [24.0, 24.5, 26.0, 28.0, 32.0, 35.0, 37.0, 38.5]},
        {"id": "O_2026_307", "systeme": "intensif", "lat": 36.5, "lon": 9.8,
         "ndvi": [0.56, 0.58, 0.57, 0.55, 0.50, 0.44, 0.38, 0.32],
         "lst": [22.1, 21.5, 23.8, 24.2, 28.5, 32.1, 36.8, 39.2]},
        {"id": "O_2026_412", "systeme": "hyper-intensif", "lat": 36.9, "lon": 9.3,
         "ndvi": [0.72, 0.68, 0.62, 0.59, 0.58, 0.61, 0.65, 0.68],
         "lst": [20.0, 20.5, 23.0, 26.0, 30.0, 28.0, 25.0, 23.0]},
        {"id": "O_2026_518", "systeme": "intensif", "lat": 36.6, "lon": 9.6,
         "ndvi": [0.52, 0.51, 0.50, 0.48, 0.45, 0.43, 0.41, 0.40],
         "lst": [22.0, 23.0, 25.0, 27.0, 30.0, 33.0, 35.0, 36.0]},
    ]

    dates = [
        "2026-01-10", "2026-01-26", "2026-02-11", "2026-02-27",
        "2026-03-15", "2026-03-31", "2026-04-16", "2026-05-02",
    ]

    results = []
    for orchard in fleet:
        req = DiagnosticRequest(
            oliveraie=OliveOrchard(
                id=orchard["id"],
                systeme=orchard["systeme"],
                centroide=GeoPoint(lat=orchard["lat"], lon=orchard["lon"]),
            ),
            dates=dates,
            ndvi=orchard["ndvi"],
            lst=orchard["lst"],
            fetch_weather=False,
        )
        result = await diagnostic_anomalie(req)
        result["centroide"] = {"lat": orchard["lat"], "lon": orchard["lon"]}
        results.append(result)

    statuts = [r["statut"] for r in results]
    return {
        "parcelles": results,
        "resume": {
            "vert": statuts.count("vert"),
            "orange": statuts.count("orange"),
            "rouge": statuts.count("rouge"),
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)