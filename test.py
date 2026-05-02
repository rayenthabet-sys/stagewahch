# test.py
import requests
import json

response = requests.post(
    "http://localhost:8000/api/diagnostic-anomalie",
    json={
        "oliveraie": {
            "id": "O_2026_307",
            "systeme": "intensif",
            "centroide": {"lat": 36.5, "lon": 9.8}
        },
        "dates": ["2026-06-01", "2026-06-15", "2026-06-30", "2026-07-15"],
        "ndvi": [0.52, 0.48, 0.42, 0.38],
        "fetch_weather": False
    }
)

print(json.dumps(response.json(), indent=2, ensure_ascii=False))






