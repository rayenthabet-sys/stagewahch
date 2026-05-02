# weather.py
"""
Weather & Climate Data Fetcher
- Open-Meteo Archive API (80 years, no key)
"""

import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import asyncio

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


async def fetch_weather_for_dates(
    lat: float,
    lon: float,
    dates: list,
) -> dict:
    """Fetch weather data from Open-Meteo for given coordinates and date range."""
    if not dates:
        return {}

    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    start_date = min(date_objs) - timedelta(days=30)
    end_date = max(date_objs)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "Africa/Tunis",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        daily = data.get("daily", {})
        all_dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_mean = daily.get("temperature_2m_mean", [])
        et0 = daily.get("et0_fao_evapotranspiration", [])

        weather_lookup = {}
        for i, d in enumerate(all_dates):
            weather_lookup[d] = {
                "rainfall_mm": precip[i] if i < len(precip) else None,
                "temp_max_C": temp_max[i] if i < len(temp_max) else None,
                "temp_mean_C": temp_mean[i] if i < len(temp_mean) else None,
                "et0_mm": et0[i] if i < len(et0) else None,
            }

        result = {}
        for target_date in dates:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            window_days = []
            for offset in range(-5, 6):
                check_date = (target_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
                if check_date in weather_lookup:
                    window_days.append(weather_lookup[check_date])

            if not window_days:
                result[target_date] = {}
                continue

            def safe_sum(key):
                vals = [d[key] for d in window_days if d.get(key) is not None]
                return round(sum(vals), 2) if vals else None

            def safe_mean(key):
                vals = [d[key] for d in window_days if d.get(key) is not None]
                return round(np.mean(vals), 2) if vals else None

            result[target_date] = {
                "rainfall_cumul_mm": safe_sum("rainfall_mm"),
                "temp_max_C": safe_max("temp_max_C"),
                "temp_mean_C": safe_mean("temp_mean_C"),
                "et0_cumul_mm": safe_sum("et0_mm"),
            }

        return result

    except Exception as e:
        print(f"[Weather] Error: {e}")
        return {}


def safe_max(key):
    vals = [d[key] for d in window_days if d.get(key) is not None]
    return round(max(vals), 2) if vals else None


def extract_weather_series(weather_data: dict, dates: list) -> dict:
    """Convert weather dict to parallel lists aligned with NDVI dates."""
    rainfall = []
    temperature = []
    et0 = []
    lst_proxy = []

    for d in dates:
        entry = weather_data.get(d, {})
        rainfall.append(entry.get("rainfall_cumul_mm") or 0.0)
        temperature.append(entry.get("temp_mean_C") or 20.0)
        et0.append(entry.get("et0_cumul_mm") or 0.0)
        tmax = entry.get("temp_max_C") or 25.0
        lst_proxy.append(tmax + 4.0)

    return {
        "rainfall": rainfall,
        "temperature": temperature,
        "et0": et0,
        "lst_proxy": lst_proxy,
    }