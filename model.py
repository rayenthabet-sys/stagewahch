"""
Agronomic Twin — Sujet 03: Crop Stress Detection
Core ML logic avec seuillage par quantiles historiques (conforme PDF)
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONSTANTS — tuned per EZZAYRA olive orchard system types
# ─────────────────────────────────────────────────────────

SYSTEM_NDVI_RANGES = {
    "extensif":       {"min": 0.15, "max": 0.45, "typical": 0.28},
    "intensif":       {"min": 0.30, "max": 0.65, "typical": 0.48},
    "hyper-intensif": {"min": 0.45, "max": 0.80, "typical": 0.62},
}

# Seasonal NDVI adjustment factors for Tunisia olive phenology
TUNISIA_OLIVE_PHENOLOGY = {
    1:  0.85, 2:  0.88, 3:  0.92, 4:  1.00, 5:  1.05, 6:  1.02,
    7:  0.97, 8:  0.93, 9:  0.90, 10: 0.92, 11: 0.88, 12: 0.85,
}

# ─────────────────────────────────────────────────────────
# 1. PROPHET BASELINE — seasonal NDVI per system type
# ─────────────────────────────────────────────────────────

def compute_baseline_prophet(dates: list, ndvi: list, systeme: str = "intensif") -> np.ndarray:
    """Fit Prophet on historical NDVI, accounting for olive phenology."""
    df = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "y": np.array(ndvi, dtype=float)
    }).dropna()

    if len(df) < 4:
        # Fallback: phenology-adjusted mean
        baseline = []
        sys_info = SYSTEM_NDVI_RANGES.get(systeme, SYSTEM_NDVI_RANGES["intensif"])
        for d in pd.to_datetime(dates):
            factor = TUNISIA_OLIVE_PHENOLOGY.get(d.month, 1.0)
            baseline.append(sys_info["typical"] * factor)
        return np.array(baseline)

    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=15.0,
            interval_width=0.95,
        )

        model.add_seasonality(name="olive_phenology", period=365.25, fourier_order=5)
        model.fit(df)

        future = pd.DataFrame({"ds": pd.to_datetime(dates)})
        forecast = model.predict(future)
        baseline = forecast["yhat"].values
    except Exception as e:
        # Fallback to phenology-based baseline if Prophet fails
        print(f"Prophet failed, using phenology fallback: {e}")
        baseline = []
        sys_info = SYSTEM_NDVI_RANGES.get(systeme, SYSTEM_NDVI_RANGES["intensif"])
        for d in pd.to_datetime(dates):
            factor = TUNISIA_OLIVE_PHENOLOGY.get(d.month, 1.0)
            baseline.append(sys_info["typical"] * factor)
        baseline = np.array(baseline)

    sys_info = SYSTEM_NDVI_RANGES.get(systeme, SYSTEM_NDVI_RANGES["intensif"])
    baseline = np.clip(baseline, sys_info["min"] * 0.8, sys_info["max"] * 1.1)

    return baseline


# ─────────────────────────────────────────────────────────
# 2. SLIDING WINDOW Z-SCORE — fenêtre glissante 3 semaines (conforme PDF)
# ─────────────────────────────────────────────────────────

def compute_sliding_zscore(observed: list, expected: list, window: int = 3) -> list:
    """
    Compute normalized deviation on sliding window.
    Window = 3 observations (≈3 Sentinel-2 revisits over ~18 days).
    Conforme PDF: "écart normalisé entre NDVI observé et NDVI attendu sur fenêtre glissante 3 semaines"
    """
    obs = np.array(observed, dtype=float)
    exp = np.array(expected, dtype=float)
    residuals = obs - exp
    z_scores = []

    for i in range(len(residuals)):
        start = max(0, i - window + 1)
        window_res = residuals[start:i + 1]
        if len(window_res) < 2:
            z_scores.append(0.0)
            continue
        mu = np.mean(window_res)
        sigma = np.std(window_res) + 1e-6
        z = (residuals[i] - mu) / sigma
        z_scores.append(float(z))

    return z_scores


def get_current_zscore(z_scores: list) -> float:
    """Return the most recent z-score (current state)."""
    return z_scores[-1] if z_scores else 0.0


# ─────────────────────────────────────────────────────────
# 3. ISOLATION FOREST — multi-feature anomaly scoring
# ─────────────────────────────────────────────────────────

def compute_isolation_forest_score(
    ndvi: list,
    rainfall: list = None,
    temperature: list = None,
    lst: list = None,
) -> tuple:
    """Multi-feature Isolation Forest."""
    ndvi_arr = np.array(ndvi, dtype=float)
    n = len(ndvi_arr)

    features = [ndvi_arr]
    ndvi_gradient = np.gradient(ndvi_arr)
    features.append(ndvi_gradient)
    ndvi_accel = np.gradient(ndvi_gradient)
    features.append(ndvi_accel)

    if rainfall is not None and len(rainfall) == n:
        features.append(np.array(rainfall, dtype=float))
    if temperature is not None and len(temperature) == n:
        features.append(np.array(temperature, dtype=float))
    if lst is not None and len(lst) == n:
        features.append(np.array(lst, dtype=float))

    X = np.column_stack(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = min(0.20, max(0.05, 1.0 / n))
    model = IsolationForest(contamination=contamination, n_estimators=200, max_samples="auto", random_state=42)
    model.fit(X_scaled)

    raw_scores = -model.decision_function(X_scaled)
    if raw_scores.max() > raw_scores.min():
        iso_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min()) * 3.0
    else:
        iso_scores = raw_scores * 0

    return float(iso_scores[-1]), iso_scores.tolist()


# ─────────────────────────────────────────────────────────
# 4. CHANGE POINT DETECTION — PELT algorithm
# ─────────────────────────────────────────────────────────

def detect_change_points(ndvi: list, dates: list) -> dict:
    """PELT change point detection for stress onset identification."""
    ndvi_arr = np.array(ndvi, dtype=float)

    if len(ndvi_arr) < 6:
        return {"stress_start_date": None, "n_breakpoints": 0, "severity": 0.0}

    try:
        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(ndvi_arr)
        breakpoints = algo.predict(pen=3)
        actual_bps = breakpoints[:-1]

        if not actual_bps:
            return {"stress_start_date": None, "n_breakpoints": 0, "severity": 0.0}

        last_bp_idx = actual_bps[-1]
        stress_start = pd.to_datetime(dates[last_bp_idx]).strftime("%Y-%m-%d")
        before_mean = np.mean(ndvi_arr[:last_bp_idx]) if last_bp_idx > 0 else ndvi_arr[0]
        after_mean = np.mean(ndvi_arr[last_bp_idx:])
        severity = float(max(0, before_mean - after_mean))

        return {
            "stress_start_date": stress_start,
            "n_breakpoints": len(actual_bps),
            "breakpoint_indices": actual_bps,
            "ndvi_drop": round(severity, 4),
        }
    except Exception:
        return {"stress_start_date": None, "n_breakpoints": 0, "severity": 0.0}


# ─────────────────────────────────────────────────────────
# 5. LST THERMAL STRESS MODULE (bonus PDF)
# ─────────────────────────────────────────────────────────

def analyze_lst_thermal_stress(lst_values: list, dates: list) -> dict:
    """
    Analyze Land Surface Temperature for pre-NDVI thermal stress detection.
    Conforme PDF: "détecter le stress thermique avant qu'il n'apparaisse en NDVI"
    """
    if not lst_values:
        return {"thermal_stress": False, "max_lst": None, "stress_days": 0}

    lst_arr = np.array(lst_values, dtype=float)
    heat_threshold = 38.0
    sustained_threshold = 35.0

    max_lst = float(np.nanmax(lst_arr))
    stress_days = int(np.sum(lst_arr > heat_threshold))

    sustained_event = False
    consecutive = 0
    for v in lst_arr:
        if v > sustained_threshold:
            consecutive += 1
            if consecutive >= 3:
                sustained_event = True
                break
        else:
            consecutive = 0

    return {
        "thermal_stress": sustained_event or stress_days > 2,
        "max_lst": round(max_lst, 1),
        "stress_days_above_38C": stress_days,
        "sustained_heat_event": sustained_event,
    }


# ─────────────────────────────────────────────────────────
# 6. CLASSIFICATION AVEC QUANTILES HISTORIQUES (conforme PDF)
# ─────────────────────────────────────────────────────────
# PDF: "Seuiller dynamiquement en vert / orange / rouge avec des seuils basés 
#       sur les quantiles historiques de la parcelle"

def compute_historical_quantiles(z_scores_history: list) -> dict:
    """
    Calcule les quantiles historiques des z-scores pour seuillage dynamique.
    Retourne les percentiles 70e, 85e, 95e.
    """
    if len(z_scores_history) < 5:
        # Seuils par défaut si peu d'historique
        return {"p70": 1.0, "p85": 1.8, "p95": 2.5}
    
    abs_scores = [abs(z) for z in z_scores_history]
    return {
        "p70": float(np.percentile(abs_scores, 70)),
        "p85": float(np.percentile(abs_scores, 85)),
        "p95": float(np.percentile(abs_scores, 95)),
    }


def classify_anomaly_with_quantiles(
    z_score: float,
    iso_score: float,
    ndvi_drop: float,
    thermal_stress: bool,
    systeme: str,
    quantiles: dict,
) -> tuple:
    """
    Classification basée sur quantiles historiques (conforme PDF).
    """
    # Composite score
    composite = (
        abs(z_score) * 0.45
        + iso_score * 0.35
        + ndvi_drop * 3.0 * 0.20
    )
    
    if thermal_stress:
        composite += 0.5
    
    # Seuillage dynamique basé sur quantiles
    thresh_orange = quantiles["p70"]
    thresh_rouge = quantiles["p85"]
    
    # Ajustement système
    multiplier = {"extensif": 1.2, "intensif": 1.0, "hyper-intensif": 0.85}.get(systeme, 1.0)
    thresh_orange *= multiplier
    thresh_rouge *= multiplier
    
    if composite >= thresh_rouge:
        status = "rouge"
        confidence = min(0.99, 0.70 + (composite - thresh_rouge) * 0.1)
    elif composite >= thresh_orange:
        status = "orange"
        confidence = min(0.85, 0.55 + (composite - thresh_orange) * 0.1)
    else:
        status = "vert"
        confidence = min(0.95, 0.80 + (thresh_orange - composite) * 0.1)
    
    return status, round(composite, 3), round(confidence, 3)


# ─────────────────────────────────────────────────────────
# 7. EXPLANATION ENGINE (conforme PDF)
# ─────────────────────────────────────────────────────────

def generate_explanation(
    z_score: float,
    iso_score: float,
    ndvi_drop: float,
    status: str,
    systeme: str,
    stress_date: str,
    thermal_stress: bool,
    rainfall_deficit: float = None,
) -> dict:
    """
    Auto-generate human-readable explanation.
    PDF: "Module d'explication : générer automatiquement un texte court qui explique 
          pourquoi la parcelle est en alerte."
    """
    reasons = []
    recommendations = []

    # Z-score interpretation (écart normalisé)
    if z_score < -2.5:
        reasons.append(f"Chute NDVI sévère ({abs(z_score):.1f}σ sous la normale saisonnière)")
        recommendations.append("Inspection terrain urgente dans les 24-48h")
    elif z_score < -1.5:
        reasons.append(f"Déclin NDVI modéré ({abs(z_score):.1f}σ sous l'attendu)")
        recommendations.append("Vérification irrigation conseillée sous 72h")
    elif z_score < -0.8:
        reasons.append(f"Légère baisse NDVI ({abs(z_score):.1f}σ sous l'attendu)")
        recommendations.append("Surveillance renforcée")

    # Rainfall deficit
    if rainfall_deficit is not None and rainfall_deficit < -20:
        reasons.append(f"Déficit pluviométrique significatif ({abs(rainfall_deficit):.0f}mm sous la normale)")
        recommendations.append("Activer irrigation d'appoint si non planifiée")

    # Thermal stress (bonus)
    if thermal_stress:
        reasons.append("Stress thermique détecté (LST > 38°C) — visible avant chute NDVI")
        recommendations.append("Surveiller humidité foliaire et stomatal conductance")

    # NDVI drop
    if ndvi_drop > 0.08:
        reasons.append(f"Perte végétative absolue de {ndvi_drop:.2f} points NDVI depuis le changement de régime")

    # System-specific context
    sys_context = {
        "extensif": "Le système extensif sec tolère plus de variabilité naturelle",
        "intensif": "Le système intensif irrigué devrait maintenir un NDVI stable",
        "hyper-intensif": "Le haies hyper-intensif est sensible aux variations — réponse rapide requise",
    }
    context = sys_context.get(systeme, "")

    # Status-based final message
    if status == "rouge":
        main_msg = "⚠️ ALERTE ROUGE : Stress probable confirmé par plusieurs indicateurs. Intervention urgente requise."
    elif status == "orange":
        main_msg = "🟠 VIGILANCE : Anomalie détectée, comportement à surveiller. Inspection recommandée."
    else:
        main_msg = "✅ NORMAL : Végétation dans les normes saisonnières attendues. Continuer surveillance."

    if stress_date:
        reasons.append(f"Début de déviation détecté autour du {stress_date}")

    return {
        "message_principal": main_msg,
        "raisons": reasons if reasons else ["Comportement NDVI conforme au baseline saisonnier"],
        "recommandations": recommendations if recommendations else ["Continuer surveillance routine"],
        "contexte_systeme": context,
    }


# ─────────────────────────────────────────────────────────
# 8. FULL PIPELINE
# ─────────────────────────────────────────────────────────

def run_full_detection(
    orchard_id: str,
    dates: list,
    ndvi: list,
    systeme: str = "intensif",
    rainfall: list = None,
    temperature: list = None,
    lst: list = None,
) -> dict:
    """
    Master orchestrator — runs the full anomaly detection pipeline.
    Conforme cahier des charges Sujet 03.
    """
    # 1. Prophet baseline
    baseline = compute_baseline_prophet(dates, ndvi, systeme)

    # 2. Sliding Z-score (fenêtre glissante 3 semaines)
    z_scores = compute_sliding_zscore(ndvi, baseline, window=3)
    current_z = get_current_zscore(z_scores)

    # 3. Isolation Forest
    iso_current, iso_all = compute_isolation_forest_score(ndvi, rainfall, temperature, lst)

    # 4. Change point detection
    cp_info = detect_change_points(ndvi, dates)
    ndvi_drop = cp_info.get("ndvi_drop", 0.0)

    # 5. LST thermal stress
    thermal_info = analyze_lst_thermal_stress(lst or [], dates)
    thermal_stress = thermal_info["thermal_stress"]

    # 6. Calcul des quantiles historiques (conforme PDF)
    quantiles = compute_historical_quantiles(z_scores)
    
    # 7. Classification avec quantiles
    status, composite_score, confidence = classify_anomaly_with_quantiles(
        current_z, iso_current, ndvi_drop, thermal_stress, systeme, quantiles
    )

    # 8. Rainfall deficit calculation
    rainfall_deficit = None
    if rainfall and len(rainfall) >= 4:
        recent_rain = sum(rainfall[-4:])
        hist_rain = sum(rainfall[:-4]) / max(1, len(rainfall) - 4) * 4
        rainfall_deficit = recent_rain - hist_rain

    # 9. Explanation
    explanation = generate_explanation(
        current_z, iso_current, ndvi_drop, status, systeme,
        cp_info.get("stress_start_date"), thermal_stress, rainfall_deficit
    )

    return {
        "id": orchard_id,
        "statut": status,
        "anomaly_score": composite_score,
        "confidence": confidence,
        "z_score_actuel": round(current_z, 3),
        "iso_score": round(iso_current, 3),
        "ndvi_observe": [round(v, 4) for v in ndvi],
        "ndvi_attendu": [round(float(v), 4) for v in baseline],
        "z_scores_historique": [round(z, 3) for z in z_scores],
        "iso_scores_historique": [round(s, 3) for s in iso_all],
        "dates": dates,
        "quantiles_seuils": {
            "p70": round(quantiles["p70"], 3),
            "p85": round(quantiles["p85"], 3),
            "p95": round(quantiles["p95"], 3),
        },
        "changement_regime": {
            "date_debut_stress": cp_info.get("stress_start_date"),
            "nb_ruptures": cp_info.get("n_breakpoints", 0),
            "chute_ndvi": round(ndvi_drop, 4),
        },
        "stress_thermique": thermal_info,
        "explication": explanation,
        "metadata": {
            "systeme": systeme,
            "n_observations": len(ndvi),
            "periode_couverte": f"{dates[0]} → {dates[-1]}" if dates else "N/A",
        }
    }