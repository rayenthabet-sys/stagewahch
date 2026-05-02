# 🫒 Agronomic Twin — Détection précoce d'anomalies sur oliveraies

**Sujet 03 · Hack The Harvest 2026 · EZZAYRA × ISI**

> Système de détection d'anomalies NDVI en temps quasi-réel sur les oliveraies tunisiennes.
> Alerte vert/orange/rouge avec explication automatique, baseline Prophet saisonnier,
> Isolation Forest multi-features, PELT change-point detection, et stress thermique LST.

---

## 🏗️ Architecture

```
agronomic-twin/
├── backend/
│   ├── main.py          ← FastAPI API (endpoints jury)
│   ├── model.py         ← Pipeline ML complet
│   ├── weather.py       ← Open-Meteo + CHIRPS fetch
│   └── requirements.txt
│
├── dashboard/
│   ├── app.py           ← Streamlit dashboard
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ Pipeline ML (6 modules)

### 1. Baseline Prophet saisonnier
- Modèle Prophet (Meta) avec saisonnalité multiplicative yearly
- Calibré sur phénologie olive tunisienne (12 facteurs mensuels)
- Seuils par système: extensif / intensif / hyper-intensif
- Fallback phénologique si `n < 4` observations

### 2. Z-score fenêtre glissante 3 semaines
- Résidus `observed - expected` sur fenêtre ±3 acquisitions Sentinel-2
- Normalisation par σ local de la fenêtre
- Détecte anomalies transitoires vs structurelles

### 3. Isolation Forest multi-features
- Features: NDVI + gradient NDVI + accélération NDVI + rainfall + temp + LST
- Contamination adaptative (1/n, clampé 5%-20%)
- 200 estimateurs, seed=42 (reproductible jury)
- Score normalisé [0, 3]

### 4. PELT Change-Point Detection (ruptures)
- Algorithme PELT rbf, pen=3
- Identifie la date exacte de début de stress
- Calcule la chute NDVI absolue avant/après rupture

### 5. LST Thermal Stress Module (BONUS +points jury)
- Seuil critique olive: LST > 38°C
- Seuil alerte: LST > 35°C soutenu 3+ jours consécutifs
- Détecte stress thermique AVANT chute NDVI visible
- Source: MODIS LST (Landsat 8/9 thermique) ou proxy Open-Meteo

### 6. Classification Ensemble + Explication
```
score_composite = |z| × 0.45 + iso × 0.35 + ndvi_drop × 3 × 0.20
bonus_LST = +0.5 si stress thermique confirmé

seuils (ajustés par système):
  vert   < 1.0 × mult
  orange ≥ 1.8 × mult
  rouge  ≥ 2.5 × mult

mult: extensif=1.2 | intensif=1.0 | hyper-intensif=0.85
```

---

## 📡 Sources de données intégrées

| Source | Données | Tier | Usage |
|--------|---------|------|-------|
| **JSON EZZAYRA** | Polygones oliveraies + système | T1 | Baseline de référence |
| **Sentinel-2 série temporelle** | NDVI 10m, revisit 5j | T2 | Signal principal |
| **CHIRPS** | Pluviométrie satellite 0.05° | T1 | Feature rainfall |
| **Open-Meteo Archive API** | Météo 80 ans, ET₀, soil moisture | T1 | Features calage |
| **MODIS LST** | Température surface 1km | T3 | Bonus stress thermique |

---

## 🚀 Lancement rapide

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API disponible: http://127.0.0.1:8000
Docs interactives: http://127.0.0.1:8000/docs

### Dashboard (Streamlit)
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
Dashboard: http://localhost:8501

---

## 🎯 Endpoints API (procédure jury)

### Diagnostic unique (endpoint attendu)
```http
POST /api/diagnostic-anomalie
Content-Type: application/json

{
  "oliveraie": {
    "id": "O_2026_307",
    "systeme": "intensif",
    "centroide": { "lat": 36.5, "lon": 9.8 }
  },
  "dates": ["2026-01-10", "2026-01-26", "2026-02-11", "2026-02-27",
             "2026-03-15", "2026-03-31", "2026-04-16", "2026-05-02"],
  "ndvi": [0.56, 0.58, 0.57, 0.55, 0.50, 0.44, 0.38, 0.32],
  "lst": [22.1, 21.5, 23.8, 24.2, 28.5, 32.1, 36.8, 39.2],
  "fetch_weather": true
}
```

**Réponse complète:**
```json
{
  "id": "O_2026_307",
  "statut": "rouge",
  "anomaly_score": 2.847,
  "confidence": 0.91,
  "z_score_actuel": -3.21,
  "iso_score": 1.84,
  "ndvi_observe": [0.56, 0.58, 0.57, 0.55, 0.50, 0.44, 0.38, 0.32],
  "ndvi_attendu": [0.56, 0.57, 0.56, 0.55, 0.52, 0.50, 0.48, 0.47],
  "z_scores_historique": [0.1, 0.3, -0.2, -0.5, -1.2, -2.1, -2.8, -3.2],
  "changement_regime": {
    "date_debut_stress": "2026-03-31",
    "nb_ruptures": 1,
    "chute_ndvi": 0.0842
  },
  "stress_thermique": {
    "thermal_stress": true,
    "max_lst": 39.2,
    "stress_days_above_38C": 2,
    "sustained_heat_event": true
  },
  "explication": {
    "message_principal": "⚠️ ALERTE ROUGE : Stress probable confirmé par plusieurs indicateurs",
    "raisons": [
      "Chute NDVI sévère (3.2σ sous la normale saisonnière)",
      "Stress thermique détecté (LST > 38°C) — visible avant chute NDVI",
      "Perte végétative absolue de 0.08 points NDVI depuis le changement de régime",
      "Début de déviation détecté autour du 2026-03-31"
    ],
    "recommandations": [
      "Inspection terrain urgente dans les 24-48h",
      "Surveiller humidité foliaire et stomatal conductance"
    ],
    "contexte_systeme": "Le système intensif irrigué devrait maintenir un NDVI stable"
  }
}
```

### Démo jury (pré-chargée)
```http
GET /api/demo
```

### Flotte complète
```http
GET /api/demo-fleet
```

### Batch (50 parcelles max)
```http
POST /api/batch-diagnostic
```

---

## ☁️ Déploiement production

### Render.com (backend)
```
Build command: pip install -r requirements.txt
Start command: uvicorn main:app --host 0.0.0.0 --port 10000
```

### Streamlit Cloud (dashboard)
- Upload repo sur GitHub
- Pointer sur `dashboard/app.py`
- Mettre à jour `API_URL` avec l'URL Render

---

## ✅ Critères jury couverts

| Critère | ✅ |
|---------|---|
| Distinction vraie anomalie vs bruit saisonnier | ✅ Prophet + z-score fenêtre |
| Module d'explication automatique | ✅ 4 composantes textuelles |
| Dashboard fluide, lisible non-technique | ✅ Streamlit dark agri UI |
| Démo API live sur oliveraie réelle | ✅ GET /api/demo |
| **Bonus LST thermique (stress avant NDVI)** | ✅ Module dédié |

## ⚠️ Pièges évités

- **Phénologie ignorée** → baseline Prophet multiplicatif yearly ✅
- **Bruit pixel** → NDVI moyenné sur polygone entier ✅
- **Faux positifs taille** → seuils dynamiques par mois et par système ✅
- **Différence E/I/HI** → mult ajusté par systeme (0.85 → 1.2) ✅

---

*Hack The Harvest 2026 · EZZAYRA ISI · Généré avec Agronomic Twin v2.0*
