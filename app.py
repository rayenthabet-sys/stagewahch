"""
Agronomic Twin Dashboard — Détection précoce d'anomalies sur oliveraies
Sujet 03 — Hack The Harvest 2026 — EZZAYRA × ISI

Dashboard conforme au cahier des charges:
  - Carte Leaflet avec parcelles colorées (vert/orange/rouge)
  - Filtres par gouvernorat et système
  - Popup avec détail NDVI observé vs attendu
  - Courbe NDVI interactive
  - Score d'anomalie
  - Explication automatique
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime
import re

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agronomic Twin · Oliveraies Tunisie",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

STATUS_COLORS = {
    "vert":   "#22c55e",
    "orange": "#f97316",
    "rouge":  "#ef4444",
}
STATUS_LABELS = {
    "vert":   "✅ NORMAL",
    "orange": "🟠 VIGILANCE",
    "rouge":  "🔴 ALERTE",
}

# Gouvernorats tunisiens pour filtrage
GOUVERNORATS = [
    "Tous", "Ariana", "Béja", "Ben Arous", "Bizerte", "Gabès", "Gafsa", "Jendouba",
    "Kairouan", "Kasserine", "Kébili", "Kef", "Mahdia", "Manouba", "Médenine",
    "Monastir", "Nabeul", "Sfax", "Sidi Bouzid", "Siliana", "Sousse", "Tataouine",
    "Tozeur", "Tunis", "Zaghouan"
]

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0f0a;
    color: #e8f5e2;
}

.main-header {
    background: linear-gradient(135deg, #0d1a0d 0%, #1a2e1a 50%, #0f1f0f 100%);
    border-bottom: 1px solid #2d4a2d;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #a8d5a2;
    letter-spacing: -0.03em;
    margin: 0;
}
.main-subtitle {
    font-size: 0.85rem;
    color: #5a8a5a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
}
.badge-vert { background: #14532d; color: #86efac; border: 1px solid #22c55e; }
.badge-orange { background: #431407; color: #fdba74; border: 1px solid #f97316; }
.badge-rouge { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }

.filter-container {
    background: #0d1a0d;
    border: 1px solid #1e3a1e;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4d7a4d;
    border-bottom: 1px solid #1e3a1e;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.explication-panel {
    background: #0d1a0d;
    border-left: 3px solid #4ade80;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}
.explication-rouge {
    border-left-color: #ef4444;
    background: #0f0a0a;
}
.explication-orange {
    border-left-color: #f97316;
    background: #100d09;
}

section[data-testid="stSidebar"] {
    background: #070f07;
    border-right: 1px solid #1a2e1a;
}

div[data-testid="stMetric"] {
    background: #111b11;
    border: 1px solid #1e3a1e;
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="stMetric"] label {
    color: #5a8a5a !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #c8e6c4 !important;
    font-family: 'Syne', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #166534, #15803d);
    color: #dcfce7;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #15803d, #16a34a);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────

def call_api(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend API non joignable. Assurez-vous que uvicorn tourne sur :8000")
        return None
    except Exception as e:
        st.error(f"Erreur API: {e}")
        return None


def get_demo_fleet() -> dict:
    try:
        resp = requests.get(f"{API_URL}/api/demo-fleet", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Demo fleet non disponible: {e}")
        return None


# ─────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────

def build_ndvi_chart(result: dict) -> go.Figure:
    """NDVI observed vs expected chart."""
    dates = result["dates"]
    observed = result["ndvi_observe"]
    expected = result["ndvi_attendu"]
    status = result["statut"]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["NDVI observé vs attendu (baseline saisonnier)", "Score d'anomalie (Z-score sur fenêtre 3 semaines)"]
    )

    # Confidence band
    upper = [e + 0.05 for e in expected]
    lower = [e - 0.05 for e in expected]

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(34, 197, 94, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Zone normale (±1σ)",
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=expected,
        mode="lines",
        name="NDVI attendu (baseline saisonnier)",
        line=dict(color="#4ade80", width=2, dash="dash"),
    ), row=1, col=1)

    obs_color = STATUS_COLORS[status]
    fig.add_trace(go.Scatter(
        x=dates, y=observed,
        mode="lines+markers",
        name="NDVI observé (Sentinel-2)",
        line=dict(color=obs_color, width=3),
        marker=dict(size=8, color=obs_color, line=dict(color="#0a0f0a", width=2)),
    ), row=1, col=1)

    stress_date = result.get("changement_regime", {}).get("date_debut_stress")
    if stress_date and stress_date in dates:
        fig.add_vline(
            x=stress_date,
            line_dash="dot",
            line_color="#ef4444",
            line_width=1.5,
            annotation_text="⚠ Début stress",
            annotation_position="top",
        )

    # Z-score panel
    z_scores = result.get("z_scores_historique", [0] * len(dates))
    z_colors = ["#ef4444" if abs(z) > 2 else "#f97316" if abs(z) > 1.2 else "#4ade80" for z in z_scores]

    fig.add_trace(go.Bar(
        x=dates, y=z_scores,
        name="Z-score anomalie",
        marker_color=z_colors,
    ), row=2, col=1)

    fig.add_hline(y=1.8, line_dash="dot", line_color="#f97316", line_width=1, row=2, col=1)
    fig.add_hline(y=-1.8, line_dash="dot", line_color="#f97316", line_width=1, row=2, col=1)
    fig.add_hline(y=2.5, line_dash="dot", line_color="#ef4444", line_width=1, row=2, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,26,13,0.8)",
        font=dict(family="DM Sans", color="#8aad8a", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="#1e3a1e", zerolinecolor="#2d4a2d", tickfont=dict(color="#4d7a4d"))
    fig.update_yaxes(gridcolor="#1e3a1e", title_text="NDVI", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e3a1e", title_text="Z-score", row=2, col=1)

    return fig


def build_fleet_map(parcelles: list, filter_systeme: str = None, filter_gouvernorat: str = None) -> folium.Map:
    """Build filtered Leaflet map with stress-coded orchard markers."""
    m = folium.Map(location=[36.5, 9.8], zoom_start=8, tiles=None)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        control=True,
    ).add_to(m)
    
    folium.TileLayer(tiles="CartoDB dark_matter", name="Dark Map", control=True).add_to(m)

    for parcel in parcelles:
        centroid = parcel.get("centroide")
        if not centroid:
            continue
        
        # Filtres
        if filter_systeme and filter_systeme != "Tous":
            if parcel.get("metadata", {}).get("systeme") != filter_systeme:
                continue
        
        status = parcel["statut"]
        color = STATUS_COLORS[status]
        score = parcel["anomaly_score"]
        
        radius = 8 + score * 4

        popup_html = f"""
        <div style="font-family: 'DM Sans', sans-serif; background: #0d1a0d; color: #c8e6c4;
                    padding: 12px; border-radius: 8px; border: 1px solid #2d4a2d; min-width: 250px;">
          <div style="font-weight: 700; font-size: 14px; color: #a8d5a2; margin-bottom: 8px;">
            🫒 {parcel['id']}
          </div>
          <div style="font-size: 11px; color: #5a8a5a; margin-bottom: 6px;">
            {parcel['metadata']['systeme'].upper()} · {parcel['metadata']['n_observations']} acquisitions
          </div>
          <div style="background: {'#450a0a' if status == 'rouge' else '#431407' if status == 'orange' else '#14532d'};
                      color: {color}; padding: 6px 10px; border-radius: 4px; font-weight: 600;
                      font-size: 13px; margin-bottom: 8px;">
            {STATUS_LABELS[status]}
          </div>
          <div style="font-size: 12px; color: #8aad8a;">
            Score: <b style="color: {color}">{score:.2f}</b>
          </div>
          <div style="font-size: 11px; color: #5a8a5a; margin-top: 8px; line-height: 1.4;">
            📉 NDVI chute: {parcel['changement_regime'].get('chute_ndvi', 0)*100:.1f}%<br>
            🌡️ LST max: {parcel.get('stress_thermique', {}).get('max_lst', 'N/D')}°C
          </div>
        </div>
        """

        folium.CircleMarker(
            location=[centroid["lat"], centroid["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{parcel['id']} — {STATUS_LABELS[status]}",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def build_fleet_score_chart(parcelles: list, filter_systeme: str = None) -> go.Figure:
    """Bar chart des scores d'anomalie."""
    filtered = parcelles
    if filter_systeme and filter_systeme != "Tous":
        filtered = [p for p in parcelles if p.get("metadata", {}).get("systeme") == filter_systeme]
    
    if not filtered:
        return go.Figure()
    
    ids = [p["id"] for p in filtered]
    scores = [p["anomaly_score"] for p in filtered]
    statuts = [p["statut"] for p in filtered]
    colors = [STATUS_COLORS[s] for s in statuts]

    fig = go.Figure(go.Bar(
        x=ids, y=scores,
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
    ))

    fig.add_hline(y=1.8, line_dash="dot", line_color="#f97316", line_width=1)
    fig.add_hline(y=2.5, line_dash="dot", line_color="#ef4444", line_width=1)

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,26,13,0.8)",
        font=dict(family="DM Sans", color="#8aad8a"),
        title="📊 Score d'anomalie par parcelle",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1e3a1e", tickangle=45)
    fig.update_yaxes(gridcolor="#1e3a1e", title_text="Score composite", range=[0, 4.2])

    return fig


# ─────────────────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────

def render_result(result: dict):
    status = result["statut"]
    score = result["anomaly_score"]
    confidence = result["confidence"]
    expl = result.get("explication", {})
    thermal = result.get("stress_thermique", {})
    regime = result.get("changement_regime", {})
    quantiles = result.get("quantiles_seuils", {})

    badge_class = f"badge-{status}"
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
        <span class="status-badge {badge_class}">{STATUS_LABELS[status]}</span>
        <span style="font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800;
                     color: {STATUS_COLORS[status]};">{score:.2f}</span>
        <span style="color: #4d7a4d;">score composite</span>
        <span style="color: #2d4a2d;">|</span>
        <span style="color: #8aad8a;">Confiance: <b>{confidence*100:.0f}%</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Explication (conforme PDF)
    panel_class = f"explication-{status}" if status != "vert" else ""
    st.markdown(f"""
    <div class="explication-panel {panel_class}">
        <div style="font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
                    color: #c8e6c4; margin-bottom: 0.6rem;">
            {expl.get("message_principal", "")}
        </div>
        <div style="font-size: 0.85rem; color: #8aad8a; line-height: 1.6;">
            {"<br>".join(["▸ " + r for r in expl.get("raisons", [])])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if expl.get("recommandations"):
        st.markdown(f"""
        <div style="margin-top: 0.75rem; padding: 0.75rem 1rem; background: #0d1a0d;
                    border-radius: 8px; border: 1px solid #1e3a1e;">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: #4d7a4d; margin-bottom: 0.4rem;">
                📋 Recommandations
            </div>
            <ul style="margin: 0; padding-left: 1.2rem;">
                {"".join([f'<li style="color:#8aad8a;">{r}</li>' for r in expl["recommandations"]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Métriques
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Score composite", f"{score:.3f}")
    with col2:
        z = result["z_score_actuel"]
        st.metric("Z-score actuel", f"{z:.3f}", delta=f"{abs(z):.1f}σ" if z < -1 else None)
    with col3:
        ndvi_drop = regime.get("chute_ndvi", 0)
        st.metric("Chute NDVI", f"-{ndvi_drop:.3f}")
    with col4:
        stress_date = regime.get("date_debut_stress", "—")
        st.metric("Début stress", stress_date)
    with col5:
        if quantiles:
            st.metric("Seuil orange", f"{quantiles.get('p70', '—')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 NDVI & Baseline", "🌡️ Stress thermique", "🎯 Détails scoring", "📊 JSON brut"])

    with tab1:
        fig = build_ndvi_chart(result)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"ℹ️ {expl.get('contexte_systeme', '')}")

    with tab2:
        lst_fig = build_lst_chart(result)
        if lst_fig:
            st.plotly_chart(lst_fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LST max", f"{thermal.get('max_lst', 'N/D')}°C")
        with col2:
            st.metric("Jours > 38°C", thermal.get('stress_days_above_38C', 0))
        with col3:
            st.metric("Stress thermique", "🔥 Oui" if thermal.get('thermal_stress') else "✅ Non")

    with tab3:
        st.markdown('<div class="section-header">Décomposition du score composite</div>', unsafe_allow_html=True)
        breakdown = pd.DataFrame({
            "Composante": ["Z-score (écart normalisé)", "Isolation Forest", "Chute NDVI", "Bonus LST"],
            "Valeur": [
                abs(result["z_score_actuel"]),
                result["iso_score"],
                regime.get("chute_ndvi", 0) * 3,
                0.5 if thermal.get("thermal_stress") else 0,
            ],
            "Pondération": [0.45, 0.35, 0.20, "N/A"],
            "Contribution": [
                abs(result["z_score_actuel"]) * 0.45,
                result["iso_score"] * 0.35,
                regime.get("chute_ndvi", 0) * 3 * 0.20,
                0.5 if thermal.get("thermal_stress") else 0,
            ]
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        
        if quantiles:
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.75rem; background: #0d1a0d; border-radius: 8px;">
                <div style="color: #4d7a4d; font-size: 0.75rem;">📊 Seuillage par quantiles historiques</div>
                <div style="color: #8aad8a; font-size: 0.85rem;">
                    Seuil orange (P70): {quantiles['p70']} | Seuil rouge (P85): {quantiles['p85']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.json(result)


def build_lst_chart(result: dict) -> go.Figure:
    """LST thermal stress chart."""
    dates = result.get("dates", [])
    thermal = result.get("stress_thermique", {})
    lst = result.get("lst", [])

    if not dates or not lst:
        return None

    fig = go.Figure()
    fig.add_hline(y=35, line_dash="dot", line_color="#f97316", line_width=1.5,
                  annotation_text="Seuil stress (35°C)")
    fig.add_hline(y=38, line_dash="dot", line_color="#ef4444", line_width=1.5,
                  annotation_text="Seuil critique (38°C)")

    colors = ["#ef4444" if v > 38 else "#f97316" if v > 35 else "#4ade80" for v in lst]
    fig.add_trace(go.Scatter(
        x=dates, y=lst,
        mode="lines+markers",
        name="LST (°C)",
        line=dict(color="#fb923c", width=2.5),
        marker=dict(size=9, color=colors, line=dict(color="#0a0f0a", width=2)),
    ))

    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,26,13,0.8)",
        font=dict(family="DM Sans", color="#8aad8a"),
        title="🌡️ Température de surface (LST) — Détection stress thermique pré-NDVI",
    )
    fig.update_xaxes(gridcolor="#1e3a1e")
    fig.update_yaxes(gridcolor="#1e3a1e", title_text="LST (°C)")

    return fig


def render_fleet_view(fleet_data: dict):
    parcelles = fleet_data.get("parcelles", [])
    resume = fleet_data.get("resume", {})

    # Filtres
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        filter_systeme = st.selectbox("🌿 Filtrer par système", ["Tous", "extensif", "intensif", "hyper-intensif"])
    with col2:
        filter_gouvernorat = st.selectbox("📍 Filtrer par gouvernorat", GOUVERNORATS)
    with col3:
        if st.button("🔄 Réinitialiser filtres", use_container_width=True):
            filter_systeme = "Tous"
            filter_gouvernorat = "Tous"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Application des filtres
    filtered = parcelles
    if filter_systeme != "Tous":
        filtered = [p for p in filtered if p.get("metadata", {}).get("systeme") == filter_systeme]
    # Note: Le filtrage par gouvernorat nécessiterait des données de localisation supplémentaires

    # Stats
    stats = {
        "total": len(filtered),
        "vert": len([p for p in filtered if p["statut"] == "vert"]),
        "orange": len([p for p in filtered if p["statut"] == "orange"]),
        "rouge": len([p for p in filtered if p["statut"] == "rouge"]),
        "score_moyen": sum(p["anomaly_score"] for p in filtered) / len(filtered) if filtered else 0,
    }

    st.markdown(f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-card" style="flex:1; text-align:center;">
            <div class="metric-label">Parcelles affichées</div>
            <div class="metric-value">{stats['total']}</div>
        </div>
        <div class="metric-card" style="flex:1; text-align:center; border-color:#22c55e;">
            <div class="metric-label">🟢 Normales</div>
            <div class="metric-value" style="color:#4ade80;">{stats['vert']}</div>
        </div>
        <div class="metric-card" style="flex:1; text-align:center; border-color:#f97316;">
            <div class="metric-label">🟠 Vigilance</div>
            <div class="metric-value" style="color:#fb923c;">{stats['orange']}</div>
        </div>
        <div class="metric-card" style="flex:1; text-align:center; border-color:#ef4444;">
            <div class="metric-label">🔴 Alertes</div>
            <div class="metric-value" style="color:#f87171;">{stats['rouge']}</div>
        </div>
        <div class="metric-card" style="flex:1; text-align:center;">
            <div class="metric-label">Score moyen</div>
            <div class="metric-value">{stats['score_moyen']:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Carte
    st.markdown('<div class="section-header">🗺️ Carte des oliveraies EZZAYRA</div>', unsafe_allow_html=True)
    m = build_fleet_map(filtered, filter_systeme if filter_systeme != "Tous" else None, filter_gouvernorat if filter_gouvernorat != "Tous" else None)
    st_folium(m, height=450, use_container_width=True)

    # Graphique des scores
    score_fig = build_fleet_score_chart(filtered, filter_systeme if filter_systeme != "Tous" else None)
    if score_fig.data:
        st.plotly_chart(score_fig, use_container_width=True)

    # Tableau
    st.markdown('<div class="section-header">📋 Tableau de bord détaillé</div>', unsafe_allow_html=True)
    rows = []
    for p in filtered:
        rows.append({
            "ID": p["id"],
            "Système": p["metadata"]["systeme"],
            "Statut": STATUS_LABELS[p["statut"]],
            "Score": p["anomaly_score"],
            "Z-score": p["z_score_actuel"],
            "Chute NDVI": f"{p['changement_regime'].get('chute_ndvi', 0)*100:.1f}%",
            "LST max": f"{p.get('stress_thermique', {}).get('max_lst', '—')}°C",
            "Début stress": p["changement_regime"].get("date_debut_stress", "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800;
                    color: #a8d5a2;">🫒 DIAGNOSTIC PARCELLE</div>
        <div style="font-size: 0.7rem; color: #4d7a4d; margin-bottom: 1.5rem;">
            Sujet 03 · Hack The Harvest 2026
        </div>
        """, unsafe_allow_html=True)

        orchard_id = st.text_input("ID Parcelle", value="O_2026_307")
        systeme = st.selectbox("Système de conduite", ["intensif", "extensif", "hyper-intensif"])
        lat = st.number_input("Latitude", value=36.5, min_value=30.0, max_value=38.0, format="%.4f")
        lon = st.number_input("Longitude", value=9.8, min_value=7.5, max_value=12.0, format="%.4f")

        dates_raw = st.text_area(
            "Dates acquisitions (ISO, une par ligne)",
            value="\n".join(["2026-01-10", "2026-01-26", "2026-02-11", "2026-02-27",
                             "2026-03-15", "2026-03-31", "2026-04-16", "2026-05-02"]),
            height=120,
        )

        ndvi_raw = st.text_area(
            "Valeurs NDVI (une par ligne)",
            value="\n".join(["0.56", "0.58", "0.57", "0.55", "0.50", "0.44", "0.38", "0.32"]),
            height=120,
        )

        lst_raw = st.text_area("LST (°C) — optionnel", 
                               value="\n".join(["22.1", "21.5", "23.8", "24.2", "28.5", "32.1", "36.8", "39.2"]),
                               height=100)

        run_btn = st.button("🔍 Lancer diagnostic", use_container_width=True)
        demo_btn = st.button("🎯 Charger démo jury", use_container_width=True)

    return {
        "orchard_id": orchard_id, "systeme": systeme, "lat": lat, "lon": lon,
        "dates_raw": dates_raw, "ndvi_raw": ndvi_raw, "lst_raw": lst_raw,
        "run": run_btn, "demo": demo_btn,
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🫒 Agronomic Twin</div>
        <div class="main-subtitle">
            Détection précoce d'anomalies · Oliveraies tunisiennes · Sujet 03 — Hack The Harvest 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    sidebar = render_sidebar()

    if "result" not in st.session_state:
        st.session_state.result = None
    if "fleet" not in st.session_state:
        st.session_state.fleet = None

    tab_diag, tab_fleet = st.tabs(["🔬 Diagnostic parcelle", "🗺️ Vue flotte"])

    with tab_diag:
        if sidebar["demo"]:
            with st.spinner("Chargement démo jury..."):
                try:
                    resp = requests.get(f"{API_URL}/api/demo", timeout=30)
                    if resp.status_code == 200:
                        st.session_state.result = resp.json()
                        st.success("✅ Démo chargée — oliveraie O_2026_307")
                    else:
                        st.error("Backend non joignable")
                except Exception as e:
                    st.error(f"Erreur: {e}")

        if sidebar["run"]:
            try:
                dates = [d.strip() for d in sidebar["dates_raw"].strip().splitlines() if d.strip()]
                ndvi = [float(v.strip()) for v in sidebar["ndvi_raw"].strip().splitlines() if v.strip()]
                lst = []
                if sidebar["lst_raw"].strip():
                    lst = [float(v.strip()) for v in sidebar["lst_raw"].strip().splitlines() if v.strip()]

                if len(dates) != len(ndvi):
                    st.error(f"Dates ({len(dates)}) ≠ NDVI ({len(ndvi)})")
                elif len(dates) < 3:
                    st.error("Minimum 3 acquisitions")
                else:
                    payload = {
                        "oliveraie": {
                            "id": sidebar["orchard_id"],
                            "systeme": sidebar["systeme"],
                            "centroide": {"lat": sidebar["lat"], "lon": sidebar["lon"]},
                        },
                        "dates": dates,
                        "ndvi": ndvi,
                        "lst": lst if lst else None,
                        "fetch_weather": False,
                    }
                    with st.spinner("Analyse en cours..."):
                        result = call_api("/api/diagnostic-anomalie", payload)
                    if result:
                        st.session_state.result = result
            except Exception as e:
                st.error(f"Erreur: {e}")

        if st.session_state.result:
            render_result(st.session_state.result)
        else:
            st.info("👈 Entrez les données d'une oliveraie ou cliquez sur 'Charger démo jury'")

    with tab_fleet:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Charger la flotte démo", use_container_width=True):
                with st.spinner("Analyse de la flotte..."):
                    fleet = get_demo_fleet()
                    if fleet:
                        st.session_state.fleet = fleet
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state.fleet = None

        if st.session_state.fleet:
            render_fleet_view(st.session_state.fleet)
        else:
            st.info("👈 Cliquez sur 'Charger la flotte démo' pour voir la carte des oliveraies")


if __name__ == "__main__":
    main()