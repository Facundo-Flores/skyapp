# web/app_streamlit.py
from __future__ import annotations
from pathlib import Path
import sys
import io
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.sky_core import compute_altaz, make_figure, MAGS  # noqa: E402

# Geolocalización (opcional)
try:
    from streamlit_geolocation import streamlit_geolocation

    GEO_OK = True
except Exception:
    GEO_OK = False

# Autorefresh (opcional)
try:
    from streamlit_autorefresh import st_autorefresh

    AUTOREFRESH_OK = True
except Exception:
    AUTOREFRESH_OK = False

TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")


def fmt_ar(dt: datetime) -> str:
    return dt.astimezone(TZ_AR).strftime("%d-%m-%Y %H:%M:%S")


def set_astro_theme() -> None:
    bg = "#0A0E1A"              # Azul noche muy oscuro
    panel = "rgba(30, 41, 59, 0.45)"  # Slate oscuro semi-transparente
    text = "#E2E8F0"            # Gris claro suave
    muted = "#94A3B8"           # Gris medio para secundario
    border = "rgba(148, 163, 184, 0.25)"
    primary_a = "#6366F1"       # Índigo principal
    primary_b = "#4F46E5"       # Índigo más oscuro
    shadow = "0 10px 25px rgba(0,0,0,0.45)"
    field_bg = "rgba(30, 41, 59, 0.6)"
    accent = "#CBD5E1"

    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

          .block-container {{ padding: 1rem 1.2rem !important; max-width: 1400px; }}
          [data-testid="stAppViewContainer"] {{
            background: {bg};
            background-image: radial-gradient(circle at 10% 20%, rgba(99,102,241,0.08) 0%, transparent 50%);
          }}
          .stAppHeader {{ background: transparent; }}
          [data-testid="stSidebar"] {{
            background: {bg};
            border-right: 1px solid {border};
          }}
          h1, h2, h3, h4 {{
            color: {text} !important;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.5px;
          }}
          p, span, label, div {{
            color: {text} !important;
            font-family: 'Roboto Mono', monospace;
          }}
          .muted {{ color: {muted}; font-size: 0.90rem; }}
          .tiny {{ color: {muted}; font-size: 0.82rem; }}
          .card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: {shadow};
            margin-bottom: 1rem;
          }}
          .plot-wrap {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 12px;
            box-shadow: {shadow};
          }}
          .stNumberInput input, .stTextInput input, .stDateInput input, .stTimeInput input {{
            background: {field_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
          }}
          .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {primary_a}, {primary_b}) !important;
            color: white !important;
            font-family: 'Orbitron', sans-serif;
            font-weight: 500 !important;
            border-radius: 12px !important;
            padding: 0.7rem 1rem !important;
          }}
          .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
            background: transparent;
          }}
          .stTabs [data-baseweb="tab"] {{
            color: {muted} !important;
            font-family: 'Orbitron', sans-serif;
          }}
          .stTabs [aria-selected="true"] {{
            color: {text} !important;
            border-bottom-color: {primary_a} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="SkyMap— Mapa Celestial", layout="wide", initial_sidebar_state="expanded")
set_astro_theme()

# Session defaults con más opciones pro
st.session_state.setdefault("place_label", "Vicente López, Buenos Aires")
st.session_state.setdefault("lat", -34.51)
st.session_state.setdefault("lon", -58.48)
st.session_state.setdefault("alt", 22.0)
st.session_state.setdefault("time_mode", "Ahora")
st.session_state.setdefault("selected_dt_ar", datetime.now(TZ_AR))
st.session_state.setdefault("show_horizon", True)
st.session_state.setdefault("selected_objects", list(MAGS.keys()))

# -------------------------
# Sidebar mejorado: tabs para config
# -------------------------
with st.sidebar:
    st.markdown("### 🌌 SkyMap")
    st.markdown('<div class="muted">Mapa interactivo del cielo: Sol, Luna y planetas desde tu ubicación.</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📍 Ubicación", "⏱️ Tiempo"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        mode = st.radio("Modo de ubicación", ["Navegador", "Manual"], horizontal=True, label_visibility="collapsed")
        lat = st.session_state.lat
        lon = st.session_state.lon
        if mode == "Navegador":
            use_geo = st.toggle("Usar mi ubicación", value=True, disabled=not GEO_OK)
            if use_geo and GEO_OK:
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") and loc.get("longitude"):
                    lat = float(loc["latitude"])
                    lon = float(loc["longitude"])
                    st.markdown(f'<div class="tiny">📍 {lat:.5f}, {lon:.5f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="tiny">Permití ubicación en el navegador.</div>', unsafe_allow_html=True)
            elif not GEO_OK:
                st.markdown('<div class="tiny">Geolocalización no disponible.</div>', unsafe_allow_html=True)
            with st.expander("Ajuste manual", expanded=False):
                lat = st.number_input("Latitud (°)", value=float(lat), format="%.6f")
                lon = st.number_input("Longitud (°)", value=float(lon), format="%.6f")
        else:
            lat = st.number_input("Latitud (°)", value=float(lat), format="%.6f")
            lon = st.number_input("Longitud (°)", value=float(lon), format="%.6f")
        alt = st.number_input("Altitud (m)", value=float(st.session_state.alt), format="%.1f")
        place_label = st.text_input("Etiqueta del lugar", value=st.session_state.place_label)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        time_mode = st.radio("Modo de tiempo", ["Ahora", "Personalizado"], horizontal=True)
        selected_dt_ar = st.session_state.selected_dt_ar
        if time_mode == "Personalizado":
            date_part = st.date_input("Fecha", value=selected_dt_ar.date())
            time_part = st.time_input("Hora", value=selected_dt_ar.time())
            selected_dt_ar = datetime.combine(date_part, time_part).replace(tzinfo=TZ_AR)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Visualización**")
    st.session_state.show_horizon = st.toggle("Mostrar horizonte", value=st.session_state.show_horizon)
    st.session_state.selected_objects = st.multiselect(
        "Objetos a mostrar",
        options=list(MAGS.keys()),
        default=st.session_state.selected_objects
    )
    auto = st.toggle("Actualizar automático (cada 10 s)", value=False)
    if auto and not AUTOREFRESH_OK:
        st.info("Instalá `streamlit-autorefresh` para esta función.")
    apply_btn = st.button("Aplicar cambios", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Aplicar cambios
if apply_btn:
    st.session_state.lat = float(lat)
    st.session_state.lon = float(lon)
    st.session_state.alt = float(alt)
    st.session_state.place_label = (place_label or "").strip()
    st.session_state.time_mode = time_mode
    st.session_state.selected_dt_ar = selected_dt_ar
    st.rerun()

# Autorefresco
if auto and AUTOREFRESH_OK:
    st_autorefresh(interval=10_000, key="auto_refresh")

# -------------------------
# Main: layout más pro, con tabs para vistas
# -------------------------
lat = float(st.session_state.lat)
lon = float(st.session_state.lon)
alt = float(st.session_state.alt)
place = (st.session_state.place_label or "").strip()

# Tiempo
if st.session_state.time_mode == "Ahora":
    dt_ar = datetime.now(TZ_AR)
else:
    dt_in = st.session_state.selected_dt_ar
    dt_ar = dt_in if dt_in.tzinfo else dt_in.replace(tzinfo=TZ_AR)
t_utc = dt_ar.astimezone(ZoneInfo("UTC"))
t_astropy = Time(t_utc)

if place and place.strip():
    header_title = f"SkyMap - {place}"
else:
    header_title = f"SkyMap - {lat:.4f}°S, {lon:.4f}°O"

st.markdown(f"# 🌌 {header_title}")
st.markdown(
    f'<div class="muted">Hora Argentina: {fmt_ar(dt_ar)} • Modo: <b>{st.session_state.time_mode}</b></div>',
    unsafe_allow_html=True
)

main_tab1, main_tab2 = st.tabs(["🗺️ Mapa", "📊 Datos"])

with main_tab1:
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    altaz, _ = compute_altaz(t_astropy, location, nombres=st.session_state.selected_objects)
    plot_title = f"Cielo visible — {fmt_ar(dt_ar)}"
    fig = make_figure(altaz, title=plot_title, mostrar_horizonte=st.session_state.show_horizon)
    fig.set_size_inches(8, 8)  # Más grande para pro
    preview = io.BytesIO()
    fig.savefig(preview, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    preview.seek(0)
    download = io.BytesIO()
    fig.savefig(download, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())  # Mayor DPI
    download.seek(0)
    plt.close(fig)
    st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
    st.image(preview.getvalue(), use_container_width=True)
    fname = f"astroview_{dt_ar.strftime('%Y%m%d_%H%M%S')}.png"
    st.download_button("⬇️ Descargar PNG (Alta Res)", data=download, file_name=fname, mime="image/png",
                       use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with main_tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estado actual")
        st.metric("Hora (Argentina)", fmt_ar(dt_ar))
        st.markdown(f"**Lat/Lon/Alt** \n`{lat:.6f}, {lon:.6f}, {alt:.1f} m`")
        if place:
            st.markdown(f"**Lugar** \n`{place}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Objetos celestes")
        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
        _, table = compute_altaz(t_astropy, location, nombres=st.session_state.selected_objects)
        rows = [
            {
                "Objeto": o.nombre,
                "Altura (°)": round(o.alt_deg, 2),
                "Azimut (°)": round(o.az_deg, 2),
                "Magnitud": o.mag,
                "Visible": "Sí" if o.visible else "No",
            }
            for o in table
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="tiny">Azimut: 0° Norte, 90° Este, 180° Sur, 270° Oeste. Magnitud: menor = más brillante.</div>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)