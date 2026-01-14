from __future__ import annotations

from pathlib import Path
import sys
import io

import streamlit as st
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sky_core import compute_altaz, make_figure  # noqa: E402

try:
    from streamlit_geolocation import streamlit_geolocation
    GEO_OK = True
except Exception:
    GEO_OK = False


def fmt_ar_from_time(t_utc: Time) -> str:
    t_art = t_utc - 3 * u.hour
    return t_art.datetime.strftime("%d-%m-%Y, %H:%M:%S") + " (UTC-3)"


def set_css(mode: str) -> None:
    if mode == "light":
        bg = "#F6F7FB"
        panel = "#FFFFFF"
        text = "#101828"
        muted = "#667085"
        border = "rgba(16, 24, 40, 0.10)"
        primary_a = "#2563EB"
        primary_b = "#1D4ED8"
        shadow = "0 10px 26px rgba(16,24,40,0.10)"
    else:
        bg = "#0B1020"
        panel = "rgba(255,255,255,0.04)"
        text = "#E6E9F2"
        muted = "rgba(230,233,242,0.70)"
        border = "rgba(255,255,255,0.10)"
        primary_a = "#4F8CFF"
        primary_b = "#2F6BFF"
        shadow = "0 14px 32px rgba(0,0,0,0.35)"

    st.markdown(
        f"""
        <style>
          .block-container {{
            padding-top: 0.9rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
            max-width: 1400px;
          }}

          [data-testid="stAppViewContainer"] {{
            background: {bg};
          }}
          .stAppHeader {{ background: transparent; }}

          /* Sidebar: compacto y profesional */
          [data-testid="stSidebar"] {{
            background: {bg};
            border-right: 1px solid {border};
          }}
          [data-testid="stSidebar"] .block-container {{
            padding-top: 0.9rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
          }}

          h1, h2, h3, h4, p, span, label {{
            color: {text};
          }}
          .muted {{ color: {muted}; font-size: 0.92rem; }}
          .tiny {{ color: {muted}; font-size: 0.85rem; }}

          .card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 10px 10px;
            box-shadow: {shadow};
          }}

          .plot-wrap {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 10px 10px;
            box-shadow: {shadow};
          }}

          .stButton>button {{
            border-radius: 12px !important;
            padding: 0.62rem 0.9rem !important;
            border: 1px solid {border} !important;
          }}
          .stButton>button[kind="primary"] {{
            background: linear-gradient(90deg, {primary_a}, {primary_b}) !important;
            border: 0 !important;
            color: white !important;
            font-weight: 800 !important;
          }}

          .element-container {{ margin-bottom: 0.55rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="Mapa estelar", layout="wide", initial_sidebar_state="expanded")

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"
if "place_label" not in st.session_state:
    st.session_state.place_label = "Vicente López, Buenos Aires"
if "lat" not in st.session_state:
    st.session_state.lat = -34.51
if "lon" not in st.session_state:
    st.session_state.lon = -58.48
if "alt" not in st.session_state:
    st.session_state.alt = 22.0

# Sidebar (solo controles)
with st.sidebar:
    st.markdown("### ✨ SkyApp")
    st.markdown('<div class="muted">Mapa estelar visible desde tu ubicación</div>', unsafe_allow_html=True)

    ui_theme = st.radio("Tema UI", ["dark", "light"], index=0 if st.session_state.ui_theme == "dark" else 1, horizontal=True)
    st.session_state.ui_theme = ui_theme

set_css(st.session_state.ui_theme)
plot_theme = st.session_state.ui_theme  # mantenemos consistente

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Ubicación**")

    mode = st.radio(
        "Modo",
        ["Navegador", "Manual"],
        horizontal=True,
        help="Navegador usa lat/lon por geolocalización. Manual permite ingresarlas a mano.",
    )

    if mode == "Navegador":
        use_geo = st.toggle("Usar mi ubicación", value=True, disabled=not GEO_OK)
        if use_geo and GEO_OK:
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") and loc.get("longitude"):
                st.session_state.lat = float(loc["latitude"])
                st.session_state.lon = float(loc["longitude"])
                st.markdown(f'<div class="tiny">📍 {st.session_state.lat:.5f}, {st.session_state.lon:.5f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tiny">Permití ubicación en tu navegador.</div>', unsafe_allow_html=True)
        elif not GEO_OK:
            st.warning("Instalá streamlit-geolocation para usar el navegador.")
    else:
        st.session_state.lat = st.number_input("Latitud (°)", value=float(st.session_state.lat), format="%.6f")
        st.session_state.lon = st.number_input("Longitud (°)", value=float(st.session_state.lon), format="%.6f")

    st.session_state.alt = st.number_input("Altitud (m)", value=float(st.session_state.alt), format="%.1f")

    # Etiqueta de lugar (para el título)
    st.session_state.place_label = st.text_input(
        "Etiqueta del lugar",
        value=st.session_state.place_label,
        help="Ej: 'Vicente López, Buenos Aires'. Si lo dejás vacío, se mostrará lat/lon."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Actualización**")
    auto = st.toggle("Auto-refrescar (10s)", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    generate = st.button("Generar ahora", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Auto refresh (temprano)
if auto:
    st.autorefresh(interval=10_000, key="auto_refresh")

# Render inicial
if "has_rendered" not in st.session_state:
    st.session_state.has_rendered = True
    generate = True

# -------------------------
# Main
# -------------------------
t_utc = Time.now()
lat = float(st.session_state.lat)
lon = float(st.session_state.lon)
alt = float(st.session_state.alt)
place = (st.session_state.place_label or "").strip()
title_place = place if place else f"{lat:.4f}, {lon:.4f}"

st.markdown(f"# Mapa estelar visible — {title_place}")
st.markdown(f'<div class="muted">Actualizado: {fmt_ar_from_time(t_utc)} • Tema: {plot_theme}</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 2.2], gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estado")
    st.metric("Hora Argentina", fmt_ar_from_time(t_utc))
    st.markdown(f"**Lat/Lon/Alt**  \n`{lat:.6f}, {lon:.6f}, {alt:.1f} m`")
    if place:
        st.markdown(f"**Lugar**  \n`{place}`")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    if generate or auto:
        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)

        t_now = Time.now()  # siempre ahora
        altaz = compute_altaz(t_now, location)

        # Título del plot: más corto (lo “largo” va en el header de la página)
        plot_title = f"Cielo visible — {fmt_ar_from_time(t_now)}"

        fig = make_figure(altaz, title=plot_title, theme=plot_theme)

        # Tamaño web-friendly (evita scroll)
        fig.set_size_inches(6.0, 6.0)

        preview = io.BytesIO()
        fig.savefig(preview, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        preview.seek(0)

        download = io.BytesIO()
        fig.savefig(download, format="png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
        download.seek(0)

        plt.close(fig)

        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.image(preview.getvalue(), use_container_width=True)

        st.download_button(
            "⬇️ Descargar PNG",
            data=download,
            file_name=f"cielo_{(t_now - 3*u.hour).datetime.strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
        )
        st.markdown("</div>", unsafe_allow_html=True)
