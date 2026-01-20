from __future__ import annotations
from pathlib import Path

import sys
import io
import os
import requests

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sky_core import compute_altaz, make_figure, MAGS  # noqa: E402
from core.sky_3d import build_sky_3d_html


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


def detectar_mobile() -> bool:
    """
    Detecta mobile de forma robusta:
      1) streamlit-browser-engine (si está)
      2) heurística por User-Agent (fallback)
    """
    # 1) streamlit-browser-engine
    try:
        from browser_detection import browser_detection_engine
        info = browser_detection_engine() or {}
        # algunas versiones usan is_mobile / mobile / device_type, etc.
        if "is_mobile" in info:
            return bool(info["is_mobile"])
        if info.get("device_type"):
            return str(info["device_type"]).lower() in ("mobile", "tablet")
        if info.get("is_tablet"):
            return bool(info.get("is_tablet"))
    except Exception:
        pass

    # 2) Fallback por User-Agent (cuando Streamlit expone headers)
    try:
        # Streamlit >= 1.27 aprox tiene st.context.headers (puede variar)
        ua = ""
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            ua = (st.context.headers.get("User-Agent") or st.context.headers.get("user-agent") or "")
        ua_l = ua.lower()
        if any(k in ua_l for k in ["android", "iphone", "ipad", "ipod", "mobile", "tablet"]):
            return True
    except Exception:
        pass

    return False



def get_bdc_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("BIGDATACLOUD_API_KEY", None)
    except Exception:
        key = None
    return key or os.environ.get("BIGDATACLOUD_API_KEY")


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode_bigdatacloud(lat_q: float, lon_q: float, locality_language: str = "es") -> dict:
    key = get_bdc_key()
    if not key:
        raise RuntimeError("Falta BIGDATACLOUD_API_KEY (st.secrets o variable de entorno).")

    url = "https://api-bdc.net/data/reverse-geocode"
    params = {
        "latitude": lat_q,
        "longitude": lon_q,
        "localityLanguage": locality_language,
        "key": key,
    }

    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def build_place_label(payload: dict) -> str:
    locality = (
        payload.get("locality")
        or payload.get("city")
        or payload.get("localityInfo", {}).get("administrative", [{}])[0].get("name")
        or ""
    )
    province = payload.get("principalSubdivision") or ""
    country = payload.get("countryName") or ""

    parts = [p.strip() for p in [locality, province, country] if p and str(p).strip()]
    label = ", ".join(parts)

    if not label:
        label = payload.get("localityInfo", {}).get("informative", [{}])[0].get("name", "")

    return label.strip()


def set_astro_theme() -> None:
    bg = "#0A0E1A"
    panel = "rgba(30, 41, 59, 0.45)"
    text = "#E2E8F0"
    muted = "#94A3B8"
    border = "rgba(148, 163, 184, 0.25)"
    primary_a = "#6366F1"
    primary_b = "#4F46E5"
    shadow = "0 10px 25px rgba(0,0,0,0.45)"
    field_bg = "rgba(30, 41, 59, 0.6)"

    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

          /* Layout base */
          .block-container {{
            padding: 1rem 1.2rem !important;
            max-width: 1400px;
          }}

          [data-testid="stAppViewContainer"] {{
            background: {bg};
            background-image: radial-gradient(circle at 10% 20%, rgba(99,102,241,0.08) 0%, transparent 50%);
          }}
          .stAppHeader {{ background: transparent; }}

          [data-testid="stSidebar"] {{
            background: {bg};
            border-right: 1px solid {border};
          }}

          /* Tipografías */
          h1, h2, h3, h4 {{
            color: {text} !important;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.5px;
          }}

          p, span, label, div {{
            color: {text} !important;
            font-family: 'Roboto Mono', monospace;
          }}

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
          
          /* Footer sutil */
          .app-footer {{
            position: fixed;
            bottom: 8px;
            left: 12px;
            z-index: 999;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.75rem;
            color: rgba(226, 232, 240, 0.55);
            pointer-events: auto;
          }}
          
          .app-footer a {{
            color: #8BE9FD;
            text-decoration: none;
            pointer-events: auto;
          }}
          
          .app-footer a:hover {{
            text-decoration: underline;
            opacity: 1.0;
          }}

          /* En celu más chico */
          @media (max-width: 640px) {{
           .app-footer {{
            font-size: 0.68rem;
            bottom: 6px;
            left: 8px;
           }}
          }}

          /* --- En celu achicar tipografías --- */
          @media (max-width: 640px) {{
            .block-container {{
              padding: 0.65rem 0.75rem !important;
            }}

            h1 {{
              font-size: 1.65rem !important;
              letter-spacing: 0.2px !important;
              line-height: 1.15 !important;
              margin-bottom: 0.35rem !important;
            }}
            h2 {{
              font-size: 1.20rem !important;
              line-height: 1.20 !important;
              margin-bottom: 0.25rem !important;
            }}
            h3 {{
              font-size: 1.05rem !important;
              line-height: 1.20 !important;
              margin-bottom: 0.20rem !important;
            }}

            /* Texto general */
            p, span, label, div {{
              font-size: 0.90rem !important;
              line-height: 1.30 !important;
            }}

            /*  */
            .muted {{ font-size: 0.82rem !important; color: {muted} !important; }}
            .tiny  {{ font-size: 0.76rem !important; color: {muted} !important; }}

            .card {{
              padding: 12px 12px !important;
              margin-bottom: 0.75rem !important;
            }}
            .plot-wrap {{
              padding: 10px !important;
            }}

            [data-testid="stMetricValue"] {{
              font-size: 1.45rem !important;
              line-height: 1.05 !important;
            }}
            [data-testid="stMetricLabel"] {{
              font-size: 0.85rem !important;
            }}

            [data-testid="stDataFrame"] {{
              font-size: 0.86rem !important;
            }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )




def _ui_modo_etiquetas_to_core(value: str) -> str:
    m = {
        "Inteligentes": "inteligentes",
        "Todas": "todas",
        "Top (más brillantes)": "top",
    }
    return m.get(value, "inteligentes")


def render_controles(prefix: str) -> dict:
    tab1, tab2, tab3 = st.tabs(["📍 Ubicación", "⏱️ Tiempo", "👁️ Visualización"])

    draft = {
        "lat": float(st.session_state.lat),
        "lon": float(st.session_state.lon),
        "alt": float(st.session_state.alt),
        "place_label": (st.session_state.place_label or "").strip(),
        "time_mode": st.session_state.time_mode,
        "selected_dt_ar": st.session_state.selected_dt_ar,
        "show_horizon": bool(st.session_state.show_horizon),
        "selected_objects": list(st.session_state.selected_objects),
        "auto_zoom": bool(st.session_state.auto_zoom),
        "zoom_rmax": int(st.session_state.zoom_rmax),
        "modo_etiquetas": st.session_state.modo_etiquetas,
        "max_etiquetas": int(st.session_state.max_etiquetas),
        "separacion_etiquetas_px": int(st.session_state.separacion_etiquetas_px),
        "cluster_px": int(st.session_state.cluster_px),
    }

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        mode = st.radio(
            "Modo de ubicación",
            ["Navegador", "Manual"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}_modo_ubicacion",
        )

        lat = float(draft["lat"])
        lon = float(draft["lon"])

        if mode == "Navegador":
            use_geo = st.toggle(
                "Usar mi ubicación",
                value=True,
                disabled=not GEO_OK,
                key=f"{prefix}_usar_geo",
            )

            if use_geo and GEO_OK:
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") and loc.get("longitude"):
                    lat = float(loc["latitude"])
                    lon = float(loc["longitude"])

                    if not st.session_state.get("place_label_user_edited", False):
                        try:
                            lat_q = round(lat, 3)
                            lon_q = round(lon, 3)
                            payload = reverse_geocode_bigdatacloud(lat_q, lon_q, locality_language="es")
                            auto_label = build_place_label(payload)
                            if auto_label:
                                st.session_state.place_label = auto_label
                                st.session_state.place_label_source = "auto"
                            else:
                                st.session_state.place_label = ""
                                st.session_state.place_label_source = "auto"
                        except Exception:
                            st.session_state.place_label_source = "auto"

                    st.markdown(f'<div class="tiny">📍 {lat:.5f}, {lon:.5f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="tiny">Permití ubicación en el navegador.</div>', unsafe_allow_html=True)
            elif not GEO_OK:
                st.markdown('<div class="tiny">Geolocalización no disponible.</div>', unsafe_allow_html=True)

            with st.expander("Ajuste manual", expanded=False):
                lat = st.number_input("Latitud (°)", value=float(lat), format="%.6f", key=f"{prefix}_lat_exp")
                lon = st.number_input("Longitud (°)", value=float(lon), format="%.6f", key=f"{prefix}_lon_exp")
        else:
            lat = st.number_input("Latitud (°)", value=float(lat), format="%.6f", key=f"{prefix}_lat")
            lon = st.number_input("Longitud (°)", value=float(lon), format="%.6f", key=f"{prefix}_lon")

        alt = st.number_input("Altitud (m)", value=float(draft["alt"]), format="%.1f", key=f"{prefix}_alt")

        place_label = st.text_input(
            "Etiqueta del lugar",
            value=(st.session_state.place_label or "").strip(),
            help="Si la dejás vacía, la app intenta poner una etiqueta automática según tu ubicación.",
            key=f"{prefix}_place_label",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        draft["lat"] = float(lat)
        draft["lon"] = float(lon)
        draft["alt"] = float(alt)
        draft["place_label"] = (place_label or "").strip()

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        time_mode = st.radio(
            "Modo de tiempo",
            ["Ahora", "Personalizado"],
            horizontal=True,
            key=f"{prefix}_time_mode",
            index=0 if draft["time_mode"] == "Ahora" else 1,
        )

        selected_dt_ar = draft["selected_dt_ar"]
        if time_mode == "Personalizado":
            date_part = st.date_input("Fecha", value=selected_dt_ar.date(), key=f"{prefix}_date")
            time_part = st.time_input("Hora", value=selected_dt_ar.time(), key=f"{prefix}_time")
            selected_dt_ar = datetime.combine(date_part, time_part).replace(tzinfo=TZ_AR)

        st.markdown("</div>", unsafe_allow_html=True)

        draft["time_mode"] = time_mode
        draft["selected_dt_ar"] = selected_dt_ar

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Visualización**")

        show_horizon = st.toggle("Mostrar horizonte", value=bool(draft["show_horizon"]), key=f"{prefix}_show_horizon")

        selected_objects = st.multiselect(
            "Objetos a mostrar",
            options=list(MAGS.keys()),
            default=list(draft["selected_objects"]),
            key=f"{prefix}_selected_objects",
        )

        auto_zoom = st.toggle(
            "Auto zoom (encuadrar objetos)",
            value=bool(draft["auto_zoom"]),
            key=f"{prefix}_auto_zoom",
        )

        zoom_rmax = st.slider(
            "Zoom manual (más bajo = más cerca)",
            min_value=18,
            max_value=90,
            value=int(draft["zoom_rmax"]),
            step=1,
            disabled=auto_zoom,
            help="90 = sin zoom. Valores más bajos acercan el zénit.",
            key=f"{prefix}_zoom_rmax",
        )

        modo_etiquetas = st.selectbox(
            "Etiquetas",
            options=["Inteligentes", "Todas", "Top (más brillantes)"],
            index=["Inteligentes", "Todas", "Top (más brillantes)"].index(draft["modo_etiquetas"])
            if draft["modo_etiquetas"] in ["Inteligentes", "Todas", "Top (más brillantes)"] else 0,
            help="Inteligentes evita amontonamiento cuando hay conjunciones.",
            key=f"{prefix}_modo_etiquetas",
        )

        max_etiquetas = int(draft["max_etiquetas"])
        if modo_etiquetas == "Top (más brillantes)":
            max_etiquetas = st.slider("Cantidad de etiquetas", 1, 7, int(draft["max_etiquetas"]), key=f"{prefix}_max_etiquetas")

        with st.expander("Ajustes finos de etiquetas", expanded=False):
            separacion_etiquetas_px = st.slider(
                "Separación mínima entre etiquetas (px)",
                0, 30, int(draft["separacion_etiquetas_px"]),
                key=f"{prefix}_sep_px",
            )
            cluster_px = st.slider(
                "Distancia para considerar objetos “juntos” (px)",
                5, 60, int(draft["cluster_px"]),
                key=f"{prefix}_cluster_px",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        draft["show_horizon"] = bool(show_horizon)
        draft["selected_objects"] = list(selected_objects)
        draft["auto_zoom"] = bool(auto_zoom)
        draft["zoom_rmax"] = int(zoom_rmax)
        draft["modo_etiquetas"] = str(modo_etiquetas)
        draft["max_etiquetas"] = int(max_etiquetas)
        draft["separacion_etiquetas_px"] = int(separacion_etiquetas_px)
        draft["cluster_px"] = int(cluster_px)

    return draft


def apply_draft(draft: dict) -> None:
    st.session_state.lat = float(draft["lat"])
    st.session_state.lon = float(draft["lon"])
    st.session_state.alt = float(draft["alt"])

    new_label = (draft.get("place_label") or "").strip()
    old_label = (st.session_state.place_label or "").strip()

    if new_label and new_label != old_label:
        st.session_state.place_label_user_edited = True
        st.session_state.place_label_source = "user"

    st.session_state.place_label = new_label

    st.session_state.time_mode = draft["time_mode"]
    st.session_state.selected_dt_ar = draft["selected_dt_ar"]

    st.session_state.show_horizon = bool(draft["show_horizon"])
    st.session_state.selected_objects = list(draft["selected_objects"])
    st.session_state.auto_zoom = bool(draft["auto_zoom"])
    st.session_state.zoom_rmax = int(draft["zoom_rmax"])
    st.session_state.modo_etiquetas = str(draft["modo_etiquetas"])
    st.session_state.max_etiquetas = int(draft["max_etiquetas"])
    st.session_state.separacion_etiquetas_px = int(draft["separacion_etiquetas_px"])
    st.session_state.cluster_px = int(draft["cluster_px"])


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="SkyMap— Mapa del cielo",
    layout="wide",
    initial_sidebar_state="collapsed",
)
set_astro_theme()
st.markdown(
    """
    <div class="app-footer">
      hecho con <a href="https://www.astropy.org" target="_blank">Astropy</a> y paciencia
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session defaults
# -------------------------
st.session_state.setdefault("place_label", "")
st.session_state.setdefault("place_label_user_edited", False)
st.session_state.setdefault("place_label_source", "auto")
st.session_state.setdefault("lat", -34.51)
st.session_state.setdefault("lon", -58.48)
st.session_state.setdefault("alt", 22.0)
st.session_state.setdefault("time_mode", "Ahora")
st.session_state.setdefault("selected_dt_ar", datetime.now(TZ_AR))
st.session_state.setdefault("show_horizon", True)
st.session_state.setdefault("selected_objects", list(MAGS.keys()))

st.session_state.setdefault("auto_zoom", True)
st.session_state.setdefault("zoom_rmax", 90)
st.session_state.setdefault("modo_etiquetas", "Inteligentes")
st.session_state.setdefault("max_etiquetas", 6)
st.session_state.setdefault("separacion_etiquetas_px", 10)
st.session_state.setdefault("cluster_px", 22)

# -------------------------
# Mobile detection
# -------------------------
is_mobile = detectar_mobile()

# -------------------------
# Controles: desktop sidebar / mobile expander
# -------------------------
auto_refresh = False
aplicar = False
draft = None

if is_mobile:
    with st.expander("⚙️ Controles", expanded=False):
        draft = render_controles(prefix="mobile")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        auto_refresh = st.toggle("Actualizar automático (cada 10 s)", value=False, key="auto_mobile")
        if auto_refresh and not AUTOREFRESH_OK:
            st.info("Instalá `streamlit-autorefresh` para esta función.")
        aplicar = st.button("Aplicar cambios", type="primary", width="stretch", key="aplicar_mobile")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown("### 🌌 SkyMap")
        st.markdown(
            '<div class="muted">Mapa interactivo del cielo: Sol, Luna y planetas desde tu ubicación.</div>',
            unsafe_allow_html=True
        )

        draft = render_controles(prefix="sidebar")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        auto_refresh = st.toggle("Actualizar automático (cada 10 s)", value=False, key="auto_sidebar")
        if auto_refresh and not AUTOREFRESH_OK:
            st.info("Instalá `streamlit-autorefresh` para esta función.")
        aplicar = st.button("Aplicar cambios", type="primary", width="stretch", key="aplicar_sidebar")
        st.markdown("</div>", unsafe_allow_html=True)

if aplicar and draft is not None:
    apply_draft(draft)
    st.rerun()

if auto_refresh and AUTOREFRESH_OK:
    st_autorefresh(interval=10_000, key="auto_refresh")

# -------------------------
# Main
# -------------------------

lat = float(st.session_state.lat)
lon = float(st.session_state.lon)
alt = float(st.session_state.alt)
place = (st.session_state.place_label or "").strip()

if st.session_state.time_mode == "Ahora":
    dt_ar = datetime.now(TZ_AR)
else:
    dt_in = st.session_state.selected_dt_ar
    dt_ar = dt_in if dt_in.tzinfo else dt_in.replace(tzinfo=TZ_AR)

t_utc = dt_ar.astimezone(ZoneInfo("UTC"))
t_astropy = Time(t_utc)

if place:
    header_title = f"SkyMap - {place}"
else:
    # En celu 2 decimales.
    if is_mobile:
        header_title = f"SkyMap - {lat:.2f}°S, {lon:.2f}°O"
    else:
        header_title = f"SkyMap - {lat:.4f}°S, {lon:.4f}°O"

st.markdown(f"# 🌌 {header_title}")
st.markdown(
    f'<div class="muted">Hora Argentina: {fmt_ar(dt_ar)} • Modo: <b>{st.session_state.time_mode}</b></div>',
    unsafe_allow_html=True
)

main_tab1, main_tab2 = st.tabs(["🗺️ Mapa", "📊 Datos"])

with main_tab1:
    vista_3d = st.toggle("Vista 3D (experimental)", value=False, key="vista_3d")
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    altaz, _ = compute_altaz(t_astropy, location, nombres=st.session_state.selected_objects)

    plot_title = f"Cielo visible — {fmt_ar(dt_ar)}"
    modo_etq = _ui_modo_etiquetas_to_core(st.session_state.modo_etiquetas)

    if vista_3d:
        html = build_sky_3d_html(
            altaz_dict=altaz,
            title=f"Cielo 3D (experimental) — {fmt_ar(dt_ar)}",
            show_horizon=st.session_state.show_horizon,
            show_stars=True,
            stars_n=700 if is_mobile else 1100,
        )

        # Alto fijo razonable: en desktop más grande, en celu más compacto
        height_px = 520 if is_mobile else 650

        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        components.html(html, height=height_px, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        fig = make_figure(
            altaz_dict=altaz,
            title=plot_title,
            theme="dark",
            mostrar_horizonte=st.session_state.show_horizon,
            rmax=float(st.session_state.zoom_rmax),
            auto_zoom=bool(st.session_state.auto_zoom),
            zoom_margin_deg=6.0,
            modo_etiquetas=modo_etq,
            max_etiquetas=int(st.session_state.max_etiquetas),
            min_sep_px=float(st.session_state.separacion_etiquetas_px),
            cluster_px=float(st.session_state.cluster_px),
        )

        fig_inches = 7.2 if is_mobile else 8.0
        fig.set_size_inches(fig_inches, fig_inches)

        preview = io.BytesIO()
        fig.savefig(preview, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        preview.seek(0)

        download = io.BytesIO()
        fig.savefig(download, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        download.seek(0)

        plt.close(fig)

        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align: center; margin: 0 0 0.35rem 0;'>{plot_title}</h3>",
            unsafe_allow_html=True
        )
        st.image(preview.getvalue(), width="stretch")

        fname = f"astroview_{dt_ar.strftime('%Y%m%d_%H%M%S')}.png"
        fname = f"astroview_{dt_ar.strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button(
            "⬇️ Descargar PNG (Alta Res)",
            data=download,
            file_name=fname,
            mime="image/png",
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)

with main_tab2:
    if is_mobile:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estado actual")
        st.metric("Hora (Argentina)", fmt_ar(dt_ar))
        st.markdown(f"**Lat/Lon/Alt** \n`{lat:.6f}, {lon:.6f}, {alt:.1f} m`")
        if place:
            st.markdown(f"**Lugar** \n`{place}`")
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.dataframe(rows, width="stretch", hide_index=True)
        st.markdown(
            '<div class="tiny">Azimut: 0° Norte, 90° Este, 180° Sur, 270° Oeste. Magnitud: menor = más brillante.</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
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
            st.dataframe(rows, width="stretch", hide_index=True)
            st.markdown(
                '<div class="tiny">Azimut: 0° Norte, 90° Este, 180° Sur, 270° Oeste. Magnitud: menor = más brillante.</div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
