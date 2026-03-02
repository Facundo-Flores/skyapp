# app_streamlit.py

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import sys
import io
import os
import requests
import base64

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

from core.sky_core import compute_altaz, make_figure, MAGS
from core.sky_3d import build_sky_3d_html

# Geolocalización y autorefresh (opcional)
try:
    from streamlit_geolocation import streamlit_geolocation
    GEO_OK = True
except Exception:
    GEO_OK = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except Exception:
    AUTOREFRESH_OK = False

TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")

# ────────────────────────────────────────────────
# Inicialización temprana de session_state (¡muy importante!)
# ────────────────────────────────────────────────

defaults = {
    "lat": -34.51,
    "lon": -58.48,
    "alt": 22.0,
    "place_label": "",
    "place_label_user_edited": False,
    "place_label_source": "auto",
    "time_mode": "Ahora",
    "selected_dt_ar": datetime.now(TZ_AR),
    "show_horizon": True,
    "selected_objects": list(MAGS.keys()),
    "auto_zoom": True,
    "zoom_rmax": 90,
    "modo_etiquetas": "Inteligentes",
    "max_etiquetas": 6,
    "separacion_etiquetas_px": 10,
    "cluster_px": 22,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ────────────────────────────────────────────────
# Helpers (mantengo los tuyos)
# ────────────────────────────────────────────────

def get_local_asset_base64(file_name: str) -> str:
    path = Path(__file__).parent.parent / "assets" / file_name
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

@st.cache_resource
def load_all_textures_parallel():
    assets_to_load = {
        "Sol": "sun.png",
        "Luna": "moon.png",
        "Mercurio": "mercury.png",
        "Venus": "venus.png",
        "Marte": "mars.png",
        "Júpiter": "jupiter.png",
        "Saturno": "saturn.png",
        "Urano": "uranus.png",
        "Neptuno": "neptune.png",
        "MilkyWay": "milkyway.png"
    }

    def load_task(name, file_name):
        return name, get_local_asset_base64(file_name)

    tex_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_task, name, file) for name, file in assets_to_load.items()]
        for future in concurrent.futures.as_completed(futures):
            try:
                name, b64 = future.result()
                tex_map[name] = b64
            except Exception as e:
                st.error(f"Error cargando {name}: {e}")
                tex_map[name] = ""
    return tex_map

def fmt_ar(dt: datetime) -> str:
    return dt.astimezone(TZ_AR).strftime("%d-%m-%Y %H:%M:%S")

def detectar_mobile() -> bool:
    try:
        from browser_detection import browser_detection_engine
        info = browser_detection_engine() or {}
        if "is_mobile" in info:
            return bool(info["is_mobile"])
        if info.get("device_type"):
            return str(info["device_type"]).lower() in ("mobile", "tablet")
        if info.get("is_tablet"):
            return bool(info.get("is_tablet"))
    except Exception:
        pass

    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            ua = (st.context.headers.get("User-Agent") or st.context.headers.get("user-agent") or "").lower()
            if any(k in ua for k in ["android", "iphone", "ipad", "ipod", "mobile", "tablet"]):
                return True
    except Exception:
        pass
    return False

def get_bdc_key() -> str | None:
    return st.secrets.get("BIGDATACLOUD_API_KEY", None) or os.environ.get("BIGDATACLOUD_API_KEY")

@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode_bigdatacloud(lat_q: float, lon_q: float, locality_language: str = "es") -> dict:
    key = get_bdc_key()
    if not key:
        raise RuntimeError("Falta BIGDATACLOUD_API_KEY")
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

# ────────────────────────────────────────────────
# Theme (mantengo el tuyo completo)
# ────────────────────────────────────────────────

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

# Page config + theme
st.set_page_config(
    page_title="SkyMap — Mapa del cielo",
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

# ────────────────────────────────────────────────
# Controles mejorados
# ────────────────────────────────────────────────

def render_controls(is_mobile: bool):
    prefix = "mobile" if is_mobile else "sidebar"

    if is_mobile:
        container = st.expander("⚙️ Controles", expanded=False)
    else:
        container = st.sidebar
        st.sidebar.markdown("### 🌌 SkyMap")
        st.sidebar.markdown(
            '<div class="muted">Mapa interactivo del cielo: Sol, Luna y planetas desde tu ubicación.</div>',
            unsafe_allow_html=True
        )

    with container:
        # Acciones arriba
        col_act1, col_act2 = st.columns([3, 2])
        with col_act1:
            aplicar = st.button("Actualizar", type="primary", width="stretch", key=f"{prefix}_aplicar")
        with col_act2:
            auto_refresh = st.toggle("Auto (cada 10s)", value=False, key=f"{prefix}_autorefresh")
            if auto_refresh and not AUTOREFRESH_OK:
                st.caption("Instalá streamlit-autorefresh")

        st.markdown("---")

        # Ubicación y Hora
        st.subheader("Ubicación y Hora", divider="gray")

        modo_ubic = st.radio(
            "Modo ubicación",
            ["Navegador", "Manual"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}_modo_ubic"
        )

        lat = float(st.session_state.lat)
        lon = float(st.session_state.lon)

        if modo_ubic == "Navegador":
            use_geo = st.toggle("Usar mi ubicación", value=True, disabled=not GEO_OK, key=f"{prefix}_usar_geo")

            if use_geo and GEO_OK:
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") and loc.get("longitude"):
                    lat = float(loc["latitude"])
                    lon = float(loc["longitude"])

                    if not st.session_state.get("place_label_user_edited", False):
                        try:
                            lat_q = round(lat, 3)
                            lon_q = round(lon, 3)
                            payload = reverse_geocode_bigdatacloud(lat_q, lon_q)
                            auto_label = build_place_label(payload)
                            if auto_label:
                                st.session_state.place_label = auto_label
                                st.session_state.place_label_source = "auto"
                                st.rerun()
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

        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitud (°)", value=lat, format="%.6f", key=f"{prefix}_lat")
        with col_lon:
            lon = st.number_input("Longitud (°)", value=lon, format="%.6f", key=f"{prefix}_lon")

        alt = st.number_input("Altitud (m)", value=float(st.session_state.alt), format="%.1f", key=f"{prefix}_alt")

        place_label = st.text_input(
            "Etiqueta del lugar",
            value=st.session_state.place_label.strip(),
            key=f"{prefix}_place_label"
        )

        time_mode = st.radio(
            "Modo tiempo",
            ["Ahora", "Personalizado"],
            horizontal=True,
            key=f"{prefix}_time_mode"
        )

        selected_dt_ar = st.session_state.selected_dt_ar
        if time_mode == "Personalizado":
            col_date, col_time = st.columns(2)
            with col_date:
                date_part = st.date_input("Fecha", value=selected_dt_ar.date(), key=f"{prefix}_date")
            with col_time:
                time_part = st.time_input("Hora", value=selected_dt_ar.time(), key=f"{prefix}_time")
            selected_dt_ar = datetime.combine(date_part, time_part).replace(tzinfo=TZ_AR)

        st.markdown("---")

        # Visualización
        with st.expander("Visualización", expanded=False):
            st.session_state.show_horizon = st.toggle("Mostrar horizonte", value=st.session_state.show_horizon, key=f"{prefix}_show_horizon")

            st.session_state.selected_objects = st.multiselect(
                "Objetos a mostrar",
                options=list(MAGS.keys()),
                default=st.session_state.selected_objects,
                key=f"{prefix}_selected_objects"
            )

            st.session_state.auto_zoom = st.toggle("Auto zoom", value=st.session_state.auto_zoom, key=f"{prefix}_auto_zoom")

            if not st.session_state.auto_zoom:
                st.session_state.zoom_rmax = st.slider(
                    "Zoom manual (rmax °)",
                    18, 90,
                    value=int(st.session_state.zoom_rmax),
                    step=1,
                    key=f"{prefix}_zoom_rmax"
                )

            st.session_state.modo_etiquetas = st.selectbox(
                "Estilo de etiquetas",
                ["Inteligentes", "Todas", "Top (más brillantes)"],
                index=["Inteligentes", "Todas", "Top (más brillantes)"].index(st.session_state.modo_etiquetas),
                key=f"{prefix}_modo_etiquetas"
            )

            if st.session_state.modo_etiquetas == "Top (más brillantes)":
                st.session_state.max_etiquetas = st.slider(
                    "Máx. etiquetas",
                    1, 7,
                    int(st.session_state.max_etiquetas),
                    key=f"{prefix}_max_etiquetas"
                )

        # Ajustes avanzados
        with st.expander("Ajustes avanzados de etiquetas", expanded=False):
            st.session_state.separacion_etiquetas_px = st.slider(
                "Separación mín. etiquetas (px)",
                0, 30,
                int(st.session_state.separacion_etiquetas_px),
                key=f"{prefix}_sep_px"
            )
            st.session_state.cluster_px = st.slider(
                "Distancia cluster (px)",
                5, 60,
                int(st.session_state.cluster_px),
                key=f"{prefix}_cluster_px"
            )

    # Guardar cambios en session_state
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.alt = alt
    st.session_state.place_label = place_label.strip()
    st.session_state.time_mode = time_mode
    st.session_state.selected_dt_ar = selected_dt_ar

    return aplicar, auto_refresh


# ────────────────────────────────────────────────
# Ejecutar controles
# ────────────────────────────────────────────────

is_mobile = detectar_mobile()
aplicar, auto_refresh = render_controls(is_mobile)

if aplicar:
    st.rerun()

if auto_refresh and AUTOREFRESH_OK:
    st_autorefresh(interval=10000, key="auto_refresh")

# ────────────────────────────────────────────────
# Contenido principal (el resto sin cambios)
# ────────────────────────────────────────────────

lat = float(st.session_state.lat)
lon = float(st.session_state.lon)
alt = float(st.session_state.alt)
place = st.session_state.place_label.strip()

if st.session_state.time_mode == "Ahora":
    dt_ar = datetime.now(TZ_AR)
else:
    dt_ar = st.session_state.selected_dt_ar

t_utc = dt_ar.astimezone(ZoneInfo("UTC"))
t_astropy = Time(t_utc)

header_title = f"SkyMap - {place}" if place else (
    f"SkyMap - {lat:.2f}°S, {lon:.2f}°O" if is_mobile else f"SkyMap - {lat:.4f}°S, {lon:.4f}°O"
)

st.markdown(f"# 🌌 {header_title}")
st.markdown(
    f'<div class="muted">Hora Argentina: {fmt_ar(dt_ar)} • Modo: <b>{st.session_state.time_mode}</b></div>',
    unsafe_allow_html=True
)

tab_map, tab_data = st.tabs(["🗺️ Mapa", "📊 Datos"])

with tab_map:
    vista_3d = st.toggle("Vista 3D (experimental)", value=False, key="vista_3d")
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    altaz, _ = compute_altaz(t_astropy, location, nombres=st.session_state.selected_objects)

    plot_title = f"Cielo visible — {fmt_ar(dt_ar)}"
    modo_etq = _ui_modo_etiquetas_to_core(st.session_state.modo_etiquetas)

    if vista_3d:
        tex_map = load_all_textures_parallel()
        lst = t_astropy.sidereal_time('mean', longitude=location.lon)
        html = build_sky_3d_html(altaz, tex_map, float(lst.deg))
        height_px = 600 if is_mobile else 850
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
            auto_zoom=st.session_state.auto_zoom,
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
        st.markdown(f"<h3 style='text-align: center; margin: 0 0 0.35rem 0;'>{plot_title}</h3>", unsafe_allow_html=True)
        st.image(preview.getvalue(), width="stretch")

        fname = f"sky_{dt_ar.strftime('%Y%m%d_%H%M%S')}.png"
        fname = f"sky_{dt_ar.strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button("⬇️ Descargar PNG (Alta Res)", data=download, file_name=fname, mime="image/png", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Hora (Argentina)", fmt_ar(dt_ar))
    st.markdown(f"**Ubicación**  \n`{lat:.6f}, {lon:.6f}, {alt:.1f} m`")
    if place:
        st.markdown(f"**Lugar**  \n`{place}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Objetos visibles")
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