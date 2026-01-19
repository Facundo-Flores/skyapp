from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers

# ---- Astropy / IERS: evitar descargas en runtime
iers.conf.auto_download = False
iers.conf.auto_max_age = None

MAGS: Dict[str, float] = {
    "Sol": -26.7,
    "Luna": -12.0,
    "Mercurio": 0.0,
    "Venus": -4.3,
    "Marte": 0.5,
    "Júpiter": -2.7,
    "Saturno": 0.8,
    "Urano": 5.7,
    "Neptuno": 7.8,
}

PLANET_COLORS: Dict[str, str] = {
    "Sol": "#FFD54A",
    "Luna": "#D9D9D9",
    "Mercurio": "#B0B0B0",
    "Venus": "#E8D8A8",
    "Marte": "#D14B3A",
    "Júpiter": "#D9B38C",
    "Saturno": "#E6D27A",
    "Urano": "#7FCFD4",
    "Neptuno": "#4B6EDC",
}

THEMES = {
    "dark": {
        "fig_bg": "#0F1115",
        "ax_bg":  "#0F1115",
        "text":   "#F2F4F8",
        "tick":   "#D5DAE3",
        "grid":   "#7A8396",
        "edge":   "#101010",
        "label":  "#8BE9FD",
        "horizon": "#AAB2C3",
    },
    "light": {
        "fig_bg": "#FFFFFF",
        "ax_bg":  "#FFFFFF",
        "text":   "#101418",
        "tick":   "#1E2430",
        "grid":   "#9AA4B2",
        "edge":   "#1A1A1A",
        "label":  "#005B6A",
        "horizon": "#2B3340",
    },
}


@dataclass(frozen=True)
class SkyObject:
    nombre: str
    alt_deg: float
    az_deg: float
    mag: float
    visible: bool  # alt > 0


def _objects_gcrs(obstime: Time) -> Dict[str, object]:
    with solar_system_ephemeris.set("builtin"):
        return {
            "Sol": get_sun(obstime),
            "Luna": get_body("moon", obstime),
            "Mercurio": get_body("mercury", obstime),
            "Venus": get_body("venus", obstime),
            "Marte": get_body("mars", obstime),
            "Júpiter": get_body("jupiter", obstime),
            "Saturno": get_body("saturn", obstime),
            "Urano": get_body("uranus", obstime),
            "Neptuno": get_body("neptune", obstime),
        }


def compute_altaz(
    obstime: Time,
    location: EarthLocation,
    nombres: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, AltAz], List[SkyObject]]:
    objs = _objects_gcrs(obstime)
    if nombres is not None:
        wanted = set(nombres)
        objs = {k: v for k, v in objs.items() if k in wanted}

    frame = AltAz(obstime=obstime, location=location)
    altaz = {name: coord.transform_to(frame) for name, coord in objs.items()}

    table: List[SkyObject] = []
    for name, c in altaz.items():
        alt = float(c.alt.deg)
        az = float(c.az.deg)
        mag = float(MAGS.get(name, 1.0))
        table.append(SkyObject(nombre=name, alt_deg=alt, az_deg=az, mag=mag, visible=alt > 0))

    table.sort(key=lambda o: o.alt_deg, reverse=True)
    return altaz, table


def _size_from_mag(mag: float) -> float:
    size = 260 - mag * 32.0
    return float(np.clip(size, 26, 520))


def _alpha_from_alt(alt_deg: float) -> float:
    if alt_deg <= 0:
        return 0.0
    return float(np.clip(0.55 + 0.45 * (alt_deg / 90.0), 0.55, 1.0))


# -------------------------
# Starfield (fondo)
# -------------------------
def _stars_field(
    rmax: float,
    n: int,
    seed: int,
    size_min: float,
    size_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera un campo de 'estrellas' en coordenadas polares (theta, r) dentro del rango visible.
    r: 0 (zénit) -> rmax (borde visible)
    """
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0.0, 2.0 * np.pi, n)

    # r proporcional al área (para que no se acumule cerca del centro)
    u = rng.uniform(0.0, 1.0, n)
    r = np.sqrt(u) * rmax

    # tamaños con sesgo a chiquitas
    s = (rng.uniform(0.0, 1.0, n) ** 2) * (size_max - size_min) + size_min

    return theta, r, s


def _draw_stars(
    ax,
    t: dict,
    rmax: float,
    enabled: bool,
    density: float,
    seed: int,
) -> None:
    """
    Dibuja un fondo de estrellas detrás de todo.
    densidad: 0..1 aprox
    """
    if not enabled:
        return

    # Ajuste: con zoom fuerte (rmax bajo) bajamos el conteo para que no se vea "ruidoso".
    base = 900
    zoom_factor = float(np.clip(rmax / 90.0, 0.35, 1.0))
    n = int(base * float(density) * zoom_factor)
    n = int(np.clip(n, 150, 1400))

    theta, r, s = _stars_field(
        rmax=float(rmax),
        n=n,
        seed=int(seed),
        size_min=2.0,
        size_max=18.0,
    )

    # Capa tenue
    ax.scatter(
        theta,
        r,
        s=s,
        c=t["tick"],
        alpha=0.10,
        linewidths=0.0,
        zorder=0,
    )

    # Capa principal (solo algunas un poco más visibles)
    idx = (s > 8.0)
    ax.scatter(
        theta[idx],
        r[idx],
        s=s[idx] * 0.9,
        c=t["text"],
        alpha=0.12,
        linewidths=0.0,
        zorder=0,
    )


# -------------------------
# Labels
# -------------------------
def _overlap_area(a: Bbox, b: Bbox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _inflate_bbox(bb: Bbox, pad_px: float) -> Bbox:
    return Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)


def _select_labels(
    ax,
    points: List[dict],
    label_mode: str,
    max_labels: int,
    cluster_px: float,
) -> List[dict]:
    """
    points: lista de dict con keys: name, theta, r, mag, alt
    label_mode: "todas" | "inteligentes" | "top"
    """
    if label_mode == "todas":
        return points

    # Orden por brillo (mag menor = más brillante)
    pts = sorted(points, key=lambda p: p["mag"])

    if label_mode == "top":
        return pts[:max_labels]

    # "inteligentes": etiqueta objetos "separados" en pantalla; si están muy cerca, gana el más brillante
    labeled: List[dict] = []
    taken_xy: List[Tuple[float, float]] = []

    for p in pts:
        x, y = ax.transData.transform((p["theta"], p["r"]))
        too_close = False
        for (tx, ty) in taken_xy:
            if (x - tx) ** 2 + (y - ty) ** 2 <= cluster_px ** 2:
                too_close = True
                break
        if too_close:
            continue
        labeled.append(p)
        taken_xy.append((x, y))
        if len(labeled) >= max_labels:
            break

    return labeled


def _place_labels_non_overlapping(
    fig: plt.Figure,
    ax,
    label_points: List[dict],
    t: dict,
    min_label_sep_px: float,
):
    """
    Ubica etiquetas evitando superposición entre ellas.
    """
    text_fx = [pe.withStroke(linewidth=3, foreground=t["ax_bg"])]

    candidates = [
        (12, 8), (12, -8), (-12, 8), (-12, -8),
        (0, 12), (0, -12),
        (18, 0), (-18, 0),
        (18, 10), (18, -10), (-18, 10), (-18, -10),
    ]

    placed_bboxes: List[Bbox] = []

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for p in label_points:
        name = p["name"]
        theta = p["theta"]
        r = p["r"]

        best_text = None
        best_bbox = None
        best_score = None

        for (dx, dy) in candidates:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")

            txt = ax.annotate(
                name,
                xy=(theta, r),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=10.5,
                color=t["label"],
                path_effects=text_fx,
                zorder=6,
            )

            fig.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer)
            bb2 = _inflate_bbox(bb, min_label_sep_px)

            score = 0.0
            for prev in placed_bboxes:
                score += _overlap_area(bb2, prev)

            if best_score is None or score < best_score:
                if best_text is not None:
                    best_text.remove()
                best_text = txt
                best_bbox = bb2
                best_score = score
            else:
                txt.remove()

            if best_score == 0.0:
                break

        if best_text is not None and best_bbox is not None:
            placed_bboxes.append(best_bbox)


def make_figure(
    altaz_dict: Dict[str, AltAz],
    title: str,
    theme: str = "dark",
    mostrar_horizonte: bool = True,
    # --- Zoom
    rmax: float = 90.0,            # zoom manual (90 = sin zoom)
    auto_zoom: bool = False,       # auto-encuadre
    zoom_margin_deg: float = 6.0,  # margen extra (en grados de radio)
    # --- Etiquetas
    modo_etiquetas: str = "inteligentes",  # "todas" | "inteligentes" | "top"
    max_etiquetas: int = 6,
    min_sep_px: float = 10.0,
    cluster_px: float = 20.0,
    # --- Estrellas (fondo)
    mostrar_estrellas: bool = True,
    densidad_estrellas: float = 0.75,  # 0..1 aprox
    seed_estrellas: int = 42,
) -> plt.Figure:
    """
    Mapa polar:
      - Norte arriba
      - Azimut crece hacia la derecha (sentido horario)
      - Radio: 0 = Zénit, 90 = Horizonte
    """
    t = THEMES.get(theme, THEMES["dark"])

    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor(t["fig_bg"])

    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(t["ax_bg"])
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    ax.spines["polar"].set_color(t["grid"])
    ax.spines["polar"].set_alpha(0.35)

    # Armamos lista de puntos visibles
    points: List[dict] = []
    for name, c in altaz_dict.items():
        alt = float(c.alt.deg)
        if alt <= 0:
            continue
        theta = float(c.az.rad)
        r = 90.0 - alt
        mag = float(MAGS.get(name, 1.0))
        points.append({"name": name, "theta": theta, "r": r, "mag": mag, "alt": alt})

    # Auto-zoom: encuadra hasta el objeto visible más bajo
    if auto_zoom and points:
        alt_min = min(p["alt"] for p in points)
        r_needed = 90.0 - alt_min
        rmax = min(90.0, float(r_needed + zoom_margin_deg))
        rmax = max(18.0, rmax)

    # Clamp manual
    rmax = float(np.clip(rmax, 18.0, 90.0))

    # Fondo: estrellas (antes de todo)
    _draw_stars(
        ax=ax,
        t=t,
        rmax=float(rmax),
        enabled=bool(mostrar_estrellas),
        density=float(densidad_estrellas),
        seed=int(seed_estrellas),
    )

    # Horizonte
    if mostrar_horizonte:
        th = np.linspace(0, 2 * np.pi, 361)
        ax.plot(
            th,
            np.full_like(th, 90.0),
            linestyle="-",
            linewidth=1.1,
            alpha=0.55,
            color=t["horizon"],
            zorder=1,
        )

    # Cardinales
    text_fx = [pe.withStroke(linewidth=3, foreground=t["ax_bg"])]
    for lab, deg in [("N", 0), ("E", 90), ("S", 180), ("O", 270)]:
        ax.text(
            np.deg2rad(deg),
            92.0,
            lab,
            ha="center",
            va="center",
            fontsize=10.5,
            color=t["tick"],
            path_effects=text_fx,
            zorder=5,
        )

    # Plot de puntos (planetas)
    for p in points:
        name = p["name"]
        theta = p["theta"]
        r = p["r"]
        mag = p["mag"]
        alt = p["alt"]

        size = _size_from_mag(mag)
        alpha = _alpha_from_alt(alt)
        face = PLANET_COLORS.get(name, "#FFFFFF")

        ax.scatter(
            theta,
            r,
            s=size,
            c=face,
            edgecolor=t["edge"],
            linewidths=1.05,
            alpha=alpha,
            zorder=4,
        )

    # Escala radial / zoom
    ax.set_rlim(0, rmax)

    # Ticks más razonables según rmax
    tick_rs = np.linspace(0, rmax, 4)
    tick_rs = [float(x) for x in tick_rs]
    tick_labels = []
    for rr in tick_rs:
        alt_lbl = 90.0 - rr
        if rr == 0:
            tick_labels.append("Zénit")
        elif rr >= rmax - 1e-6:
            tick_labels.append("Horizonte" if rmax >= 89.5 else f"{alt_lbl:.0f}°")
        else:
            tick_labels.append(f"{alt_lbl:.0f}°")

    ax.set_rticks(tick_rs)
    ax.set_yticklabels(tick_labels, color=t["tick"])
    ax.tick_params(colors=t["tick"])
    ax.grid(True, color=t["grid"], linestyle="--", alpha=0.28)

    # Etiquetas: selección + colocación sin superposición
    label_mode_norm = (modo_etiquetas or "").strip().lower()
    if label_mode_norm not in ("todas", "inteligentes", "top"):
        label_mode_norm = "inteligentes"

    label_points = _select_labels(
        ax=ax,
        points=points,
        label_mode=label_mode_norm,
        max_labels=max_etiquetas,
        cluster_px=float(cluster_px),
    )

    _place_labels_non_overlapping(
        fig=fig,
        ax=ax,
        label_points=label_points,
        t=t,
        min_label_sep_px=float(min_sep_px),
    )

    fig.tight_layout(pad=0.5)
    return fig
