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
}

PLANET_COLORS: Dict[str, str] = {
    "Sol": "#FFD54A",
    "Luna": "#D9D9D9",
    "Mercurio": "#B0B0B0",
    "Venus": "#E8D8A8",
    "Marte": "#D14B3A",
    "Júpiter": "#D9B38C",
    "Saturno": "#E6D27A",
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
    Coloca etiquetas evitando superposición entre ellas (heurística por candidatos).
    """
    text_fx = [pe.withStroke(linewidth=3, foreground=t["ax_bg"])]

    # Candidatos de offset (en puntos). Probamos varios.
    candidates = [
        (12, 8), (12, -8), (-12, 8), (-12, -8),
        (0, 12), (0, -12),
        (18, 0), (-18, 0),
        (18, 10), (18, -10), (-18, 10), (-18, -10),
    ]

    placed_bboxes: List[Bbox] = []

    # Necesitamos renderer para medir bboxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for p in label_points:
        name = p["name"]
        theta = p["theta"]
        r = p["r"]

        best_text = None
        best_bbox = None
        best_score = None  # menor = mejor (menos overlap total)

        # Probamos offsets
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

            # Puntaje = suma de áreas de intersección con labels ya puestos
            score = 0.0
            for prev in placed_bboxes:
                score += _overlap_area(bb2, prev)

            # Elegimos el que menos se superponga
            if best_score is None or score < best_score:
                # si ya teníamos uno, lo borramos
                if best_text is not None:
                    best_text.remove()
                best_text = txt
                best_bbox = bb2
                best_score = score
            else:
                txt.remove()

            # Si es perfecto (0 overlap), cortamos
            if best_score == 0.0:
                break

        # Si quedó (aunque tenga algo de overlap), lo aceptamos
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
) -> plt.Figure:
    """
    Mapa polar:
      - Norte arriba
      - Azimut crece hacia la derecha (sentido horario)
      - Radio: 0 = Zénit, 90 = Horizonte

    Zoom:
      - manual: rmax
      - auto_zoom: calcula rmax según el objeto visible más bajo + margen
    Etiquetas:
      - evita superposición y permite declutter.
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

    # Cardianles
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
        alt_min = min(p["alt"] for p in points)  # más cerca del horizonte
        r_needed = 90.0 - alt_min
        rmax = min(90.0, float(r_needed + zoom_margin_deg))
        rmax = max(18.0, rmax)  # evita zoom extremo que rompe ticks

    # Clamp manual
    rmax = float(np.clip(rmax, 18.0, 90.0))

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

    # Plot de puntos
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
    # r=0 (Zénit) ... r=rmax (borde visible)
    # Convertimos a altitud aproximada: alt = 90 - r
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

    #ax.set_title(title, fontsize=13, color=t["text"], pad=14)


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


#    fig.suptitle(
#        title,
#        fontsize=13,
#        color=t["text"],
#        y=0.88,
#        va="bottom"
#    )

    fig.tight_layout(pad=0.5)
    return fig
