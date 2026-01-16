# core/sky_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers

# ---- Astropy / IERS: evitar descargas en runtime (ideal para apps offline)
iers.conf.auto_download = False
iers.conf.auto_max_age = None

# -------------------------
# Catálogo básico (simple y útil)
# -------------------------
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

# Tema oscuro con toques astronómicos (más inmersivo)
DARK = {
    "fig_bg": "#020308",  # Negro profundo como el espacio
    "ax_bg": "#020308",
    "text": "#E0E7FF",  # Blanco azulado para texto
    "tick": "#A5B4FC",
    "grid": "#3730A3",  # Morado oscuro para rejilla
    "edge": "#101010",
    "label": "#A5B4FC",  # Azul claro para etiquetas
    "horizon": "#6366F1",  # Horizonte con tono índigo
    "star_glow": "#E0E7FF",  # Para efectos de brillo
}


@dataclass(frozen=True)
class SkyObject:
    nombre: str
    alt_deg: float
    az_deg: float
    mag: float
    visible: bool  # alt > 0


def _objects_gcrs(obstime: Time) -> Dict[str, object]:
    """Devuelve coords (GCRS/ICRS astropy) de un set simple."""
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
    """
    Calcula Alt/Az de objetos del sistema solar para una ubicación y momento.
    Retorna:
      - dict nombre -> coord AltAz
      - lista de SkyObject (tabla) ordenada por altitud descendente
    """
    objs = _objects_gcrs(obstime)
    if nombres is not None:
        nombres_set = set(nombres)
        objs = {k: v for k, v in objs.items() if k in nombres_set}
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
    """Escala visual simple por magnitud (más brillante => más grande)."""
    size = 260 - mag * 32.0
    return float(np.clip(size, 26, 520))


def _alpha_from_alt(alt_deg: float) -> float:
    """Más alto => más visible. Cerca del horizonte => un poco más tenue."""
    if alt_deg <= 0:
        return 0.0
    return float(np.clip(0.55 + 0.45 * (alt_deg / 90.0), 0.55, 1.0))


def make_figure(
        altaz_dict: Dict[str, AltAz],
        title: str,
        mostrar_horizonte: bool = True,
) -> plt.Figure:
    """
    Renderiza un mapa polar:
      - Norte arriba
      - Azimut crece hacia la derecha (sentido horario)
      - Radio: 0 = Zénit, 90 = Horizonte
    Agregados: fondo estrellado sutil, glow en objetos para tema astro.
    """
    t = DARK
    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor(t["fig_bg"])
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(t["ax_bg"])
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.spines["polar"].set_color(t["grid"])
    ax.spines["polar"].set_alpha(0.35)

    # Fondo estrellado sutil (simulando estrellas aleatorias)
    num_stars = 150
    star_theta = np.random.uniform(0, 2 * np.pi, num_stars)
    star_r = np.random.uniform(0, 90, num_stars)
    star_sizes = np.random.uniform(1, 4, num_stars)
    ax.scatter(star_theta, star_r, s=star_sizes, c=t["star_glow"], alpha=0.3, zorder=0)

    text_fx = [pe.withStroke(linewidth=3, foreground=t["ax_bg"])]
    if mostrar_horizonte:
        th = np.linspace(0, 2 * np.pi, 361)
        ax.plot(th, np.full_like(th, 90.0), linestyle="-", linewidth=1.1, alpha=0.55, color=t["horizon"], zorder=1)

    # Puntos cardinales con fuente más pro
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

    for name, c in altaz_dict.items():
        alt = float(c.alt.deg)
        if alt <= 0:
            continue
        theta = float(c.az.rad)
        r = 90.0 - alt
        mag = float(MAGS.get(name, 1.0))
        size = _size_from_mag(mag)
        alpha = _alpha_from_alt(alt)
        face = PLANET_COLORS.get(name, "#FFFFFF")
        # Scatter principal
        ax.scatter(
            theta, r,
            s=size,
            c=face,
            edgecolor=t["edge"],
            linewidths=1.05,
            alpha=alpha,
            zorder=3,
        )
        # Glow effect para brillo astronómico
        ax.scatter(
            theta, r,
            s=size * 1.8,
            c=face,
            alpha=alpha * 0.25,
            zorder=2,
        )
        dx = 8 if np.cos(theta) >= 0 else -8
        dy = 6
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            name,
            xy=(theta, r),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=10.5,
            color=t["label"],
            path_effects=text_fx,
            zorder=4,
        )

    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["Zénit", "60°", "30°", "Horizonte"], color=t["tick"])
    ax.tick_params(colors=t["tick"])
    ax.grid(True, color=t["grid"], linestyle="--", alpha=0.28)
    ax.set_title(title, fontsize=13, color=t["text"], pad=14)
    fig.tight_layout()
    return fig