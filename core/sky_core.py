from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import astropy.units as u
from astropy.coordinates import AltAz, get_sun, get_body, solar_system_ephemeris
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None

MAGS = {
    "Sol": -26.7,
    "Luna": -12.0,
    "Mercurio": 0.0,
    "Venus": -4.3,
    "Marte": 0.5,
    "Júpiter": -2.7,
    "Saturno": 0.8,
}

PLANET_COLORS = {
    "Sol":      "#FFD54A",
    "Luna":     "#D9D9D9",
    "Mercurio": "#B0B0B0",
    "Venus":    "#E8D8A8",
    "Marte":    "#D14B3A",
    "Júpiter":  "#D9B38C",
    "Saturno":  "#E6D27A",
}

THEMES = {
    "dark": {
        "fig_bg": "#0F1115",
        "ax_bg":  "#0F1115",
        "text":   "#F2F4F8",
        "tick":   "#D5DAE3",
        "grid":   "#7A8396",   # luego usamos alpha bajo
        "edge":   "#101010",
        "label":  "#8BE9FD",   # cyan más brillante
    },
    "light": {
        "fig_bg": "#FFFFFF",
        "ax_bg":  "#FFFFFF",
        "text":   "#101418",
        "tick":   "#1E2430",
        "grid":   "#9AA4B2",
        "edge":   "#1A1A1A",
        "label":  "#005B6A",
    },
}

def compute_altaz(obstime, location):
    with solar_system_ephemeris.set("builtin"):
        objs = {
            "Sol": get_sun(obstime),
            "Luna": get_body("moon", obstime),
            "Mercurio": get_body("mercury", obstime),
            "Venus": get_body("venus", obstime),
            "Marte": get_body("mars", obstime),
            "Júpiter": get_body("jupiter", obstime),
            "Saturno": get_body("saturn", obstime),
        }
    frame = AltAz(obstime=obstime, location=location)
    return {name: coord.transform_to(frame) for name, coord in objs.items()}

def make_figure(altaz_dict, title: str, theme: str = "dark"):
    t = THEMES.get(theme, THEMES["dark"])

    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor(t["fig_bg"])

    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(t["ax_bg"])
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    # Un poco más visible el marco
    ax.spines["polar"].set_color(t["grid"])
    ax.spines["polar"].set_alpha(0.35)

    # Para el halo del texto (mejora legibilidad)
    text_fx = [pe.withStroke(linewidth=3, foreground=t["ax_bg"])]

    for name, c in altaz_dict.items():
        alt = float(c.alt.deg)
        if alt <= 0:
            continue

        theta = float(c.az.rad)
        r = 90.0 - alt

        mag = MAGS.get(name, 1.0)
        size = float(np.clip(280 - mag * 35, 28, 520))

        face = PLANET_COLORS.get(name, "#FFFFFF")
        edge = t["edge"]

        ax.scatter(theta, r, s=size, c=face, edgecolor=edge, linewidths=1.1, alpha=0.95, zorder=3)

        # Offset del label: lo empujamos radialmente hacia afuera y un toque angular según cuadrante
        # (para que no tape el punto ni quede cortado)
        dx = 8 if np.cos(theta) >= 0 else -8   # derecha/izquierda
        dy = 6                                  # hacia “afuera” (en pantalla)
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
            # bbox opcional (si querés aún más legibilidad):
            # bbox=dict(boxstyle="round,pad=0.15", facecolor=t["ax_bg"], edgecolor="none", alpha=0.35),
        )

    # Escala radial
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["Zenit", "60°", "30°", "Horizonte"], color=t["tick"])
    ax.tick_params(colors=t["tick"])

    # Grid con mejor contraste (pero suave)
    ax.grid(True, color=t["grid"], linestyle="--", alpha=0.28)

    ax.set_title(title, fontsize=13, color=t["text"], pad=14)
    fig.tight_layout()
    return fig
