from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, get_sun, get_body, solar_system_ephemeris
from astropy.utils import iers

# Evita errores offline (menos precisión; suficiente para un mapa “qué hay ahora”)
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

def compute_altaz(obstime, location):
    """
    obstime: astropy.time.Time
    location: astropy.coordinates.EarthLocation
    returns: dict[str, SkyCoord in AltAz frame]
    """

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

def make_figure(altaz_dict, title: str):
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    for name, c in altaz_dict.items():
        alt = float(c.alt.deg)
        if alt <= 0:
            continue

        theta = float(c.az.rad)
        r = 90.0 - alt

        size = float(np.clip(300 - MAGS.get(name, 1.0) * 40, 30, 650))
        ax.scatter(theta, r, s=size, edgecolor="black", alpha=0.9, zorder=3)
        ax.text(theta, r + 3, name, fontsize=10, color="cyan", ha="center", va="center", zorder=4)

    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["Zenit", "60°", "30°", "Horizonte"])
    ax.grid(True, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    return fig