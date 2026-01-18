import pytest
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from core.sky_core import compute_altaz, _size_from_mag, _alpha_from_alt


def test_size_from_mag():
    """Verifica que objetos más brillantes sean más grandes."""
    sirio_mag = -1.46
    estrella_tenue_mag = 4.0

    size_bright = _size_from_mag(sirio_mag)
    size_dim = _size_from_mag(estrella_tenue_mag)

    assert size_bright > size_dim
    # Verifica el clipping (límite máximo/mínimo)
    assert _size_from_mag(-30) <= 520
    assert _size_from_mag(20) >= 26


def test_visibility_logic():
    """Verifica que el cálculo de visibilidad funcione según la ubicación."""
    # Buenos Aires
    loc = EarthLocation(lat=-34.6 * u.deg, lon=-58.4 * u.deg, height=20 * u.m)
    t = Time("2026-01-17 12:00:00")  # Mediodía

    altaz_dict, table = compute_altaz(t, loc, nombres=["Sol"])

    sol = next(o for o in table if o.nombre == "Sol")
    assert sol.visible is True
    assert sol.alt_deg > 0


def test_alpha_transparency():
    """Verifica que la transparencia cambie con la altitud."""
    assert _alpha_from_alt(90) == 1.0
    assert _alpha_from_alt(0) == 0.0
    assert _alpha_from_alt(45) > 0.55