from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeoResult:
    latitude: float
    longitude: float
    altitude_m: float


async def _get_windows_location_async() -> GeoResult:
    # Import tardío: si no existe winrt, levanta ImportError
    from winrt.windows.devices.geolocation import Geolocator  # type: ignore

    locator = Geolocator()
    locator.desired_accuracy_in_meters = 50

    pos = await locator.get_geoposition_async()
    p = pos.coordinate.point.position
    alt = float(getattr(p, "altitude", 0.0) or 0.0)

    return GeoResult(latitude=float(p.latitude), longitude=float(p.longitude), altitude_m=alt)


def get_windows_location_sync(timeout_s: float = 10.0) -> GeoResult:
    """
    Bloqueante. Llamar desde un thread (no desde el hilo de UI).
    """
    # No usamos timeout agresivo en WinRT; si querés, luego lo agregamos con asyncio.wait_for.
    return asyncio.run(_get_windows_location_async())
