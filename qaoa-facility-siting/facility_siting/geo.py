"""Great-circle distance. One place, verified against a known pair.

London (51.5074, -0.1278) -> Edinburgh (55.9533, -3.1883) is 534 km by the
standard reference; haversine_km gives 533.6 km.
"""

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
