"""Storage-connection cost S_i: distance from each site to the nearest real UK
CO2 transport tie-in hub, times a cost-per-km.

The DAC pathway's main site differentiator. Reuses haversine_km from geo.py.
Data: data/uk_storage_hubs.json (4 publicly announced UK CO2 transport and
storage clusters, approximate coordinates for illustration only).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import json
from .geo import haversine_km

HUBS_PATH = Path(__file__).resolve().parent / "data" / "uk_storage_hubs.json"

# £ per km of CO2 transport spur to the storage network. Placeholder — revisit
# with real pipeline-cost data.
DEFAULT_COST_PER_KM = 0.05


@dataclass
class StorageHub:
    name: str
    cluster: str
    region: str
    lat: float
    lon: float


def load_hubs(path=HUBS_PATH) -> List[StorageHub]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [StorageHub(name=h["name"], cluster=h["cluster"], region=h["region"],
                       lat=h["lat"], lon=h["lon"]) for h in raw["hubs"]]


def nearest_hub(lat: float, lon: float, hubs: Sequence[StorageHub]):
    """(hub, distance_km) for the closest hub to a point."""
    best = min(hubs, key=lambda h: haversine_km(lat, lon, h.lat, h.lon))
    return best, haversine_km(lat, lon, best.lat, best.lon)


def compute_storage_cost(sites, hubs=None, cost_per_km: float = DEFAULT_COST_PER_KM) -> Dict[int, dict]:
    """For each site, the cost of connecting it to the nearest storage hub.

    Args:
        sites: objects with .id, .lat, .lon (e.g. CandidateSite)
        hubs: list of StorageHub; loaded from the packaged file if None
        cost_per_km: £ per km of transport spur

    Returns: {site_id: {"storage_cost": float, "nearest_hub": str,
                        "distance_km": float}}
    """
    if hubs is None:
        hubs = load_hubs()
    out = {}
    for s in sites:
        hub, dist = nearest_hub(s.lat, s.lon, hubs)
        out[s.id] = {
            "storage_cost": cost_per_km * dist,
            "nearest_hub": hub.name,
            "distance_km": dist,
        }
    return out
