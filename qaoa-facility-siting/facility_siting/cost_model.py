"""Cost model: turn site locations + material properties (E, C) into the
A_i / B_i / C_ij coefficients qubo_builder consumes.

Direct-air-capture (DAC) pathway:
    A_i combined = A_base * remoteness(i) * C_material  +  S_i  +  Op_i
    B_i          = E_material * AMBIENT_CO2_FRACTION * capacity   (≈ constant)
    C_ij         = shared-trunk-pipeline discount when two sites feed one hub

Every weight lives in DAC_COST_PARAMS so the coefficients are auditable, not
buried in a formula.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from .geo import haversine_km
from .storage_proximity import compute_storage_cost, load_hubs

# Ambient CO2 mole fraction (~420 ppm). Fixed across the whole region — this is
# the atmospheric fact that makes DAC a real facility-location problem.
AMBIENT_CO2_FRACTION = 420e-6

DAC_COST_PARAMS = {
    "a_base": 10.0,              # base install cost (land + construction), arbitrary units
    "storage_cost_per_km": 0.05,  # £/km spur to the storage network
    "capacity": 1.0,             # facility capture capacity (relative)
    "benefit_scale": 1.0e6,      # scales E * ambient * capacity into the same units as A
    "shared_hub_discount": 2.0,  # magnitude of the negative C_ij for hub-sharing pairs
    "shared_hub_radius_km": 60.0,  # two sites closer than this that feed one hub share a trunk
    "energy_cost": 0.0,          # Op_i placeholder (no energy-price layer yet)
}


@dataclass
class ChampionMaterial:
    """Capture-material properties. The defaults are ILLUSTRATIVE placeholders,
    not measured values for any real material."""
    efficiency: float = 0.30            # E: fraction of contacted CO2 actually captured
    production_cost: float = 1.0        # C: relative material production cost -> A_i multiplier
    provenance: dict = field(default_factory=lambda: {"stub": True})

    def to_dict(self):
        return {"efficiency": self.efficiency,
                "production_cost": self.production_cost,
                "provenance": self.provenance}


def _remoteness_multiplier(site) -> float:
    """>= 1.0; higher for sites far from infrastructure. Stub: flat 1.0 until a
    road/grid distance layer exists."""
    return 1.0


def build_costs(sites: Sequence,
                champion: Optional[ChampionMaterial] = None,
                params: Optional[dict] = None,
                hubs=None) -> Tuple[Dict[int, float], Dict[int, float],
                                    Dict[Tuple[int, int], float], dict]:
    """Assemble (A, B, C) for the DAC pathway.

    Args:
        sites: CandidateSite objects (need .id, .lat, .lon)
        champion: capture material; illustrative placeholder used if None
        params: overrides for DAC_COST_PARAMS
        hubs: StorageHub list; packaged file used if None

    Returns: (A, B, C, breakdown) where breakdown[site_id] itemises the terms.
    """
    champion = champion or ChampionMaterial()
    p = {**DAC_COST_PARAMS, **(params or {})}
    if hubs is None:
        hubs = load_hubs()

    storage = compute_storage_cost(sites, hubs, cost_per_km=p["storage_cost_per_km"])

    A: Dict[int, float] = {}
    B: Dict[int, float] = {}
    breakdown: Dict[int, dict] = {}

    b_const = (champion.efficiency * AMBIENT_CO2_FRACTION
               * p["capacity"] * p["benefit_scale"])

    for i, s in enumerate(sites):
        a_base = p["a_base"] * _remoteness_multiplier(s) * champion.production_cost
        s_i = storage[s.id]["storage_cost"]
        op_i = p["energy_cost"]
        A[i] = a_base + s_i + op_i
        B[i] = b_const
        breakdown[i] = {
            "a_base": a_base, "storage_cost": s_i, "energy_cost": op_i,
            "A_combined": A[i], "B": B[i],
            "nearest_hub": storage[s.id]["nearest_hub"],
            "hub_distance_km": storage[s.id]["distance_km"],
        }

    # C_ij: two sites that would feed the same hub and sit within the trunk radius
    # of each other share a pipeline -> a discount (negative coupling).
    C: Dict[Tuple[int, int], float] = {}
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            same_hub = breakdown[i]["nearest_hub"] == breakdown[j]["nearest_hub"]
            d = haversine_km(sites[i].lat, sites[i].lon, sites[j].lat, sites[j].lon)
            if same_hub and d <= p["shared_hub_radius_km"]:
                C[(i, j)] = -p["shared_hub_discount"]

    return A, B, C, {"per_site": breakdown, "b_const": b_const, "params": p}
