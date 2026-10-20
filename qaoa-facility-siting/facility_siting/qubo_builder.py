"""QUBO builder: per-site costs + budget K → Ising (h, J).

Objective  O(x) = Σ A_i x_i − Σ B_i x_i + Σ C_ij x_i x_j + P (Σ x_i − K)²
with x_i ∈ {0, 1}. Mapped to the Ising form H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const
via x_i = (1 − z_i) / 2. The (h, J) pair is the only thing the solvers see.

"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Sequence, Tuple


@dataclass
class CandidateSite:
    """A potential facility location."""
    id: int
    lat: float
    lon: float
    buildable: bool
    install_cost: float
    capture_potential: float = 0.0
    storage_cost: float = 0.0
    energy_cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateSite":
        return cls(**d)


@dataclass
class QUBO:
    """A facility-location instance in both QUBO and Ising form.

    QUBO form (x ∈ {0,1}):   O(x) = Σ linear[i] x_i + Σ quadratic[(i,j)] x_i x_j + offset
    Ising form (z ∈ {-1,1}): H(z) = Σ h[i] z_i    + Σ J[(i,j)] z_i z_j       + const
    The two are equal under x_i = (1 - z_i) / 2.
    """
    n: int
    K: int
    P: float
    linear: Dict[int, float]
    quadratic: Dict[Tuple[int, int], float]
    offset: float
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    const: float
    meta: Dict = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.n

    def evaluate(self, bitstring: str) -> float:
        """True objective O(x) for a bitstring ('0110'), via the QUBO form."""
        x = [int(b) for b in bitstring]
        if len(x) != self.n:
            raise ValueError(f"bitstring length {len(x)} != n {self.n}")
        val = self.offset
        for i in range(self.n):
            val += self.linear.get(i, 0.0) * x[i]
        for (i, j), q in self.quadratic.items():
            val += q * x[i] * x[j]
        return val

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON needs string keys for tuple-keyed dicts
        d["quadratic"] = {f"{i},{j}": v for (i, j), v in self.quadratic.items()}
        d["J"] = {f"{i},{j}": v for (i, j), v in self.J.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QUBO":
        d = dict(d)
        d["linear"] = {int(k): v for k, v in d["linear"].items()}
        d["h"] = {int(k): v for k, v in d["h"].items()}
        d["quadratic"] = {_pair(k): v for k, v in d["quadratic"].items()}
        d["J"] = {_pair(k): v for k, v in d["J"].items()}
        return cls(**d)


def _pair(key) -> Tuple[int, int]:
    if isinstance(key, (tuple, list)):
        return (int(key[0]), int(key[1]))
    a, b = str(key).split(",")
    return (int(a), int(b))


def qubo_to_ising(linear: Mapping[int, float],
                  quadratic: Mapping[Tuple[int, int], float],
                  offset: float,
                  n: int) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """Convert a QUBO (x ∈ {0,1}) to Ising (z ∈ {-1,+1}) under x_i = (1 - z_i)/2.

        h_i   = -q_i/2 - (1/4) Σ_{j≠i} q_{ij}
        J_ij  =  q_ij/4                            (i < j)
        const =  offset + (1/2) Σ_i q_i + (1/4) Σ_{i<j} q_ij
    """
    h = {i: 0.0 for i in range(n)}
    J: Dict[Tuple[int, int], float] = {}
    const = offset

    for i in range(n):
        q_i = linear.get(i, 0.0)
        h[i] += -q_i / 2.0
        const += q_i / 2.0

    for (i, j), q_ij in quadratic.items():
        a, b = (i, j) if i < j else (j, i)
        J[(a, b)] = J.get((a, b), 0.0) + q_ij / 4.0
        h[a] += -q_ij / 4.0
        h[b] += -q_ij / 4.0
        const += q_ij / 4.0

    return h, J, const


def calibrate_penalty(A: Mapping[int, float], B: Mapping[int, float],
                      C: Mapping[Tuple[int, int], float], n: int,
                      safety: float = 1.25) -> float:
    """Penalty weight P that dominates any single-site change to the budget.

    Moving a solution from |Σx - K| = m to m-1 drops the penalty by P·(m² - (m-1)²)
    ≥ P, and m = 1 is the binding case: it is sufficient that P exceeds the
    largest objective swing from adding or removing any one site,

        P > max_i ( |A_i - B_i| + Σ_j |C_ij| )

    This is much tighter than the sum-over-all-sites bound in Ayodele (2022) and
    keeps the energy landscape well-conditioned. `verify_facility_qubo` asserts
    the resulting optimum actually respects the budget.
    """
    coupling_abs = {i: 0.0 for i in range(n)}
    for (i, j), c in C.items():
        coupling_abs[i] = coupling_abs.get(i, 0.0) + abs(c)
        coupling_abs[j] = coupling_abs.get(j, 0.0) + abs(c)
    max_swing = max(
        (abs(A.get(i, 0.0) - B.get(i, 0.0)) + coupling_abs.get(i, 0.0)
         for i in range(n)),
        default=0.0,
    )
    return safety * max_swing + 1.0


def build_qubo(sites: Sequence[CandidateSite],
               A: Mapping[int, float],
               B: Mapping[int, float],
               C: Mapping[Tuple[int, int], float],
               K: int,
               P: float = None,
               penalty_safety: float = 1.25) -> QUBO:
    """Build the facility-location QUBO and its Ising form.

    Args:
        sites: candidate sites, index i is position in this list
        A: {i: install-side cost}  (A_i + S_i + Op_i already combined by cost_model)
        B: {i: capture benefit}
        C: {(i,j): pairwise coupling}, i < j
        K: facility budget
        P: penalty weight; computed via calibrate_penalty if None
        penalty_safety: safety factor for the computed P

    Returns: QUBO
    """
    n = len(sites)
    if not 0 <= K <= n:
        raise ValueError(f"budget K={K} out of range [0, {n}]")

    if P is None:
        P = calibrate_penalty(A, B, C, n, safety=penalty_safety)

    # QUBO coefficients (x ∈ {0,1}).
    # Penalty: P(Σx_i - K)² = P(1-2K) Σx_i + 2P Σ_{i<j} x_i x_j + P K²
    linear: Dict[int, float] = {}
    for i in range(n):
        linear[i] = A.get(i, 0.0) - B.get(i, 0.0) + P * (1 - 2 * K)

    quadratic: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = C.get((i, j), 0.0) + 2.0 * P

    offset = P * K * K

    h, J, const = qubo_to_ising(linear, quadratic, offset, n)

    return QUBO(
        n=n, K=K, P=P,
        linear=linear, quadratic=quadratic, offset=offset,
        h=h, J=J, const=const,
        meta={"n_sites": n, "budget": K, "penalty": P},
    )
