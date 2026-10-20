"""Classical backend: simulated annealing on the Ising (h, J).

Single-spin-flip Metropolis with a geometric inverse-temperature schedule and
multiple random restarts. No qubit ceiling — this is the backend that scales
past QAOA, which is itself part of the classical-vs-quantum comparison.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .qubo_builder import QUBO
from .solver_interface import SolverResult, repair_to_budget


def _ising_arrays(h: Dict[int, float], J: Dict[Tuple[int, int], float], n: int):
    h_vec = np.zeros(n)
    for i, hi in h.items():
        h_vec[i] = hi
    J_mat = np.zeros((n, n))
    for (i, j), jij in J.items():
        J_mat[i, j] = jij
        J_mat[j, i] = jij
    return h_vec, J_mat


def _energy(z: np.ndarray, h_vec: np.ndarray, J_mat: np.ndarray, const: float) -> float:
    return float(h_vec @ z + 0.5 * z @ J_mat @ z + const)


def solve(h: Dict[int, float], J: Dict[Tuple[int, int], float],
          const: float = 0.0,
          n: Optional[int] = None,
          qubo: Optional[QUBO] = None,
          K: Optional[int] = None,
          n_sweeps: int = 2000,
          n_restarts: int = 20,
          beta0: float = 0.05,
          beta1: float = 20.0,
          seed: int = 0) -> SolverResult:
    """Simulated annealing on H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const.

    Args:
        h, J, const: Ising Hamiltonian (from QUBO)
        n: number of spins; inferred from qubo or h/J if omitted
        qubo: if given, objective_value is the true O(x) and feasibility uses qubo.K
        K: budget for feasibility check (overrides qubo.K)
        n_sweeps: spin-flip sweeps per restart
        n_restarts: independent random starts; best is returned
        beta0, beta1: start/end inverse temperature (geometric schedule)
        seed: RNG seed

    Returns: SolverResult
    """
    if qubo is not None:
        n = qubo.n
        const = qubo.const
        h, J = qubo.h, qubo.J
    if n is None:
        n = (max(max(h, default=-1),
                 max((k for pair in J for k in pair), default=-1)) + 1)
    if K is None and qubo is not None:
        K = qubo.K

    h_vec, J_mat = _ising_arrays(h, J, n)
    rng = np.random.default_rng(seed)
    betas = np.geomspace(beta0, beta1, n_sweeps)

    start = time.perf_counter()
    best_z = None
    best_e = float("inf")

    for _ in range(n_restarts):
        z = rng.choice([-1, 1], size=n).astype(float)
        e = _energy(z, h_vec, J_mat, const)
        for beta in betas:
            order = rng.permutation(n)
            for i in order:
                # ΔE for flipping spin i: z_i -> -z_i
                local = h_vec[i] + J_mat[i] @ z
                delta = -2.0 * z[i] * local
                if delta <= 0 or rng.random() < np.exp(-beta * delta):
                    z[i] = -z[i]
                    e += delta
        if e < best_e:
            best_e, best_z = e, z.copy()

    wall = time.perf_counter() - start

    # z -> x: x_i = (1 - z_i) / 2
    x = ((1 - best_z) / 2).astype(int)
    bitstring = "".join(str(b) for b in x)

    if qubo is not None:
        objective = qubo.evaluate(bitstring)
    else:
        objective = best_e

    feasible = (K is None) or (bitstring.count("1") == K)
    repaired = None
    if not feasible and K is not None:
        repaired = repair_to_budget(bitstring, K, h, J, const)

    return SolverResult(
        bitstring=bitstring,
        objective_value=objective,
        wall_time_s=wall,
        feasible=feasible,
        repaired_bitstring=repaired,
        extra_info={
            "backend": "simulated_annealing",
            "n_sweeps": n_sweeps,
            "n_restarts": n_restarts,
            "ising_energy": best_e,
            "beta_schedule": [beta0, beta1],
        },
    )
