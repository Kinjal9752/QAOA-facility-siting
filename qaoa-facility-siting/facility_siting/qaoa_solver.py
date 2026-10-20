"""Quantum backend: QAOA on the Ising (h, J), simulated exactly with NumPy.

Same solve(...) -> SolverResult contract as classical_solver, so the two are
drop-in interchangeable and can be compared against the brute-force optimum.

The simulator is a plain statevector: 2^N complex amplitudes, the diagonal
cost layer exp(-i*gamma*H_C) applied elementwise, and the mixer
exp(-i*beta*X) applied qubit by qubit. That is exact (no hardware noise) and
comfortably fast up to N ~ 16-18 on a laptop.

Three design decisions carried over from the original project:

1. HAMILTONIAN NORMALISATION. With realistic cost data the (h, J)
   coefficients can be tens of times larger than in a toy instance. QAOA's
   angles then sit in a badly conditioned landscape and the optimiser
   stalls: in the original runs it converged to states that selected every
   site. Before QAOA runs, a COPY of (h, J) is divided by

       scale = max(max_i |h_i|, max_ij |J_ij|)

   Dividing every coefficient by one positive constant is a strictly
   monotonic transformation of the energy, so the ranking of all 2^N states
   (and therefore which state is optimal) is unchanged; only the numerical
   conditioning the optimiser sees improves. The returned bitstring is always
   scored against the ORIGINAL, unscaled QUBO.

2. MULTIPLE RESTARTS. Identical QAOA runs from different starting angles can
   land in very different places (near-optimal, or an infeasible all-zeros
   state). solve() therefore runs `n_restarts` independent restarts and keeps
   the best FEASIBLE result by original-QUBO objective. The optimiser's own
   "success" flag is recorded but never used to choose the winner.

3. EXPLICIT INFEASIBILITY. If no restart respects the budget, the best overall
   result is returned with extra_info["any_feasible"] = False rather than a
   silent fallback.

Because this simulator runs locally, `seed` makes the WHOLE run reproducible:
the starting angles and the measurement sampling are both seeded.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .qubo_builder import QUBO
from .solver_interface import SolverResult, evaluate_objective, repair_to_budget

DEFAULT_P = 6
DEFAULT_SHOTS = 5000
DEFAULT_N_RESTARTS = 5
MAX_QUBITS = 20


def _normalize(h: Dict[int, float], J: Dict[Tuple[int, int], float]):
    """Divide h and J by scale = max(max|h_i|, max|J_ij|). Returns
    (h_scaled, J_scaled, scale); scale = 1.0 for an all-zero Hamiltonian."""
    max_h = max((abs(v) for v in h.values()), default=0.0)
    max_j = max((abs(v) for v in J.values()), default=0.0)
    scale = max(max_h, max_j)
    if scale == 0:
        return dict(h), dict(J), 1.0
    return ({i: v / scale for i, v in h.items()},
            {k: v / scale for k, v in J.items()},
            scale)


def _bit_matrix(n: int) -> np.ndarray:
    """(2^n, n) array of x-values. Row k is basis state k, MSB-first, so
    column i is site i (matches format(k, f'0{n}b'))."""
    codes = np.arange(1 << n, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((codes[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def _diagonal_energies(h: Dict[int, float], J: Dict[Tuple[int, int], float],
                       n: int) -> np.ndarray:
    """Ising energy sum h_i z_i + sum J_ij z_i z_j for every basis state
    (z_i = 1 - 2 x_i). The constant term is dropped: it only shifts the
    energy and has no effect on QAOA."""
    Z = 1.0 - 2.0 * _bit_matrix(n)
    energies = np.zeros(1 << n)
    for i, hi in h.items():
        energies += hi * Z[:, i]
    for (i, j), jij in J.items():
        energies += jij * Z[:, i] * Z[:, j]
    return energies


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i * beta * X) to every qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for q in range(n):
        psi = np.moveaxis(psi, q, 0)
        a0, a1 = psi[0].copy(), psi[1].copy()
        psi[0] = c * a0 + s * a1
        psi[1] = s * a0 + c * a1
        psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def qaoa_state(params: np.ndarray, energies: np.ndarray, n: int) -> np.ndarray:
    """Statevector after p layers. params = [gamma_1..gamma_p, beta_1..beta_p]."""
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)   # |+>^n
    for gamma, beta in zip(gammas, betas):
        psi = psi * np.exp(-1j * gamma * energies)                # cost layer
        psi = _apply_mixer(psi, beta, n)                          # mixer layer
    return psi


def _run_single(energies: np.ndarray, n: int, p: int, shots: int,
                init: np.ndarray, rng: np.random.Generator,
                maxiter: int) -> Tuple[str, dict]:
    """Optimise the 2p angles from `init`, then sample the final circuit."""
    def expectation(params):
        probs = np.abs(qaoa_state(params, energies, n)) ** 2
        return float(probs @ energies)

    res = minimize(expectation, init, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})

    probs = np.abs(qaoa_state(res.x, energies, n)) ** 2
    probs /= probs.sum()
    samples = rng.choice(len(probs), size=shots, p=probs)
    counts = np.bincount(samples, minlength=len(probs))
    top = int(np.argmax(counts))
    bitstring = format(top, f"0{n}b")

    info = {
        "optimizer": {"success": bool(res.success), "status": int(res.status),
                      "message": str(res.message), "nfev": int(res.nfev)},
        "expectation_value_scaled": float(res.fun),
        "top_probability": float(probs[top]),
        "top_count": int(counts[top]),
        "n_distinct_sampled": int((counts > 0).sum()),
        "final_parameters": res.x.tolist(),
    }
    return bitstring, info


def solve(h: Optional[Dict[int, float]] = None,
          J: Optional[Dict[Tuple[int, int], float]] = None,
          const: float = 0.0,
          n: Optional[int] = None,
          qubo: Optional[QUBO] = None,
          K: Optional[int] = None,
          p: int = DEFAULT_P,
          shots: int = DEFAULT_SHOTS,
          n_restarts: int = DEFAULT_N_RESTARTS,
          seed: int = 0,
          normalize: bool = True,
          maxiter: int = 3000,
          verbose: bool = False) -> SolverResult:
    """QAOA on H = sum h_i Z_i + sum J_ij Z_i Z_j + const, with restarts.

    Args:
        h, J, const: Ising Hamiltonian; or pass qubo= instead
        n: number of spins; inferred from qubo or h/J if omitted
        qubo: if given, the objective and feasibility use the ORIGINAL QUBO
        K: budget for the feasibility check (overrides qubo.K)
        p: circuit depth (number of cost + mixer layers)
        shots: measurement samples drawn from the optimised circuit
        n_restarts: independent restarts from different starting angles
        seed: seeds both the starting angles and the measurement sampling
        normalize: rescale (h, J) before optimising (see module docstring);
            set False to reproduce the un-normalised behaviour
        maxiter: COBYLA iteration limit per restart
        verbose: print one line per restart

    Returns: SolverResult for the best feasible restart (see module docstring).
    """
    if qubo is not None:
        n, const, h, J = qubo.n, qubo.const, qubo.h, qubo.J
    if n is None:
        n = (max(max(h, default=-1),
                 max((k for pair in J for k in pair), default=-1)) + 1)
    if n > MAX_QUBITS:
        raise ValueError(f"N={n} qubits is past this simulator's limit ({MAX_QUBITS}).")
    if K is None and qubo is not None:
        K = qubo.K

    h_run, J_run, scale = _normalize(h, J) if normalize else (dict(h), dict(J), 1.0)
    energies = _diagonal_energies(h_run, J_run, n)

    def objective(bitstring):
        if qubo is not None:
            return qubo.evaluate(bitstring)
        return evaluate_objective(bitstring, h, J, const)

    rng = np.random.default_rng(seed)
    inits = [rng.uniform(0, np.pi, size=2 * p) for _ in range(n_restarts)]

    restarts: List[dict] = []
    for i, init in enumerate(inits):
        t0 = time.perf_counter()
        bitstring, info = _run_single(energies, n, p, shots, init, rng, maxiter)
        wall = time.perf_counter() - t0
        feasible = (K is None) or (bitstring.count("1") == K)
        restarts.append({
            "index": i, "bitstring": bitstring, "objective": objective(bitstring),
            "feasible": feasible, "hamming_weight": bitstring.count("1"),
            "wall_time_s": wall, "initial_parameters": init.tolist(), **info,
        })
        if verbose:
            r = restarts[-1]
            print(f"  [restart {i}] {bitstring}  weight={r['hamming_weight']}  "
                  f"feasible={feasible}  objective={r['objective']:.4f}  "
                  f"top={r['top_count']}/{shots}  ({wall:.1f}s)", flush=True)

    feasible_restarts = [r for r in restarts if r["feasible"]]
    if feasible_restarts:
        best = min(feasible_restarts, key=lambda r: r["objective"])
        selection_reason = "best_feasible"
    else:
        best = min(restarts, key=lambda r: r["objective"])
        selection_reason = "no_feasible_found_using_best_overall"

    repaired = None
    if not best["feasible"] and K is not None:
        repaired = repair_to_budget(best["bitstring"], K, h, J, const)

    return SolverResult(
        bitstring=best["bitstring"],
        objective_value=best["objective"],
        wall_time_s=sum(r["wall_time_s"] for r in restarts),
        feasible=best["feasible"],
        repaired_bitstring=repaired,
        extra_info={
            "backend": "numpy_statevector",
            "p": p, "shots": shots, "normalize": normalize, "scale": scale,
            "n_restarts": n_restarts,
            "n_feasible_restarts": len(feasible_restarts),
            "selected_restart_index": best["index"],
            "selection_reason": selection_reason,
            "any_feasible": bool(feasible_restarts),
            "optimizer": best["optimizer"],
            "top_probability": best["top_probability"],
            "top_count": best["top_count"],
            "hamming_weight": best["hamming_weight"],
            "restarts": restarts,
        },
    )
