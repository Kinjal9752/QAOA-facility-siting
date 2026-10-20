"""Brute-force ground truth for the facility QUBO (N ≤ ~22).

Independent of any solver: enumerates all 2^N bitstrings to find the exact
optimum. Used to (1) validate that qubo_builder produced the right (h, J), and
(2) score both backends' approximation ratio.

"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .qubo_builder import QUBO

DEFAULT_MAX_QUBITS = 22
_CHUNK_ROWS = 1 << 20   # 2^20 bitstrings per chunk -- bounds memory regardless of N


@dataclass
class BruteForceResult:
    bitstring: str
    objective_value: float
    feasible: bool                 # Σ x_i == K
    n_evaluated: int
    worst_value: float = 0.0       # max objective over all states (for approx. ratio)
    best_feasible_bitstring: Optional[str] = None
    best_feasible_value: Optional[float] = None
    best_infeasible_value: Optional[float] = None
    penalty_dominates: bool = False   # global optimum respects the budget
    top_k: List[tuple] = field(default_factory=list)  # [(bitstring, value), ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _linear_vector(qubo: QUBO) -> np.ndarray:
    a = np.zeros(qubo.n)
    for i, v in qubo.linear.items():
        a[i] = v
    return a


def _quadratic_matrix(qubo: QUBO) -> np.ndarray:
    """Symmetric n x n matrix with Q[i,j] = Q[j,i] = quadratic[(i,j)], 0 diagonal.

    x^T Q x (elementwise-summed) then double-counts each i<j pair, so callers
    take 0.5 * that sum -- see _evaluate_batch.
    """
    Q = np.zeros((qubo.n, qubo.n))
    for (i, j), v in qubo.quadratic.items():
        Q[i, j] = v
        Q[j, i] = v
    return Q


def _bits_batch(codes: np.ndarray, n: int) -> np.ndarray:
    """codes (shape (m,)) -> bit matrix (m, n), MSB-first (bit 0 = site 0),
    matching format(code, f'0{n}b')."""
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((codes[:, None].astype(np.int64) >> shifts[None, :]) & 1).astype(np.int8)


def _evaluate_batch(X: np.ndarray, a: np.ndarray, Q: np.ndarray, offset: float) -> np.ndarray:
    """O(x) for every row of X (m, n), fully vectorized (BLAS matmul, not a
    Python loop) -- this is what makes brute force tractable past N~18."""
    Xf = X.astype(np.float64)
    linear_term = Xf @ a
    quad_term = 0.5 * np.einsum("ij,jk,ik->i", Xf, Q, Xf)
    return offset + linear_term + quad_term


def brute_force(qubo: QUBO, max_qubits: int = DEFAULT_MAX_QUBITS,
                top_k: int = 5) -> BruteForceResult:
    """Enumerate all 2^N bitstrings and return the exact optimum.

    Vectorized: bitstrings are evaluated in chunks via matrix ops (BLAS), not
    one Python-level QUBO.evaluate() call per state -- a naive per-bitstring
    loop is O(N^2 * 2^N) with heavy interpreter overhead and stalls well
    before N=24; this chunked matmul form evaluates each chunk in one shot.

    Raises ValueError if N > max_qubits (2^24 is the default ceiling; raise
    it explicitly to go further, but this is fundamentally exponential).
    """
    n = qubo.n
    if n > max_qubits:
        raise ValueError(
            f"N={n} exceeds max_qubits={max_qubits}; brute force is 2^N. "
            f"Raise max_qubits explicitly to override."
        )

    a = _linear_vector(qubo)
    Q = _quadratic_matrix(qubo)
    total = 1 << n

    best_val, best_bits = float("inf"), None
    best_feasible_val, best_feasible_bits = float("inf"), None
    best_infeasible_val = float("inf")
    worst_val = float("-inf")
    # running top_k as parallel (values, bitstrings) arrays, kept sorted ascending
    top_vals = np.array([], dtype=np.float64)
    top_bits: List[str] = []

    start = 0
    while start < total:
        stop = min(start + _CHUNK_ROWS, total)
        codes = np.arange(start, stop, dtype=np.int64)
        X = _bits_batch(codes, n)
        vals = _evaluate_batch(X, a, Q, qubo.offset)
        popcounts = X.sum(axis=1)
        feasible_mask = popcounts == qubo.K

        i_min = int(np.argmin(vals))
        if vals[i_min] < best_val:
            best_val = float(vals[i_min])
            best_bits = format(int(codes[i_min]), f"0{n}b")

        if feasible_mask.any():
            fvals = np.where(feasible_mask, vals, np.inf)
            i_fmin = int(np.argmin(fvals))
            if fvals[i_fmin] < best_feasible_val:
                best_feasible_val = float(fvals[i_fmin])
                best_feasible_bits = format(int(codes[i_fmin]), f"0{n}b")
        if (~feasible_mask).any():
            ivals = np.where(~feasible_mask, vals, np.inf)
            i_imin = int(np.argmin(ivals))
            if ivals[i_imin] < best_infeasible_val:
                best_infeasible_val = float(ivals[i_imin])

        worst_val = max(worst_val, float(np.max(vals)))

        # merge this chunk's smallest top_k into the running top_k
        k = min(top_k, len(vals))
        local_idx = np.argpartition(vals, k - 1)[:k]
        local_idx = local_idx[np.argsort(vals[local_idx])]
        merged_vals = np.concatenate([top_vals, vals[local_idx]])
        merged_bits = top_bits + [format(int(codes[i]), f"0{n}b") for i in local_idx]
        order = np.argsort(merged_vals)[:top_k]
        top_vals = merged_vals[order]
        top_bits = [merged_bits[i] for i in order]

        start = stop

    return BruteForceResult(
        bitstring=best_bits,
        objective_value=best_val,
        feasible=best_bits.count("1") == qubo.K,
        n_evaluated=total,
        worst_value=worst_val,
        best_feasible_bitstring=best_feasible_bits,
        best_feasible_value=best_feasible_val if best_feasible_bits else None,
        best_infeasible_value=best_infeasible_val if best_infeasible_val != float("inf") else None,
        penalty_dominates=(best_bits.count("1") == qubo.K),
        top_k=list(zip(top_bits, top_vals.tolist())),
    )


def approximation_ratio(candidate_value: float, exact_value: float,
                        worst_value: float) -> float:
    """Quality of a solution on [0, 1], where 1.0 is the exact optimum.

        r = (worst - candidate) / (worst - exact)

    This is the standard QAOA benchmarking ratio (Farhi et al.): it is
    well-defined for negative objectives, unlike a raw candidate/exact ratio.
    `worst_value` is the maximum objective over all states (from brute force).
    """
    span = worst_value - exact_value
    if span == 0:
        return 1.0
    return (worst_value - candidate_value) / span
