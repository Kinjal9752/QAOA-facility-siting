"""Shared solver interface: SolverResult, evaluate_objective, repair_to_budget.

Both classical (simulated annealing) and quantum (QAOA) backends implement the
solve(h, J, **params) -> SolverResult protocol defined here.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class SolverResult:
    """Outcome of a QUBO solver (classical or quantum).

    bitstring: the solution as a binary string ('0110' for 4 sites)
    objective_value: O(x) = Σ A_i x_i - Σ B_i x_i + Σ C_ij x_i x_j + P(Σ x_i - K)²
    wall_time_s: actual runtime
    feasible: True if Σ x_i == K (budget respected)
    repaired_bitstring: if infeasible, a greedy repair to budget
    extra_info: backend-specific details (e.g., QAOA: p, n_restarts, expectation)
    """
    bitstring: str
    objective_value: float
    wall_time_s: float
    feasible: bool
    repaired_bitstring: Optional[str] = None
    extra_info: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SolverResult":
        return cls(**d)


def evaluate_objective(bitstring: str, h: Dict[int, float], J: Dict[tuple, float],
                       const: float = 0.0) -> float:
    """Compute O(x) for a given bitstring under the Ising Hamiltonian.

    H = Σ h_i Z_i + Σ J_ij Z_i Z_j + const
    x_i = (1 - z_i) / 2, z_i ∈ {-1, +1}

    Args:
        bitstring: '0110' format (x_i ∈ {0, 1})
        h: {site_i: bias}
        J: {(site_i, site_j): coupling}, assumed i < j
        const: constant term

    Returns: the objective value
    """
    x = [int(bit) for bit in bitstring]
    n = len(x)

    # Expand to Ising form: z_i = 1 - 2*x_i
    z = [1 - 2*xi for xi in x]

    # Σ h_i Z_i
    linear_term = sum(h.get(i, 0.0) * z[i] for i in range(n))

    # Σ J_ij Z_i Z_j
    quadratic_term = sum(J.get((i, j), 0.0) * z[i] * z[j]
                         for i in range(n) for j in range(i+1, n))

    return linear_term + quadratic_term + const


def repair_to_budget(bitstring: str, budget: int, h: Dict[int, float],
                     J: Dict[tuple, float], const: float = 0.0) -> str:
    """Greedily repair an infeasible bitstring to exactly K facilities.

    If Σ x_i < K: add the K - Σ x_i sites with smallest cost increase.
    If Σ x_i > K: remove the K - Σ x_i sites with largest cost decrease.

    Args:
        bitstring: starting solution ('0110')
        budget: target number of 1s (K)
        h, J, const: QUBO parameters for cost evaluation

    Returns: repaired bitstring with exactly K sites selected
    """
    x = [int(bit) for bit in bitstring]
    current_sum = sum(x)

    if current_sum == budget:
        return bitstring

    # Marginal cost of flipping each site (at current state)
    n = len(x)
    z = [1 - 2*xi for xi in x]

    marginal_cost = {}
    for i in range(n):
        # Cost to flip x_i from its current value
        z_new = -z[i]
        delta_linear = h.get(i, 0.0) * (z_new - z[i])
        delta_quad = sum(J.get((i, j), 0.0) * (z_new - z[i]) * z[j]
                         for j in range(n) if j != i)
        marginal_cost[i] = delta_linear + delta_quad

    if current_sum < budget:
        # Add sites with smallest cost increase
        zeros = [i for i in range(n) if x[i] == 0]
        zeros.sort(key=lambda i: marginal_cost.get(i, 0.0))
        for i in zeros[:budget - current_sum]:
            x[i] = 1
    else:
        # Remove sites with largest cost decrease (most negative marginal cost)
        ones = [i for i in range(n) if x[i] == 1]
        ones.sort(key=lambda i: marginal_cost.get(i, 0.0))
        for i in ones[:current_sum - budget]:
            x[i] = 0

    return ''.join(str(xi) for xi in x)
