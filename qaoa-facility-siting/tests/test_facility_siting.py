import itertools

import numpy as np

from facility_siting import classical_solver, qaoa_solver
from facility_siting.brute_force import brute_force
from facility_siting.demo import synthetic_instance, uk_instance
from facility_siting.qubo_builder import build_qubo
from facility_siting.solver_interface import evaluate_objective


def _toy_qubo(n=6, K=2, seed=1):
    sites, A, B, C, _ = synthetic_instance(n, seed)
    return build_qubo(sites, A, B, C, K=K)


def test_qubo_and_ising_forms_agree_on_every_state():
    qubo = _toy_qubo()
    for bits in itertools.product("01", repeat=qubo.n):
        b = "".join(bits)
        assert np.isclose(qubo.evaluate(b), evaluate_objective(b, qubo.h, qubo.J, qubo.const))


def test_penalty_makes_optimum_respect_budget():
    for K in (1, 2, 3):
        result = brute_force(_toy_qubo(K=K))
        assert result.penalty_dominates
        assert result.bitstring.count("1") == K


def test_normalisation_preserves_ranking_of_all_states():
    sites, A, B, C, _ = uk_instance(10)
    qubo = build_qubo(sites, A, B, C, K=3)
    h_s, J_s, scale = qaoa_solver._normalize(qubo.h, qubo.J)
    assert scale > 1
    raw = qaoa_solver._diagonal_energies(qubo.h, qubo.J, qubo.n)
    scaled = qaoa_solver._diagonal_energies(h_s, J_s, qubo.n)
    assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(scaled, kind="stable"))
    assert np.argmin(raw) == np.argmin(scaled)


def test_statevector_is_normalised():
    qubo = _toy_qubo()
    energies = qaoa_solver._diagonal_energies(qubo.h, qubo.J, qubo.n)
    psi = qaoa_solver.qaoa_state(np.array([0.3, 0.7, 0.2, 0.5]), energies, qubo.n)
    assert np.isclose(np.sum(np.abs(psi) ** 2), 1.0)


def test_annealing_matches_brute_force_on_toy():
    qubo = _toy_qubo()
    exact = brute_force(qubo)
    sa = classical_solver.solve(None, None, qubo=qubo, n_restarts=10, n_sweeps=500)
    assert np.isclose(sa.objective_value, exact.objective_value)


def test_qaoa_finds_feasible_solution_on_toy():
    qubo = _toy_qubo()
    qa = qaoa_solver.solve(qubo=qubo, p=3, shots=2000, n_restarts=3, seed=0)
    assert qa.feasible
