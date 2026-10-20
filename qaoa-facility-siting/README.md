# QAOA Facility Siting

Choosing where to build carbon-capture facilities, formulated as a QUBO and solved three ways: **QAOA** (a variational quantum algorithm, simulated exactly in NumPy), **simulated annealing**, and **brute-force enumeration** as ground truth.

This repository is my part of a three-person team project for the Fujitsu Quantum Simulator Challenge 2025–26, where I led the QUBO/QAOA pipeline. The original ran on Fujitsu's own quantum-simulation software. This version is rebuilt with open-source tools so it runs on any laptop, and it contains no Fujitsu code, data or outputs.

## The problem

Given *N* candidate sites and a budget of *K* facilities, pick the *K* sites that minimise

```
O(x) = Σ A_i x_i  −  Σ B_i x_i  +  Σ C_ij x_i x_j  +  P (Σ x_i − K)²        x_i ∈ {0, 1}
```

- **A_i**: install cost, plus the cost of a pipeline spur to the nearest CO₂ storage hub
- **B_i**: capture benefit
- **C_ij**: a discount when two nearby sites can share one trunk pipeline
- **P**: a penalty weight that enforces the budget. It is calibrated automatically to be large enough that the optimum always builds exactly *K* sites, but no larger, so the energy landscape stays well-conditioned.

Substituting `x_i = (1 − z_i)/2` turns this into an Ising Hamiltonian `H = Σ h_i Z_i + Σ J_ij Z_i Z_j`. Each site becomes one qubit, and the lowest-energy state is the best set of sites.

## How it works

```
sites + costs ──► qubo_builder ──► Ising (h, J) ──┬──► qaoa_solver       (quantum)
                                                  ├──► classical_solver  (simulated annealing)
                                                  └──► brute_force       (exact ground truth)
```

| File | What it does |
|---|---|
| `qubo_builder.py` | Builds the QUBO, calibrates the penalty, maps it to Ising form |
| `cost_model.py` | Turns site locations and material properties into A, B, C |
| `storage_proximity.py`, `geo.py` | Distance from each site to the nearest storage hub |
| `qaoa_solver.py` | QAOA statevector simulator, with normalisation and multi-restart selection |
| `classical_solver.py` | Simulated annealing with random restarts |
| `brute_force.py` | Vectorised enumeration of all 2^N states; approximation ratio |
| `demo.py` | Runs everything and prints a side-by-side comparison |

## Debugging QAOA: Hamiltonian normalisation

With realistic costs, the Ising coefficients were dozens of times larger than in the toy problem. QAOA's optimiser then saw a badly scaled landscape: it stopped almost immediately and returned solutions that built the wrong number of facilities.

The fix is to divide every coefficient by the largest one before optimising. Dividing by a single positive constant keeps the ranking of all 2^N states the same, so the optimal answer is unchanged, and the angles QAOA has to find become well-conditioned. `tests/` checks that the ranking is identical across all 1,024 states of the 10-site instance. Results are always scored against the original, unscaled QUBO.

Two more lessons are built into `qaoa_solver.py`:
- **Restarts matter.** Identical runs from different starting angles land in very different places, sometimes on the empty "build nothing" state. The solver runs several restarts and keeps the best one that respects the budget.
- **Don't trust the optimiser's success flag.** "Failed" runs can land on good answers and "successful" ones can be infeasible, so selection uses only the true objective.

## Results

Run on a laptop CPU with depth p = 6, 5,000 shots and 5 restarts (`--seed 0`). Full outputs are in `results/`.

| Instance | Exact optimum | Simulated annealing | QAOA | QAOA gap | Feasible QAOA restarts |
|---|---|---|---|---|---|
| Toy, 8 sites, K = 3 | −13.18 | −13.18 (exact) | −13.18 (exact) | 0.00% | 4 / 5 |
| UK towns, 10 sites, K = 3 (normalised) | −351.63 | −351.63 (exact) | −348.14 | 0.99% | 2 / 5 |
| UK towns, 10 sites, K = 3 (not normalised) | −351.63 | −351.63 (exact) | −343.13 | 2.42% | 1 / 5 |

What the results show:
- **Simulated annealing** finds the exact optimum on both instances, and the brute-force check confirms that the QUBO enforces the budget.
- **QAOA** solves the toy problem exactly. On the 10-site UK instance it gets within 1% of the optimum (approximation ratio 0.9995).
- **Without normalisation**, the optimiser stops after very few evaluations and 4 of 5 restarts pick the wrong number of sites. With normalisation it explores properly, though some restarts still collapse to the empty state. That's why restarts are part of the method.
- At this size, classical methods are faster and more reliable. The aim is to understand how QAOA behaves and where it breaks down, not to claim a quantum advantage.

## Run it

```bash
pip install -r requirements.txt

python -m facility_siting.demo                                        # toy, exact vs annealing
python -m facility_siting.demo --qaoa                                 # + QAOA
python -m facility_siting.demo --real-uk --qaoa --n 10 --budget 3     # UK towns
python -m facility_siting.demo --real-uk --qaoa --n 10 --no-normalize # without the fix
python -m pytest tests -q
```

## Limitations

- **Illustrative data.** The material properties and cost weights are placeholders, and the storage-hub coordinates are approximate public locations. The numbers show how the method behaves; they are not a real siting recommendation.
- **Exact, noise-free simulation.** This is fine up to about 16–18 qubits. Real quantum hardware would add noise.
- **Standard X mixer.** A budget-preserving mixer (e.g. XY) would keep QAOA inside the feasible space and is a natural next step.
