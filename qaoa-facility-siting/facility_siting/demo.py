"""Runnable demo: a facility-siting instance end to end.

    python -m facility_siting.demo                      # synthetic 8-site toy
    python -m facility_siting.demo --real-uk            # named UK industrial towns
    python -m facility_siting.demo --real-uk --qaoa --n 10 --budget 3
                                                        # + QUANTUM (QAOA) column
    python -m facility_siting.demo --real-uk --qaoa --n 10 --no-normalize
                                                        # QAOA without the fix

Builds the QUBO, finds the exact optimum by brute force, runs the classical
(simulated annealing) backend and, with --qaoa, the QAOA backend, then prints
a side-by-side comparison.
"""

import argparse

from . import classical_solver, qaoa_solver
from .brute_force import approximation_ratio, brute_force
from .cost_model import ChampionMaterial, build_costs
from .qubo_builder import CandidateSite, build_qubo


def synthetic_instance(n=8, seed=0):
    """A toy region: n sites on a line, random costs and benefits."""
    import random
    rng = random.Random(seed)
    sites = [CandidateSite(id=i, lat=54.0 + 0.1 * i, lon=-2.0, buildable=True,
                           install_cost=round(rng.uniform(4, 12), 2))
             for i in range(n)]
    A = {i: sites[i].install_cost for i in range(n)}
    B = {i: round(rng.uniform(3, 14), 2) for i in range(n)}
    C = {(i, i + 1): -1.0 for i in range(n - 1)}
    return sites, A, B, C, None


def uk_instance(n=9):
    """Named UK towns; costs from the DAC cost model + approximate public
    storage-hub locations. Material properties (E, C) are illustrative."""
    # Named UK locations spanning the industrial belt, near/far from storage hubs.
    points = [
        ("Scunthorpe", 53.59, -0.65), ("Sheffield", 53.38, -1.47),
        ("Teesside", 54.58, -1.23), ("Port Talbot", 51.59, -3.77),
        ("Grangemouth", 56.01, -3.72), ("Hull", 53.74, -0.33),
        ("Ellesmere Port", 53.28, -2.90), ("Immingham", 53.61, -0.22),
        ("Aberdeen", 57.15, -2.09), ("Bristol", 51.45, -2.59),
        ("Southampton", 50.90, -1.40), ("Norwich", 52.63, 1.30),
    ][:n]
    sites = [CandidateSite(id=i, lat=lat, lon=lon, buildable=True, install_cost=0.0)
             for i, (_, lat, lon) in enumerate(points)]
    A, B, C, bd = build_costs(sites, champion=ChampionMaterial())
    return sites, A, B, C, {"names": [p[0] for p in points], "breakdown": bd}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-uk", action="store_true",
                   help="use named UK towns + DAC cost model + storage hubs")
    p.add_argument("--n", type=int, default=8, help="number of candidate sites")
    p.add_argument("--budget", type=int, default=3, help="K: facilities to build")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qaoa", action="store_true",
                   help="also run the QAOA backend and add a QUANTUM column")
    p.add_argument("--p", type=int, default=qaoa_solver.DEFAULT_P,
                   help="QAOA circuit depth (only with --qaoa)")
    p.add_argument("--shots", type=int, default=qaoa_solver.DEFAULT_SHOTS,
                   help="QAOA measurement shots (only with --qaoa)")
    p.add_argument("--restarts", type=int, default=qaoa_solver.DEFAULT_N_RESTARTS,
                   help="independent QAOA restarts (only with --qaoa)")
    p.add_argument("--no-normalize", action="store_true",
                   help="run QAOA WITHOUT Hamiltonian normalisation, for comparison")
    args = p.parse_args(argv)

    if args.real_uk:
        sites, A, B, C, extra = uk_instance(max(args.n, 9))
        names = extra["names"]
        bd = extra["breakdown"]["per_site"]
    else:
        sites, A, B, C, _ = synthetic_instance(args.n, args.seed)
        names = [f"site {i}" for i in range(len(sites))]
        bd = None

    n = len(sites)
    budget = min(args.budget, n - 1)
    qubo = build_qubo(sites, A, B, C, K=budget)

    mode = "UK (named towns)" if args.real_uk else "synthetic"
    print(f"\n{'='*72}")
    print(f"  {mode}  —  {n} sites,  budget K = {budget}")
    print(f"{'='*72}")
    print(f"  penalty weight P (calibrated) : {qubo.P:.3f}")
    print(f"  qubits / Ising couplings      : {qubo.n} / {len(qubo.J)}")

    if bd:
        print(f"\n  {'site':16}{'A (install+S+Op)':>18}{'B (capture)':>14}"
              f"{'nearest hub':>20}{'km':>8}")
        for i in range(n):
            print(f"  {names[i]:16}{A[i]:>18.3f}{B[i]:>14.3f}"
                  f"{bd[i]['nearest_hub']:>20}{bd[i]['hub_distance_km']:>8.0f}")
        if C:
            print(f"\n  shared-hub discounts C_ij: "
                  + ", ".join(f"({names[i]},{names[j]})={v:+.1f}" for (i, j), v in C.items()))
    else:
        print(f"\n  site :  A (install)   B (benefit)   net (A-B)")
        for i in range(n):
            print(f"   {i}   :   {A[i]:8.2f}     {B[i]:8.2f}     {A[i]-B[i]:+7.2f}")

    EXACT_CEILING = 22
    exact = brute_force(qubo, max_qubits=EXACT_CEILING) if n <= EXACT_CEILING else None
    sa = classical_solver.solve(None, None, qubo=qubo, n_restarts=40,
                                n_sweeps=1500, seed=args.seed)

    qa = None
    if args.qaoa:
        qa = qaoa_solver.solve(qubo=qubo, p=args.p, shots=args.shots, seed=args.seed,
                               n_restarts=args.restarts,
                               normalize=not args.no_normalize, verbose=True)

    def sel(bits):
        return "[" + ", ".join(names[i] if args.real_uk else str(i)
                               for i, b in enumerate(bits) if b == "1") + "]"

    def pct_gap(candidate_value, exact_value):
        """% objective gap vs the exact optimum -- only meaningful when exact
        enumeration was practical (n <= EXACT_CEILING). None otherwise."""
        if exact_value == 0:
            return None
        return abs(candidate_value - exact_value) / abs(exact_value) * 100.0

    header = f"  {'':24}{'EXACT (brute force)':>22}{'CLASSICAL (anneal)':>24}"
    if qa:
        header += f"{'QUANTUM (QAOA)':>20}"
    width = len(header) if qa else 72

    print(f"\n{'-'*width}")
    print(header)
    print(f"{'-'*width}")

    if exact is None:
        print(f"  {'objective value':24}{'— (N>' + str(EXACT_CEILING) + ')':>22}"
              f"{sa.objective_value:>24.4f}"
              + (f"{qa.objective_value:>20.4f}" if qa else ""))
        print(f"  {'feasible (built K)':24}{'—':>22}{str(sa.feasible):>24}"
              + (f"{str(qa.feasible):>20}" if qa else ""))
        if qa:
            opt = qa.extra_info["optimizer"]
            print(f"  {'QAOA optimizer status':24}{'':>22}{'':>24}"
                  f"{(str(opt['success']) + '/' + str(opt['status'])):>20}")
            print(f"  {'QAOA optimizer message':24}{'':>22}{'':>24}{str(opt['message'])}")
            print(f"  {'Hamming weight':24}{'':>22}{'':>24}{qa.extra_info['hamming_weight']:>20}")
            count_str = f"{qa.extra_info['top_count']}/{qa.extra_info['shots']}"
            print(f"  {'top sample count/shots':24}{'':>22}{'':>24}{count_str:>20}")
        print(f"  {'wall-clock (s)':24}{'—':>22}{sa.wall_time_s:>24.4f}"
              + (f"{qa.wall_time_s:>20.4f}" if qa else ""))
        print(f"{'-'*width}")
        print(f"  classical picks: {sel(sa.bitstring)}")
        if qa:
            print(f"  quantum picks   : {sel(qa.bitstring)}"
                 + (f"  [repaired -> {sel(qa.repaired_bitstring)}]" if qa.repaired_bitstring else ""))
        print(f"  no EXACT column: 2^{n} combinations is past the brute-force ceiling,")
        print(f"  so there is no ground truth to check against, and no % objective gap can be reported.\n")
        return

    ar = approximation_ratio(sa.objective_value, exact.objective_value, exact.worst_value)
    qa_ar = (approximation_ratio(qa.objective_value, exact.objective_value, exact.worst_value)
            if qa else None)
    sa_gap = pct_gap(sa.objective_value, exact.objective_value)
    qa_gap = pct_gap(qa.objective_value, exact.objective_value) if qa else None

    def row(label, exact_v, sa_v, qa_v=""):
        line = f"  {label:24}{exact_v:>22}{sa_v:>24}"
        if qa:
            line += f"{qa_v:>20}"
        print(line)

    row("objective value", f"{exact.objective_value:.4f}", f"{sa.objective_value:.4f}",
       f"{qa.objective_value:.4f}" if qa else "")
    row("% gap vs exact", "0.0000", f"{sa_gap:.4f}" if sa_gap is not None else "n/a",
       (f"{qa_gap:.4f}" if qa_gap is not None else "n/a") if qa else "")
    row("feasible (solution)", str(exact.feasible), str(sa.feasible),
       str(qa.feasible) if qa else "")
    if qa:
        opt = qa.extra_info["optimizer"]
        row("QAOA optimizer status", "", "", f"{opt['success']}/{opt['status']}")
        row("Hamming weight", "", "", str(qa.extra_info["hamming_weight"]))
        row("top sample count/shots", "", "",
           f"{qa.extra_info['top_count']}/{qa.extra_info['shots']}")
    row("wall-clock (s)", "—", f"{sa.wall_time_s:.4f}", f"{qa.wall_time_s:.4f}" if qa else "")
    row("states evaluated", str(exact.n_evaluated), "—", "—" if qa else "")
    row("approximation ratio", "1.0000", f"{ar:.4f}", f"{qa_ar:.4f}" if qa else "")

    print(f"{'-'*width}")
    print(f"  exact optimum : {sel(exact.bitstring)}")
    print(f"  classical     : {sel(sa.bitstring)}")
    if qa:
        print(f"  quantum (QAOA): {sel(qa.bitstring)}"
             + (f"  [repaired -> {sel(qa.repaired_bitstring)}]" if qa.repaired_bitstring else ""))
    print(f"  penalty dominates (optimum respects budget): {exact.penalty_dominates}")
    print(f"  classical vs exact: {'MATCH' if sa.bitstring == exact.bitstring else 'DIFFERS'}")
    if qa:
        print(f"  quantum vs exact  : {'MATCH' if qa.bitstring == exact.bitstring else 'DIFFERS'}")
    print()


if __name__ == "__main__":
    main()
