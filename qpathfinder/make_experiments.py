"""Builds experiments.json: one entry per hardware circuit in the experiment plan, with its gate
list, the ideal output distribution over valid path strings, and the optimum string.

Every gate list with <= 24 qubits is replayed on the gate-level simulator and compared with the
ideal distribution before it is written. Larger circuits are checked against the classical
(DP-based) formula only.
"""
import json
import math

import numpy as np

from instances import suite
from qpathfinder import (StateVector, SubspaceGMQAOA, best_angles, conditional_probs, cx_count,
                         dequantized_ratio, dequantized_terms, exact_csp, gibbs_probs,
                         gm_qaoa_gates, larac_lambda, score_counts, state_prep_gates)

rng = np.random.default_rng(20261110)
out, report = [], []


def ideal_from_probs(dag, paths, probs):
    return {dag.flow_string(p): float(q) for p, q in zip(paths, probs)}


def add(exp, dag, label, gates, ideal, optimum, note=""):
    n = dag.n_qubits
    if n <= 24:
        sim = StateVector(n).run(gates).probabilities()
        keys = set(sim) | set(ideal)
        err = max(abs(sim.get(k, 0) - ideal.get(k, 0)) for k in keys)
        assert err < 1e-9, (exp, dag.name, err)
        check = f"gate-level check passed ({err:.0e})"
    else:
        check = "classical formula only (too many qubits to replay here)"
    n_refl = sum(1 for g in gates if g[0] == "refl0")
    entry = dict(experiment=exp, instance=dag.name, circuit=label, n_qubits=n,
                 cx_equivalent_without_reflection=cx_count([g for g in gates if g[0] != "refl0"]),
                 reflections=n_refl, gates=[list(g) for g in gates], ideal=ideal, optimum=optimum,
                 ideal_p_opt=ideal.get(optimum, 0.0), note=note)
    out.append(entry)
    report.append(f"{exp:4s} {dag.name:22s} {label:34s} qubits={n:2d} CX={entry['cx_equivalent_without_reflection']:4d} "
                  f"+{n_refl} reflection(s)  ideal P(opt)={entry['ideal_p_opt']:.4f}  {check}")


meta = {}
for k, (dag, bound) in enumerate(suite(), start=1):
    best, opt, paths, c, d = exact_csp(dag, bound)
    F = len(paths)
    good = np.zeros(F, bool); good[opt] = True
    optimum = dag.flow_string(paths[opt[0]])
    lam = larac_lambda(c, d, bound)
    pg, tau = max((gibbs_probs(c, d, lam, t)[opt].sum(), t) for t in np.linspace(0, 6, 601))
    tau = float(min(tau, 3.0))                       # cap: beyond ~3 the state is almost a single path pair
    pi = gibbs_probs(c, d, lam, tau)
    L = [dag.cost[e] + lam * dag.delay[e] for e in range(dag.n_qubits)]
    meta[dag.name] = dict(nodes=dag.n_nodes, arcs=dag.arcs, cost=dag.cost, delay=dag.delay, s=dag.s, t=dag.t,
                          delay_bound=bound, optimum=optimum, optimum_path=dag.path_nodes(paths[opt[0]]),
                          optimum_cost=best, optimum_delay=d[opt[0]], n_paths=F, larac_lambda=lam, tau=tau)

    uni = state_prep_gates(dag, "uniform")
    if k == 1:   # E0: the 2020 circuit's distribution on the Figure 2 graph
        walk = conditional_probs(dag, "walk")
        probs = [math.prod(walk[e] for e in p) for p in paths]
        add("E0", dag, "equal-split prep (2020)", state_prep_gates(dag, "walk"), ideal_from_probs(dag, paths, probs), optimum)
    add("E1", dag, "uniform prep", uni, ideal_from_probs(dag, paths, [1 / F] * F), optimum)

    # decoder A, one round, uniform start (E3 for I1..I4, E5 for I5)
    sim = SubspaceGMQAOA(dag)
    p1, x1 = best_angles(sim, good, 1, rng)
    terms = dequantized_terms(dag, x1[:1], x1[1:2], x1[2:])
    probs = [dequantized_ratio(terms, c[i], d[i]) / F for i in range(F)]
    assert abs(sum(probs) - 1) < 1e-9 and abs(sum(probs[i] for i in opt) - p1) < 1e-9
    add("E5" if k == 5 else "E3", dag, "uniform prep + decoder A, 1 round",
        gm_qaoa_gates(dag, x1[:1], x1[1:2], x1[2:], prep=uni), ideal_from_probs(dag, paths, probs), optimum,
        note=f"gamma_cost={x1[0]:.6f}, gamma_delay={x1[1]:.6f}, beta={x1[2]:.6f}")

    if k in (2, 5):   # duality-gap instances: Boltzmann preparation
        gib = state_prep_gates(dag, "gibbs", L, tau)
        add("E4a", dag, "Boltzmann prep", gib, ideal_from_probs(dag, paths, pi), optimum,
            note=f"lambda={lam:.6f}, tau={tau:.4f}")
        gsim = SubspaceGMQAOA(dag, prep_probs=pi)
        g1, y1 = best_angles(gsim, good, 1, rng)
        cond = conditional_probs(dag, "gibbs", L, tau)
        terms = dequantized_terms(dag, y1[:1], y1[1:2], y1[2:], prep_cond=cond)
        probs = [dequantized_ratio(terms, c[i], d[i]) * pi[i] for i in range(F)]
        assert abs(sum(probs) - 1) < 1e-9
        add("E4b", dag, "Boltzmann prep + decoder A, 1 round",
            gm_qaoa_gates(dag, y1[:1], y1[1:2], y1[2:], prep=gib), ideal_from_probs(dag, paths, probs), optimum,
            note=f"lambda={lam:.6f}, tau={tau:.4f}, gamma_cost={y1[0]:.6f}, gamma_delay={y1[1]:.6f}, beta={y1[2]:.6f}")

json.dump({"instances": meta, "circuits": out}, open("experiments.json", "w"), indent=1)
print("\n".join(report))
print(f"\nwrote experiments.json: {len(out)} circuits")

# scoring self-test: sample the ideal E1/I3 distribution, add 20% uniform garbage, score it
e = next(x for x in out if x["experiment"] == "E1" and x["n_qubits"] == 12)
dag = [g for g, _ in suite()][2]
keys = list(e["ideal"]); pr = np.array([e["ideal"][k] for k in keys])
counts = {}
for b in rng.choice(keys, size=3277, p=pr / pr.sum()):
    counts[b] = counts.get(b, 0) + 1
for v in rng.integers(0, 2 ** 12, size=819):
    b = format(int(v), "012b"); counts[b] = counts.get(b, 0) + 1
print("scoring self-test (80% ideal shots + 20% random strings):",
      {k: (round(v, 4) if isinstance(v, float) else v) for k, v in score_counts(dag, counts, e["ideal"], e["optimum"]).items()})
