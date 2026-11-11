"""Classical baselines and a gate-level resource estimate for the Grover instances.

For every instance of
grover_mcp.py (I1-I5 and the 17 layered DAGs) this script computes

  * Grover: k* = floor(pi / (4 theta)), the best k in 0..2k*+2, and the expected number of
    path evaluations to reach 90 % success when an attempt with k rounds costs k oracle calls
    plus one classical check of the measured path, minimised over k;
  * uniform sampling: evaluations to reach 90 % success;
  * exact dynamic programming over (node, partial totals) labels with upper-bound pruning:
    number of labels created, which decides feasibility and counts M exactly;
  * exact meet-in-the-middle (layered DAGs): forward and backward half-paths stored;
  * sub-sampled meet-in-the-middle (layered DAGs): s random forward and s random backward
    half-paths, all s^2 joins over the middle arcs tested; the smallest 2s that reaches 90 %
    success over TRIALS trials;
  * a Toffoli and logical-qubit estimate of one oracle call and one Grover round, under the
    cost model stated in RESOURCE_MODEL below.

Run: python3 baselines.py  ->  baselines.txt, baselines.json
"""
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qpathfinder"))
from grover_mcp import best_rounds, choose_ranges, marked, metrics_of, path_totals, theory  # noqa: E402
from instances import fig2_dag, layered, transit_mesh                                       # noqa: E402

TRIALS = 400
OUT = []

RESOURCE_MODEL = """Cost model (logical gates, no error-correction overhead):
  controlled addition of a constant into a b-bit register: load the constant into a b-bit
    scratch register with CNOTs controlled on the arc qubit (Clifford), add it with Gidney's
    adder (b - 1 Toffoli, uncomputation of its logical ANDs by measurement), unload;
  subtraction: the same, b - 1 Toffoli;
  comparison of a b-bit register with a constant: b Toffoli to compute, measurement-based uncompute;
  conjunction of the 2K test bits onto the flag: 2K - 1 Toffoli;
  reflection about |0...0> on |E| qubits: |E| - 1 Toffoli;
  U_S: |E| - |V| + 1 controlled RY gates, each two single-qubit RY rotations and two CNOTs;
    one round applies U_S and its inverse, so 4(|E| - |V| + 1) arbitrary rotations.
Logical qubits: |E| arc qubits + sum_k b_k total-register qubits + 1 flag + max_k b_k scratch
  + (max_k b_k - 1) adder ancillas + 2K test bits + (2K - 2) AND ancillas + (|E| - 2) ancillas
  of the reflection (ancillas are reused across steps, so the widest step counts)."""


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def grover_to_90(n, m):
    k_star = best_rounds(n, m)
    best = None
    for k in range(0, 2 * k_star + 3):
        p = theory(n, m, k)
        if p <= 0:
            continue
        r = 1 if p >= 0.9 else math.ceil(math.log(0.1) / math.log(1 - p))
        cost = r * (k + 1)
        if best is None or cost < best[0]:
            best = (cost, k, r, p)
    k_max = max(range(0, 2 * k_star + 3), key=lambda k: theory(n, m, k))
    return dict(k_star=k_star, p_k_star=theory(n, m, k_star), k_argmax=k_max, p_argmax=theory(n, m, k_max),
                evals_90=best[0], k_used=best[1], attempts=best[2], p_used=best[3])


def dp_labels(dag, metrics, U):
    """Forward labelling in topological order; a label is a vector of partial totals with
    T_k <= U_k. Returns labels created and the label multiset at t (totals -> number of paths)."""
    K = len(metrics)
    lab = {dag.s: {(0,) * K: 1}}
    created = 1
    for v in dag.topo:
        if v not in lab:
            continue
        for e in dag.out_arcs[v]:
            w = dag.arcs[e][1]
            tgt = lab.setdefault(w, {})
            for tot, cnt in lab[v].items():
                nt = tuple(tot[k] + metrics[k][e] for k in range(K))
                if all(nt[k] <= U[k] for k in range(K)):
                    if nt not in tgt:
                        created += 1
                    tgt[nt] = tgt.get(nt, 0) + cnt
    return created, lab.get(dag.t, {})


def layered_parts(dag, width, depth, metrics):
    """Weight tensors of a complete layered DAG: first arcs (width, K), middle arcs
    (depth-1, width, width, K), last arcs (width, K)."""
    K = len(metrics)
    idx = {a: i for i, a in enumerate(dag.arcs)}
    node = lambda layer, i: 1 + layer * width + i
    W = np.array(metrics).T                      # arcs x K
    first = np.array([W[idx[(0, node(0, i))]] for i in range(width)])
    mid = np.zeros((depth - 1, width, width, K), dtype=int)
    for layer in range(depth - 1):
        for i in range(width):
            for j in range(width):
                mid[layer, i, j] = W[idx[(node(layer, i), node(layer + 1, j))]]
    last = np.array([W[idx[(node(depth - 1, i), dag.t)]] for i in range(width)])
    return first, mid, last


def sample_halves(rng, s, width, depth, h, first, mid, last):
    """s forward half-paths over layers 0..h-1 and s backward half-paths over layers h..depth-1."""
    fn = rng.integers(0, width, size=(s, h))
    ft = first[fn[:, 0]].copy()
    for layer in range(h - 1):
        ft += mid[layer, fn[:, layer], fn[:, layer + 1]]
    bn = rng.integers(0, width, size=(s, depth - h))
    bt = last[bn[:, -1]].copy()
    for j in range(depth - h - 1):
        layer = h + j
        bt += mid[layer, bn[:, j], bn[:, j + 1]]
    return fn[:, -1], ft, bn[:, 0], bt


def mitm_success(seed, s, width, depth, first, mid, last, L, U, trials=TRIALS):
    rng = np.random.default_rng(seed)
    h = depth // 2
    hits = 0
    for _ in range(trials):
        fu, ft, bv, bt = sample_halves(rng, s, width, depth, h, first, mid, last)
        tot = ft[:, None, :] + mid[h - 1][fu[:, None], bv[None, :]] + bt[None, :, :]
        ok = np.all((tot >= L) & (tot <= U), axis=2)
        hits += bool(ok.any())
    return hits / trials


def mitm_to_90(seed, width, depth, first, mid, last, L, U, n, m):
    """Smallest s on a geometric grid with success >= 0.9; returns 2s (half-paths evaluated)."""
    s = max(2, int(math.sqrt(n / m) * 0.5))
    while True:
        p = mitm_success(seed, s, width, depth, first, mid, last, L, U)
        if p >= 0.9:
            return 2 * s, p
        s = int(math.ceil(s * 1.15))


def resources(dag, bits, K, k_star):
    E, V = dag.n_qubits, dag.n_nodes
    add = sum(E * (b - 1) for b in bits)        # K|E| controlled additions
    sub = add
    cmp_ = sum(2 * b for b in bits)             # two comparators per metric
    conj = 2 * K - 1
    refl = E - 1
    tof_oracle = add + sub + cmp_ + conj
    tof_round = tof_oracle + refl
    rot_round = 4 * (E - V + 1)
    bmax = max(bits)
    qubits = E + sum(bits) + 1 + bmax + (bmax - 1) + 2 * K + max(2 * K - 2, E - 2)
    return dict(toffoli_oracle=tof_oracle, toffoli_round=tof_round, rotations_round=rot_round,
                toffoli_run=tof_round * max(k_star, 1), rotations_run=rot_round * max(k_star, 1),
                logical_qubits=qubits, arc_qubits=E)


def successor_qubits(dag):
    """d * ceil(log2 Delta) for an encoding that stores one successor choice per hop."""
    hops = max(len(p) for p in dag.enumerate_paths())
    delta = max(len(o) for o in dag.out_arcs)
    return hops * max(1, math.ceil(math.log2(delta))) if delta > 1 else 0


def analyse(name, dag, mseed, layered_shape=None, mitm_seed=0):
    metrics = metrics_of(dag, mseed)
    paths, tot = path_totals(dag, metrics)
    _, L, U = choose_ranges(tot, 1)
    good = marked(tot, L, U)
    n, m = len(paths), int(good.sum())
    K = len(metrics)
    bits = [max(1, int(tot[:, k].max()).bit_length()) for k in range(K)]
    g = grover_to_90(n, m)
    labels, at_t = dp_labels(dag, metrics, U)
    m_dp = sum(c for t_, c in at_t.items() if all(L[k] <= t_[k] for k in range(K)))
    assert m_dp == m, (name, m_dp, m)
    hops = max(len(p) for p in paths)
    row = dict(name=name, nodes=dag.n_nodes, arcs=dag.n_qubits, hops=hops, paths=n, feasible=m,
               avg_out=dag.n_qubits / dag.n_nodes, max_out=max(len(o) for o in dag.out_arcs),
               uniform_90=math.log(0.1) / math.log(1 - m / n) if m < n else 1.0,
               dp_labels=labels, bits=bits, successor_qubits=successor_qubits(dag), **g,
               **resources(dag, bits, K, g["k_star"]))
    if layered_shape:
        w, d = layered_shape
        first, mid, last = layered_parts(dag, w, d, metrics)
        h = d // 2
        row["mitm_exact_halves"] = w ** h + w ** (d - h)
        row["mitm_sub_90"], row["mitm_sub_p"] = mitm_to_90(mitm_seed, w, d, first, mid, last,
                                                            np.array(L), np.array(U), n, m)
    return row


def main():
    say(RESOURCE_MODEL)
    say("")
    res = {"model": RESOURCE_MODEL, "instances": [], "scaling": []}
    suite = [("I1", fig2_dag(), 101, None), ("I2", transit_mesh(4, seed=11), 102, None),
             ("I3", layered(2, 3, seed=7, name="I3"), 103, (2, 3)),
             ("I4", layered(3, 3, seed=5, name="I4"), 104, (3, 3)),
             ("I5", layered(4, 4, seed=3, name="I5"), 105, (4, 4))]
    hdr = (f"{'inst':>6} {'|E|':>4} {'N':>7} {'M':>4} {'k*':>3} {'p(k*)':>6} {'kmax':>4} {'p(kmax)':>7} "
           f"{'G90':>5} {'unif90':>8} {'DPlab':>6} {'MITMex':>7} {'MITMsub90':>9} "
           f"{'Tof/orc':>8} {'Tof/run':>9} {'rot/run':>7} {'qubits':>6} {'succ.q':>6}")
    say(hdr)

    def show(r):
        say(f"{r['name']:>6} {r['arcs']:>4} {r['paths']:>7} {r['feasible']:>4} {r['k_star']:>3} {r['p_k_star']:>6.3f} "
            f"{r['k_argmax']:>4} {r['p_argmax']:>7.3f} {r['evals_90']:>5} {r['uniform_90']:>8.1f} {r['dp_labels']:>6} "
            f"{r.get('mitm_exact_halves', '-'):>7} {r.get('mitm_sub_90', '-'):>9} {r['toffoli_oracle']:>8} "
            f"{r['toffoli_run']:>9} {r['rotations_run']:>7} {r['logical_qubits']:>6} {r['successor_qubits']:>6}")

    for name, dag, seed, shape in suite:
        r = analyse(name, dag, seed, shape, mitm_seed=seed)
        res["instances"].append(r)
        show(r)
    cases = [(2, 4), (2, 6), (3, 4), (2, 8), (3, 5), (4, 4), (2, 10), (3, 6), (4, 5), (3, 7), (4, 6), (3, 8),
             (4, 7), (3, 9), (4, 8), (3, 10), (4, 9)]
    for w, d in cases:
        dag = layered(w, d, seed=1000 + 10 * w + d, name=f"L{w}x{d}")
        r = analyse(f"L{w}x{d}", dag, 2000 + 10 * w + d, (w, d), mitm_seed=10 * w + d)
        r.update(w=w, d=d)
        res["scaling"].append(r)
        show(r)

    big = [r for r in res["scaling"] if r["k_star"] >= 4]
    x = np.log([r["paths"] / r["feasible"] for r in big])
    fits = {}
    for key in ("evals_90", "uniform_90", "mitm_sub_90", "dp_labels"):
        fits[key] = float(np.polyfit(x, np.log([r[key] for r in big]), 1)[0])
    res["fit"] = fits
    say("")
    say(f"log-log slopes against N/M over the {len(big)} layered instances with k* >= 4: "
        + ", ".join(f"{k} {v:.3f}" for k, v in fits.items()))
    open(os.path.join(HERE, "baselines.txt"), "w").write("\n".join(OUT) + "\n")
    json.dump(res, open(os.path.join(HERE, "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
