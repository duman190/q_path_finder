"""Grover search for a multi-constrained path with range constraints on a DAG.

Problem: K additive integer metrics on the arcs, a range [L_k, U_k] per metric.
Find an s-t path whose K totals all fall in their ranges.

Two simulators:
  * register level: arc qubits x K accumulator registers, every gate of U_S applied one by one,
    the oracle as controlled additions, range test, and un-computation (small instances);
  * path subspace: amplitudes over the enumerated paths (all instances).
Run: python3 grover_mcp.py  ->  results.txt, results.json
"""
import json
import math
import random
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qpathfinder"))
from instances import fig2_dag, layered, transit_mesh          # noqa: E402
from qpathfinder import PathDAG, inverse, state_prep_gates      # noqa: E402

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ------------------------------------------------------------------------------------------
# instances: the five DAGs I1-I5 plus a third metric (seeded)
# ------------------------------------------------------------------------------------------
def third_metric(dag, seed):
    rng = random.Random(seed)
    return [rng.randint(1, 4) for _ in dag.arcs]


def metrics_of(dag, seed):
    return [list(dag.cost), list(dag.delay), third_metric(dag, seed)]


def path_totals(dag, metrics):
    paths = dag.enumerate_paths()
    tot = np.array([[sum(m[e] for e in p) for m in metrics] for p in paths], dtype=int)
    return paths, tot


def choose_ranges(tot, half_width):
    """Ranges centred on the most balanced path: the one whose totals, each scaled to [0, 1]
    over all paths, have the smallest maximum. Deterministic; guarantees a nonempty answer set."""
    lo, hi = tot.min(axis=0), tot.max(axis=0)
    span = np.maximum(hi - lo, 1)
    score = ((tot - lo) / span).max(axis=1)
    target = int(np.argmin(score))
    L = tot[target] - half_width
    U = tot[target] + half_width
    return target, L, U


def marked(tot, L, U):
    return np.all((tot >= L) & (tot <= U), axis=1)


# ------------------------------------------------------------------------------------------
# path-subspace Grover
# ------------------------------------------------------------------------------------------
def subspace_curve(good, rounds):
    n = len(good)
    f = np.full(n, 1 / math.sqrt(n))
    a = f.copy()
    out = [float(np.sum(a[good] ** 2))]
    for _ in range(rounds):
        a = np.where(good, -a, a)
        a = 2 * f * np.dot(f, a) - a
        out.append(float(np.sum(a[good] ** 2)))
    return out


def theory(n, m, k):
    th = math.asin(math.sqrt(m / n))
    return math.sin((2 * k + 1) * th) ** 2


def best_rounds(n, m):
    th = math.asin(math.sqrt(m / n))
    return max(0, int(math.floor(math.pi / (4 * th))))


# ------------------------------------------------------------------------------------------
# register-level simulator: axis 0..E-1 are arc qubits, then K accumulator registers
# ------------------------------------------------------------------------------------------
class RegisterSim:
    def __init__(self, dag, metrics, L, U):
        self.dag, self.metrics, self.L, self.U = dag, metrics, L, U
        self.E = dag.n_qubits
        self.K = len(metrics)
        self.bits = [max(1, int(sum(m)).bit_length()) for m in metrics]   # no overflow for any bit string
        self.shape = (2,) * self.E + tuple(2 ** b for b in self.bits)
        self.psi = np.zeros(self.shape, dtype=complex)
        self.psi[(0,) * (self.E + self.K)] = 1.0
        self.prep = state_prep_gates(dag, "uniform")
        self.counts = {"x": 0, "cx": 0, "cry": 0, "cadd": 0, "range_test": 0, "refl0": 0}

    # --- gates on arc qubits -------------------------------------------------------------
    def _sl(self, fixed):
        idx = [slice(None)] * (self.E + self.K)
        for q, v in fixed.items():
            idx[q] = v
        return tuple(idx)

    def gate(self, g):
        self.counts[g[0]] += 1
        if g[0] == "x":
            self.psi = np.flip(self.psi, axis=g[1])
        elif g[0] == "cx":
            c, t = g[1], g[2]
            a0, a1 = self._sl({c: 1, t: 0}), self._sl({c: 1, t: 1})
            tmp = self.psi[a0].copy()
            self.psi[a0] = self.psi[a1]
            self.psi[a1] = tmp
        elif g[0] == "cry":
            th, c, t = g[1], g[2], g[3]
            a0, a1 = self._sl({c: 1, t: 0}), self._sl({c: 1, t: 1})
            v0, v1 = self.psi[a0].copy(), self.psi[a1].copy()
            co, si = math.cos(th / 2), math.sin(th / 2)
            self.psi[a0] = co * v0 - si * v1
            self.psi[a1] = si * v0 + co * v1
        else:
            raise ValueError(g)

    def run(self, gates):
        for g in gates:
            self.gate(g)

    # --- oracle ---------------------------------------------------------------------------
    def controlled_add(self, e, k, sign):
        """acc_k += sign * w_k(e) (mod 2^bits) on the branch where arc qubit e is 1."""
        self.counts["cadd"] += 1
        s = self._sl({e: 1})
        sub = self.psi[s]
        axis = self.E - 1 + k            # one arc axis was removed by the slice
        self.psi[s] = np.roll(sub, sign * self.metrics[k][e], axis=axis)

    def range_phase(self):
        self.counts["range_test"] += self.K
        mask = np.ones((1,) * self.E + tuple(2 ** b for b in self.bits), dtype=float)
        for k in range(self.K):
            v = np.arange(2 ** self.bits[k])
            ok = (v >= self.L[k]) & (v <= self.U[k])
            shp = [1] * (self.E + self.K)
            shp[self.E + k] = len(v)
            mask = mask * ok.reshape(shp)
        self.psi = self.psi * (1 - 2 * mask)

    def oracle(self):
        for k in range(self.K):
            for e in range(self.E):
                self.controlled_add(e, k, +1)
        self.range_phase()
        for k in reversed(range(self.K)):
            for e in reversed(range(self.E)):
                self.controlled_add(e, k, -1)

    def diffusion(self):
        self.run(inverse(self.prep))
        self.counts["refl0"] += 1
        zero = self._sl({q: 0 for q in range(self.E)})
        keep = self.psi[zero].copy()
        self.psi = -self.psi
        self.psi[zero] = keep
        self.run(self.prep)

    # --- read-out -------------------------------------------------------------------------
    def arc_distribution(self):
        """Probability of each arc bit string, and the weight left outside accumulator = 0."""
        acc0 = (slice(None),) * self.E + (0,) * self.K
        p = np.abs(self.psi[acc0]) ** 2
        leak = float(np.sum(np.abs(self.psi) ** 2) - np.sum(p))
        return p, leak


def register_check(dag, metrics, L, U, rounds):
    paths, tot = path_totals(dag, metrics)
    good = marked(tot, L, U)
    sim = RegisterSim(dag, metrics, L, U)
    sim.run(sim.prep)
    worst, curve = 0.0, []
    for k in range(rounds + 1):
        p, leak = sim.arc_distribution()
        probs = []
        for path in paths:
            idx = [0] * dag.n_qubits
            for e in path:
                idx[e] = 1
            probs.append(float(p[tuple(idx)]))
        probs = np.array(probs)
        invalid = float(p.sum() - probs.sum())
        succ = float(probs[good].sum())
        curve.append(succ)
        worst = max(worst, abs(succ - theory(len(paths), int(good.sum()), k)), abs(leak), abs(invalid))
        if k < rounds:
            sim.oracle()
            sim.diffusion()
    return curve, worst, sim


# ------------------------------------------------------------------------------------------
def main():
    say("=" * 90)
    say("A. Instances and range constraints (K = 3 metrics: cost, delay, third metric)")
    say("=" * 90)
    suite = [("I1", fig2_dag(), 101, 1), ("I2", transit_mesh(4, seed=11), 102, 1),
             ("I3", layered(2, 3, seed=7, name="I3"), 103, 1), ("I4", layered(3, 3, seed=5, name="I4"), 104, 1),
             ("I5", layered(4, 4, seed=3, name="I5"), 105, 1)]
    res = {"instances": [], "scaling": []}
    for name, dag, seed, hw in suite:
        metrics = metrics_of(dag, seed)
        paths, tot = path_totals(dag, metrics)
        target, L, U = choose_ranges(tot, hw)
        good = marked(tot, L, U)
        n, m = len(paths), int(good.sum())
        k_opt = best_rounds(n, m)
        curve = subspace_curve(good, max(k_opt + 3, 4))
        err = max(abs(c - theory(n, m, k)) for k, c in enumerate(curve))
        # single-constraint feasibility counts, to show that the conjunction is what makes it hard
        single = [int(((tot[:, k] >= L[k]) & (tot[:, k] <= U[k])).sum()) for k in range(3)]
        sim_bits = [max(1, int(sum(mm)).bit_length()) for mm in metrics]        # width used by the register simulator
        need_bits = [max(1, int(tot[:, k].max()).bit_length()) for k in range(3)]  # width a real circuit needs
        classical_90 = math.log(0.1) / math.log(1 - m / n) if m < n else 1.0
        row = dict(name=name, nodes=dag.n_nodes, arcs=dag.n_qubits, paths=n, ranges=[[int(a), int(b)] for a, b in zip(L, U)],
                   feasible=m, feasible_single=single, k_opt=k_opt, p0=curve[0], p_opt=curve[k_opt], curve=curve,
                   acc_bits=need_bits, sim_bits=sim_bits, qubits=dag.n_qubits + sum(need_bits) + 1,
                   classical_walks_90=classical_90, subspace_vs_formula=err)
        res["instances"].append(row)
        say(f"{name}: |V|={dag.n_nodes} |E|={dag.n_qubits} paths N={n} ranges={row['ranges']} "
            f"in range per metric={single} all three M={m}")
        say(f"    Grover rounds k*={k_opt}: success {curve[0]:.4f} -> {curve[k_opt]:.4f} "
            f"(formula {theory(n, m, k_opt):.4f}); |subspace - formula| <= {err:.1e}")
        say(f"    random walks for 90%: {classical_90:.1f}; oracle calls for Grover: {k_opt}; "
            f"accumulator bits {need_bits}; qubits = {dag.n_qubits} + {sum(need_bits)} + 1 = {row['qubits']}")

    say("")
    say("=" * 90)
    say("B. Register-level simulation (every gate of U_S, controlled adders, range test, un-computation)")
    say("=" * 90)
    for (name, dag, seed, hw), row in zip(suite[:2], res["instances"][:2]):
        metrics = metrics_of(dag, seed)
        L, U = [r[0] for r in row["ranges"]], [r[1] for r in row["ranges"]]
        rounds = row["k_opt"] + 2
        curve, worst, sim = register_check(dag, metrics, L, U, rounds)
        size = sim.psi.size
        say(f"{name}: state of {dag.n_qubits} arc qubits x accumulators {sim.bits} bits = {size:,} amplitudes")
        say(f"    success after 0..{rounds} rounds: {[round(c, 6) for c in curve]}")
        say(f"    max deviation from sin^2((2k+1)theta), accumulator leak, invalid strings: {worst:.1e}")
        say(f"    gate counts over {rounds} rounds: {sim.counts}")
        row["register_curve"], row["register_worst"] = curve, worst
        assert worst < 1e-12

    say("")
    say("=" * 90)
    say("C. Scaling on layered DAGs (width w, depth d, N = w^d paths), path-subspace simulation")
    say("=" * 90)
    cases = [(2, 4), (2, 6), (3, 4), (2, 8), (3, 5), (4, 4), (2, 10), (3, 6), (4, 5), (3, 7), (4, 6), (3, 8), (4, 7), (3, 9), (4, 8), (3, 10), (4, 9)]
    for w, d in cases:
        dag = layered(w, d, seed=1000 + 10 * w + d, name=f"L{w}x{d}")
        metrics = metrics_of(dag, 2000 + 10 * w + d)
        paths, tot = path_totals(dag, metrics)
        target, L, U = choose_ranges(tot, 1)
        good = marked(tot, L, U)
        n, m = len(paths), int(good.sum())
        k_opt = best_rounds(n, m)
        curve = subspace_curve(good, k_opt)
        classical_90 = math.log(0.1) / math.log(1 - m / n)
        res["scaling"].append(dict(w=w, d=d, arcs=dag.n_qubits, paths=n, feasible=m, k_opt=k_opt, p_opt=curve[k_opt],
                                   classical_walks_90=classical_90, classical_mean=n / m))
        say(f"w={w} d={d:2d} |E|={dag.n_qubits:3d} N={n:6d} M={m:4d} N/M={n / m:9.1f} k*={k_opt:3d} "
            f"success={curve[k_opt]:.4f} walks for 90%={classical_90:9.1f}")
    big = [r for r in res["scaling"] if r["k_opt"] >= 4]
    x = np.log([r["paths"] / r["feasible"] for r in big])
    yq = np.log([r["k_opt"] for r in big])
    yc = np.log([r["classical_walks_90"] for r in big])
    sq, iq = np.polyfit(x, yq, 1)
    sc, ic = np.polyfit(x, yc, 1)
    res["fit"] = dict(grover_slope=float(sq), classical_slope=float(sc), points=len(big))
    say(f"log-log slope against N/M over the {len(big)} instances with k* >= 4: Grover rounds {sq:.3f} (theory 0.5), random walks {sc:.3f} (theory 1)")

    open(os.path.join(HERE, "results.txt"), "w").write("\n".join(OUT) + "\n")
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
