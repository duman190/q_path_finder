"""
qpathfinder.py  --  reference implementation for the qPathFinder note (2026 revision)

What is in here
---------------
1. PathDAG            : a single-commodity DAG (source s, sink t) with per-arc cost and delay.
2. state_prep_gates() : gate list for U_S, the circuit that prepares a superposition over
                        ALL s-t paths, one qubit per arc ("arc-indicator" encoding).
                        mode="uniform"  -> every path has the same amplitude (angles from a
                                           classical path-count DP)
                        mode="walk"     -> equal split at every node (the 2020 circuit's
                                           distribution; NOT uniform over paths)
                        mode="gibbs"    -> amplitude ~ exp(-tau * L(path) / 2) for any additive
                                           arc length L (angles from a partition-function DP)
3. StateVector        : tiny numpy gate-level simulator (X, CX, CRY, phase, reflection on |0..0>).
4. GM-QAOA            : phase separator = one single-qubit phase gate per arc (cost and delay
                        are additive), mixer = U_S (reflection about |0..0>) U_S^dagger.
                        Simulated two ways: gate level (small instances) and inside the
                        path subspace (any instance whose paths can be enumerated).
5. Classical baselines: exact constrained shortest path by enumeration, LARAC multiplier,
                        Boltzmann ("Gibbs") path sampling.
6. to_qiskit()        : thin adapter that turns a gate list into a qiskit QuantumCircuit.
                        NOTE: qiskit was not installed where this file was written, so this
                        adapter is the one part of the file that has NOT been executed.

Bit-string convention: flow_string() prints q_{n-1} ... q_0 (qiskit order), so strings can be
compared directly with IBM histograms.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# 1. Graph
# --------------------------------------------------------------------------------------
@dataclass
class PathDAG:
    n_nodes: int
    arcs: list            # list of (u, v); arc i lives on qubit i
    s: int
    t: int
    cost: list            # per-arc additive cost
    delay: list           # per-arc additive delay
    name: str = ""
    node_names: list = field(default_factory=list)

    def __post_init__(self):
        if not self.node_names:
            self.node_names = [f"n{i}" for i in range(self.n_nodes)]
        self.out_arcs = [[] for _ in range(self.n_nodes)]
        self.in_arcs = [[] for _ in range(self.n_nodes)]
        for i, (u, v) in enumerate(self.arcs):
            self.out_arcs[u].append(i)
            self.in_arcs[v].append(i)
        self.topo = self._topological_order()
        self._check_pruned()

    @property
    def n_qubits(self):
        return len(self.arcs)

    def _topological_order(self):
        indeg = [len(a) for a in self.in_arcs]
        order, stack = [], [v for v in range(self.n_nodes) if indeg[v] == 0]
        while stack:
            v = stack.pop()
            order.append(v)
            for e in self.out_arcs[v]:
                w = self.arcs[e][1]
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack.append(w)
        if len(order) != self.n_nodes:
            raise ValueError("graph has a cycle")
        return order

    def _check_pruned(self):
        """Every arc must lie on some s-t path (prune the graph classically first)."""
        n_to_t = self.path_counts()
        reach = [False] * self.n_nodes
        reach[self.s] = True
        for v in self.topo:
            if reach[v]:
                for e in self.out_arcs[v]:
                    reach[self.arcs[e][1]] = True
        for i, (u, v) in enumerate(self.arcs):
            if not reach[u] or n_to_t[v] == 0:
                raise ValueError(f"arc {i} {(u, v)} is on no s-t path; prune it first")

    def path_counts(self):
        """N[v] = number of v->t paths (classical DP, O(|E|))."""
        n = [0] * self.n_nodes
        n[self.t] = 1
        for v in reversed(self.topo):
            if v != self.t:
                n[v] = sum(n[self.arcs[e][1]] for e in self.out_arcs[v])
        return n

    def partition(self, weight):
        """Z[v] = sum over v->t paths of prod(weight[e])  (classical DP, O(|E|))."""
        z = [0.0] * self.n_nodes          # weights may be complex (see char_fn)
        z[self.t] = 1.0
        for v in reversed(self.topo):
            if v != self.t:
                z[v] = sum(weight[e] * z[self.arcs[e][1]] for e in self.out_arcs[v])
        return z

    def enumerate_paths(self):
        """All s-t paths as tuples of arc indices (exponential in general; fine for tests)."""
        paths = []

        def rec(v, acc):
            if v == self.t:
                paths.append(tuple(acc))
                return
            for e in self.out_arcs[v]:
                acc.append(e)
                rec(self.arcs[e][1], acc)
                acc.pop()

        rec(self.s, [])
        return paths

    def path_metric(self, path, values):
        return sum(values[e] for e in path)

    def flow_string(self, path):
        bits = ["0"] * self.n_qubits
        for e in path:
            bits[e] = "1"
        return "".join(reversed(bits))            # q_{n-1} ... q_0

    def path_nodes(self, path):
        return "->".join([self.node_names[self.s]] + [self.node_names[self.arcs[e][1]] for e in path])

    # ---- classical decoding of a measured bit string -------------------------------
    def decode(self, bitstring):
        """Flow-conservation check. bitstring is q_{n-1}..q_0. Returns the path (tuple of
        arcs) if the string is a valid s-t path, else None. This is the error-detection step."""
        x = [int(b) for b in reversed(bitstring)]
        for v in range(self.n_nodes):
            fin = sum(x[e] for e in self.in_arcs[v])
            fout = sum(x[e] for e in self.out_arcs[v])
            want_in = 0 if v == self.s else (1 if v == self.t else None)
            want_out = 1 if v == self.s else (0 if v == self.t else None)
            if v in (self.s, self.t):
                if fin != want_in or fout != want_out:
                    return None
            elif fin != fout or fin > 1:
                return None
        path, v = [], self.s
        while v != self.t:
            nxt = [e for e in self.out_arcs[v] if x[e]]
            if len(nxt) != 1:
                return None
            path.append(nxt[0])
            v = self.arcs[nxt[0]][1]
        return tuple(path) if len(path) == sum(x) else None

    def code_distance(self):
        """Minimum Hamming distance between two valid path strings."""
        ps = [set(p) for p in self.enumerate_paths()]
        if len(ps) < 2:
            return None
        return min(len(a ^ b) for a, b in itertools.combinations(ps, 2))


# --------------------------------------------------------------------------------------
# 2. State preparation circuit U_S
# --------------------------------------------------------------------------------------
def conditional_probs(dag: PathDAG, mode="uniform", arc_length=None, tau=0.0):
    """P(arc e | flow is at tail(e)). Product along a path = probability of that path."""
    m = dag.n_qubits
    if mode == "walk":
        return [1.0 / len(dag.out_arcs[dag.arcs[e][0]]) for e in range(m)]
    if mode == "uniform":
        w = [1.0] * m
    elif mode == "gibbs":
        w = [math.exp(-tau * arc_length[e]) for e in range(m)]
    else:
        raise ValueError(mode)
    z = dag.partition(w)
    return [w[e] * z[dag.arcs[e][1]] / z[dag.arcs[e][0]] for e in range(m)]


def state_prep_gates(dag: PathDAG, mode="uniform", arc_length=None, tau=0.0):
    """Gate list for U_S. Gates: ("x", q), ("cx", c, t), ("cry", theta, c, t).

    Node by node in topological order:
      * the 'token' (flow indicator of node v) is written onto v's first out-arc with one CX
        per in-arc (at most one in-arc is 1, so XOR = OR);
      * a W-state cascade then moves the token along the out-arcs: CRY(theta) puts part of the
        amplitude on the next out-arc, CX clears the previous one.
    No ancilla qubits. CX-equivalent count: 4|E| - indeg(t) - 3|V| + 3   (CRY = 2 CX).
    """
    p = conditional_probs(dag, mode, arc_length, tau)
    gates = []
    for v in dag.topo:
        outs = dag.out_arcs[v]
        if v == dag.t or not outs:
            continue
        if v == dag.s:
            gates.append(("x", outs[0]))
        else:
            for e in dag.in_arcs[v]:
                gates.append(("cx", e, outs[0]))
        remaining = 1.0
        for j in range(len(outs) - 1):
            stay = min(1.0, max(0.0, p[outs[j]] / remaining))
            theta = 2.0 * math.acos(math.sqrt(stay))
            gates.append(("cry", theta, outs[j], outs[j + 1]))
            gates.append(("cx", outs[j + 1], outs[j]))
            remaining -= p[outs[j]]
    return gates


def inverse(gates):
    inv = []
    for g in reversed(gates):
        if g[0] == "cry":
            inv.append(("cry", -g[1], g[2], g[3]))
        elif g[0] in ("x", "cx"):
            inv.append(g)
        elif g[0] == "p":
            inv.append(("p", -g[1], g[2]))
        elif g[0] == "refl0":
            inv.append(("refl0", -g[1]))
        else:
            raise ValueError(g)
    return inv


def cx_count(gates, n_qubits=None, mcp_cx_per_qubit=8):
    """Logical CX-equivalent count. CRY = 2 CX. The n-qubit reflection (multi-controlled phase)
    is costed at mcp_cx_per_qubit * n as a planning number only; replace with transpiler output."""
    c = 0
    for g in gates:
        if g[0] == "cx":
            c += 1
        elif g[0] == "cry":
            c += 2
        elif g[0] == "refl0":
            c += mcp_cx_per_qubit * n_qubits
    return c


def gm_qaoa_gates(dag: PathDAG, gammas_cost, gammas_delay, betas, prep=None):
    """Full GM-QAOA circuit as a gate list.
    Phase separator: ("p", angle, q) = diag(1, exp(-i*angle)) on every arc qubit
                     with angle = gamma_c*cost[e] + gamma_d*delay[e]  (single-qubit gates only).
    Mixer:           U_S^dagger, ("refl0", beta) = phase exp(-i*beta) on |0...0>, U_S.
    """
    prep = prep if prep is not None else state_prep_gates(dag, "uniform")
    gates = list(prep)
    for gc, gd, b in zip(gammas_cost, gammas_delay, betas):
        for e in range(dag.n_qubits):
            gates.append(("p", gc * dag.cost[e] + gd * dag.delay[e], e))
        gates += inverse(prep)
        gates.append(("refl0", b))
        gates += prep
    return gates


# --------------------------------------------------------------------------------------
# 3. Gate-level simulator
# --------------------------------------------------------------------------------------
class StateVector:
    def __init__(self, n):
        self.n = n
        self.psi = np.zeros((2,) * n, dtype=complex)
        self.psi[(0,) * n] = 1.0

    def _apply_1q(self, u, q, sub=None, axis=None):
        arr = self.psi if sub is None else sub
        ax = q if axis is None else axis
        moved = np.moveaxis(arr, ax, 0)
        new = np.tensordot(u, moved, axes=([1], [0]))
        moved[...] = new

    def run(self, gates):
        n = self.n
        for g in gates:
            kind = g[0]
            if kind == "x":
                self._apply_1q(np.array([[0, 1], [1, 0]], dtype=complex), g[1])
            elif kind == "p":
                self._apply_1q(np.diag([1.0, np.exp(-1j * g[1])]), g[2])
            elif kind in ("cx", "cry"):
                if kind == "cx":
                    c, t = g[1], g[2]
                    u = np.array([[0, 1], [1, 0]], dtype=complex)
                else:
                    th, c, t = g[1], g[2], g[3]
                    u = np.array([[math.cos(th / 2), -math.sin(th / 2)],
                                  [math.sin(th / 2), math.cos(th / 2)]], dtype=complex)
                idx = [slice(None)] * n
                idx[c] = 1
                sub = self.psi[tuple(idx)]
                self._apply_1q(u, t, sub=sub, axis=t if t < c else t - 1)
            elif kind == "refl0":
                self.psi[(0,) * n] *= np.exp(-1j * g[1])
            else:
                raise ValueError(g)
        return self

    def probabilities(self, tol=1e-12):
        """dict bitstring(q_{n-1}..q_0) -> probability."""
        out = {}
        p = np.abs(self.psi) ** 2
        for idx in zip(*np.nonzero(p > tol)):
            out["".join(str(b) for b in reversed(idx))] = float(p[idx])
        return out


# --------------------------------------------------------------------------------------
# 4. Path-subspace simulator (GM-QAOA never leaves the span of valid paths)
# --------------------------------------------------------------------------------------
class SubspaceGMQAOA:
    def __init__(self, dag: PathDAG, prep_probs=None):
        self.dag = dag
        self.paths = dag.enumerate_paths()
        self.c = np.array([dag.path_metric(p, dag.cost) for p in self.paths], dtype=float)
        self.d = np.array([dag.path_metric(p, dag.delay) for p in self.paths], dtype=float)
        f = len(self.paths)
        probs = np.full(f, 1.0 / f) if prep_probs is None else np.asarray(prep_probs, float)
        self.f = np.sqrt(probs).astype(complex)

    def state(self, gammas_cost, gammas_delay, betas):
        a = self.f.copy()
        for gc, gd, b in zip(gammas_cost, gammas_delay, betas):
            a = a * np.exp(-1j * (gc * self.c + gd * self.d))
            a = a - (1 - np.exp(-1j * b)) * self.f * np.vdot(self.f, a)
        return a

    def threshold_state(self, good_mask, rounds):
        """Exact amplitude amplification with a (cost<=c*, delay<=D) phase oracle."""
        a = self.f.copy()
        for _ in range(rounds):
            a = np.where(good_mask, -a, a)
            a = 2 * self.f * np.vdot(self.f, a) - a
        return a


# --------------------------------------------------------------------------------------
# 5. Classical baselines
# --------------------------------------------------------------------------------------
def exact_csp(dag: PathDAG, delay_bound):
    """Brute force: (best_cost, [indices of optimal feasible paths], paths, costs, delays)."""
    paths = dag.enumerate_paths()
    c = [dag.path_metric(p, dag.cost) for p in paths]
    d = [dag.path_metric(p, dag.delay) for p in paths]
    feas = [i for i in range(len(paths)) if d[i] <= delay_bound]
    if not feas:
        return None, [], paths, c, d
    best = min(c[i] for i in feas)
    return best, [i for i in feas if c[i] == best], paths, c, d


def larac_lambda(costs, delays, delay_bound):
    """LARAC multiplier (on the enumerated path list; a real run would use Dijkstra/DP)."""
    idx = range(len(costs))
    pc = min(idx, key=lambda i: (costs[i], delays[i]))
    if delays[pc] <= delay_bound:
        return 0.0
    pd = min(idx, key=lambda i: (delays[i], costs[i]))
    if delays[pd] > delay_bound:
        return None
    while True:
        lam = (costs[pc] - costs[pd]) / (delays[pd] - delays[pc])
        r = min(idx, key=lambda i: costs[i] + lam * delays[i])
        if abs((costs[r] + lam * delays[r]) - (costs[pc] + lam * delays[pc])) < 1e-9:
            return lam
        if delays[r] <= delay_bound:
            pd = r
        else:
            pc = r


def gibbs_probs(costs, delays, lam, tau):
    L = np.array(costs, float) + lam * np.array(delays, float)
    w = np.exp(-tau * (L - L.min()))
    return w / w.sum()


# --------------------------------------------------------------------------------------
# 5b. Classical simulation of linear-phase GM-QAOA at any depth ("dequantised" decoder)
# --------------------------------------------------------------------------------------
def char_fn(dag: PathDAG, prep_cond, g_cost, g_delay):
    """phi(G) = sum over paths of pi(path) * exp(-i (Gc*cost + Gd*delay)).
    One complex-weight DP over the DAG, O(|E|), no path enumeration, any real weights."""
    w = [prep_cond[e] * complex(math.cos(g_cost * dag.cost[e] + g_delay * dag.delay[e]),
                                -math.sin(g_cost * dag.cost[e] + g_delay * dag.delay[e]))
         for e in range(dag.n_qubits)]
    return dag.partition(w)[dag.s]


def dequantized_terms(dag: PathDAG, gammas_cost, gammas_delay, betas, prep_cond=None):
    """After p layers the state is  sum_j alpha_j * U_P(G_j) |F>  with only p+1 terms.
    Returns [(alpha_j, Gc_j, Gd_j)]. Cost: O(p^2 |E|) classical operations."""
    prep_cond = prep_cond if prep_cond is not None else conditional_probs(dag, "uniform")
    terms = [(1.0 + 0j, 0.0, 0.0)]
    for gc, gd, b in zip(gammas_cost, gammas_delay, betas):
        terms = [(a, x + gc, y + gd) for a, x, y in terms]
        overlap = sum(a * char_fn(dag, prep_cond, x, y) for a, x, y in terms)
        terms.append((-(1 - complex(math.cos(b), -math.sin(b))) * overlap, 0.0, 0.0))
    return terms


def dequantized_ratio(terms, cost, delay):
    """P_QAOA(path) / pi(path) for a path with the given total cost and delay."""
    return abs(sum(a * complex(math.cos(x * cost + y * delay), -math.sin(x * cost + y * delay))
                   for a, x, y in terms)) ** 2


def dequantized_sample(dag: PathDAG, terms, n_samples, rng, prep_cond=None):
    """Exact classical sampler for the GM-QAOA output: draw a path from the preparation
    distribution (a random walk), accept with probability ratio / (sum |alpha_j|)^2.
    Returns (list of accepted paths, number of proposals used)."""
    prep_cond = prep_cond if prep_cond is not None else conditional_probs(dag, "uniform")
    bound = sum(abs(a) for a, _, _ in terms) ** 2
    out, proposals = [], 0
    while len(out) < n_samples:
        proposals += 1
        v, path = dag.s, []
        while v != dag.t:
            outs = dag.out_arcs[v]
            e = outs[rng.choice(len(outs), p=[prep_cond[o] for o in outs])]
            path.append(e)
            v = dag.arcs[e][1]
        c = dag.path_metric(path, dag.cost)
        d = dag.path_metric(path, dag.delay)
        if rng.random() < dequantized_ratio(terms, c, d) / bound:
            out.append(tuple(path))
    return out, proposals


# --------------------------------------------------------------------------------------
# 6. Qiskit adapter  (NOT executed in the session that wrote this file)
# --------------------------------------------------------------------------------------
def to_qiskit(gates, n_qubits, measure=True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for g in gates:
        if g[0] == "x":
            qc.x(g[1])
        elif g[0] == "cx":
            qc.cx(g[1], g[2])
        elif g[0] == "cry":
            qc.cry(g[1], g[2], g[3])
        elif g[0] == "p":
            qc.p(-g[1], g[2])                       # diag(1, exp(-i*angle))
        elif g[0] == "refl0":                       # phase exp(-i*beta) on |0...0>
            qc.x(range(n_qubits))
            qc.mcp(-g[1], list(range(n_qubits - 1)), n_qubits - 1)
            qc.x(range(n_qubits))
        else:
            raise ValueError(g)
    if measure:
        qc.measure_all()
    return qc


def postselect(dag: PathDAG, counts):
    """counts: dict bitstring -> shots (qiskit order). Returns (valid_fraction, path_counts)."""
    total = sum(counts.values())
    valid = {}
    for b, k in counts.items():
        p = dag.decode(b.replace(" ", ""))
        if p is not None:
            valid[p] = valid.get(p, 0) + k
    return sum(valid.values()) / total, valid


# --------------------------------------------------------------------------------------
# 7. Angle search and scoring of measured counts
# --------------------------------------------------------------------------------------
def best_angles(sim: "SubspaceGMQAOA", good, p, rng, restarts=60):
    """Angles (gamma_cost[0..p), gamma_delay[0..p), beta[0..p)) that maximise P(good).
    Tuned with knowledge of the answer: a best case for the decoder, used to fix the
    hardware circuits. Returns (probability, angle vector)."""
    from scipy.optimize import minimize

    def neg(x):
        a = sim.state(x[:p], x[p:2 * p], x[2 * p:])
        return -float(np.sum(np.abs(a[good]) ** 2))

    best = (0.0, None)
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(-1.5, 1.5, 2 * p), rng.uniform(0, 2 * math.pi, p)])
        res = minimize(neg, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    return best


def score_counts(dag: PathDAG, counts, ideal=None, optimum=None):
    """Metrics for one measured circuit.
    counts : dict bitstring (qiskit order) -> shots
    ideal  : dict bitstring -> ideal probability (valid strings only), optional
    optimum: bitstring of the constrained optimum, optional
    Returns dict with
      valid_fraction     V  = shots that pass every flow-conservation check / shots
      tvd_postselected       = total variation distance between the valid shots and `ideal`
      tvd_raw                = same, without discarding invalid shots
      p_opt_given_valid      = share of valid shots equal to `optimum`
      success_per_shot   S  = shots equal to `optimum` / all shots
    """
    counts = {b.replace(" ", ""): k for b, k in counts.items()}
    total = sum(counts.values())
    valid = {b: k for b, k in counts.items() if dag.decode(b) is not None}
    nv = sum(valid.values())
    out = {"shots": total, "valid_shots": nv, "valid_fraction": nv / total}
    if ideal is not None:
        keys = set(ideal) | set(valid)
        out["tvd_postselected"] = 0.5 * sum(abs(valid.get(k, 0) / nv - ideal.get(k, 0)) for k in keys) if nv else None
        keys = set(ideal) | set(counts)
        out["tvd_raw"] = 0.5 * sum(abs(counts.get(k, 0) / total - ideal.get(k, 0)) for k in keys)
    if optimum is not None:
        out["p_opt_given_valid"] = valid.get(optimum, 0) / nv if nv else None
        out["success_per_shot"] = counts.get(optimum, 0) / total
    return out
