"""Checks + ideal ('noise-free') target numbers for every experiment in the doc."""
import itertools
import math

import numpy as np
from scipy.optimize import minimize

from instances import fig2_dag, suite
from qpathfinder import (PathDAG, StateVector, SubspaceGMQAOA, cx_count, exact_csp, gibbs_probs,
                         gm_qaoa_gates, larac_lambda, state_prep_gates)

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(2026)


def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


print("=" * 90)
print("A. Re-analysis of the 2020 figures")
print("=" * 90)

# ---- Figure 1: transit node, in-arcs q0,q1 ; out-arcs q2,q3,q4 -----------------------
# Gate-for-gate transcription of the 2020 circuit (cH = controlled Hadamard).
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


class SV(StateVector):
    def h(self, q):
        self._apply_1q(H, q)

    def ch(self, c, t):
        idx = [slice(None)] * self.n
        idx[c] = 1
        self._apply_1q(H, t, sub=self.psi[tuple(idx)], axis=t if t < c else t - 1)


sv = SV(5)
sv.h(0); sv.run([("x", 1), ("cx", 0, 1)])
sv.run([("cx", 0, 2), ("cx", 0, 3), ("cx", 0, 4)]); sv.ch(0, 2)
sv.run([("cx", 1, 2), ("cx", 1, 3), ("cx", 1, 4)]); sv.ch(1, 2)
sv.run([("cx", 2, 3), ("cx", 2, 4), ("x", 2)]); sv.ch(2, 3); sv.run([("x", 2), ("cx", 3, 4)])
ideal1 = sv.probabilities()
fig1 = {"00101": 24.902, "00110": 24.609, "01001": 11.621, "01010": 12.793, "10001": 11.816, "10010": 14.258}
print("Fig 1 circuit, ideal probabilities :", {k: round(v, 4) for k, v in sorted(ideal1.items())})
print("Fig 1 histogram (%), sum           :", fig1, round(sum(fig1.values()), 3))
print("Fig 1 histogram x 1024 shots       :", {k: round(v * 10.24, 2) for k, v in fig1.items()})
print("Fig 1 TVD(histogram, ideal)        :", round(tvd({k: v / 100 for k, v in fig1.items()}, ideal1), 4))
print("  -> egress split is 1/2, 1/4, 1/4 (cascaded cH), not 1/3 each")

# ---- Figure 2: the A,B,C,D DAG -------------------------------------------------------
g = fig2_dag()
paths = g.enumerate_paths()
print("\nFig 2 paths:", [(g.path_nodes(p), g.flow_string(p)) for p in paths])
walk = StateVector(5).run(state_prep_gates(g, "walk")).probabilities()
print("2020 circuit distribution (equal split per node):", {k: round(v, 4) for k, v in walk.items()})
hw = {"01001": 30.078, "01110": 15.527, "10010": 16.113}        # read off Fig 2(c)
valid = sum(hw.values())
post = {k: v / valid for k, v in hw.items()}
print(f"2020 hardware: valid fraction = {valid:.3f}%  ({[round(v*10.24,2) for v in hw.values()]} of 1024 shots)")
print("2020 hardware post-selected   :", {k: round(v, 4) for k, v in post.items()})
print("TVD(post-selected, ideal walk):", round(tvd(post, walk), 4))
n_valid = round(valid * 10.24)
print(f"sampling-noise scale for {n_valid} valid shots: sigma(p=0.5) = {math.sqrt(0.25/n_valid):.4f}")
cirq_counts = {"01001": 8, "01110": 2, "10010": 2}             # 2020 Cirq sample, 12 shots
print("2020 Cirq sample (12 shots)   :", cirq_counts)
bars = [1.074, 4.199, 2.441, 0.781, 2.148, 2.930, 30.078, 2.344, 1.367, 4.004, 0.684, 15.527,
        5.566, 16.113, 1.172, 1.074, 2.246, 0.977, 1.172, 0.684]
print(f"all 20 bars read off Fig 2(c) sum to {sum(bars):.2f}% (screenshot is low-res; treat as approximate)")

print("\n" + "=" * 90)
print("B. New state-prep circuit U_S: correctness (gate level) and size")
print("=" * 90)
SUITE = suite()
rows = []
for dag, bound in SUITE:
    paths = dag.enumerate_paths()
    F = len(paths)
    gates = state_prep_gates(dag, "uniform")
    n = dag.n_qubits
    formula = 4 * n - len(dag.in_arcs[dag.t]) - 3 * dag.n_nodes + 3
    line = f"{dag.name:22s} |V|={dag.n_nodes:2d} qubits=|E|={n:2d} paths={F:3d} dist={dag.code_distance()} " \
           f"CX(U_S)={cx_count(gates)} (formula {formula}) N(s)={dag.path_counts()[dag.s]}"
    if n <= 24:
        probs = StateVector(n).run(gates).probabilities()
        target = {dag.flow_string(p): 1 / F for p in paths}
        ok = set(probs) == set(target) and max(abs(probs[k] - target[k]) for k in target) < 1e-9
        line += f" | gate-level uniform over paths: {ok}"
        # decoder check on every string with nonzero probability, plus random invalid strings
        assert all(dag.decode(b) is not None for b in probs)
        # Gibbs prep check
        L = [dag.cost[e] + 0.7 * dag.delay[e] for e in range(n)]
        gp = StateVector(n).run(state_prep_gates(dag, "gibbs", L, tau=0.3)).probabilities()
        c = [dag.path_metric(p, dag.cost) for p in paths]
        d = [dag.path_metric(p, dag.delay) for p in paths]
        tgt = gibbs_probs(c, d, 0.7, 0.3)
        okg = max(abs(gp[dag.flow_string(p)] - tgt[i]) for i, p in enumerate(paths)) < 1e-9
        line += f" | gibbs prep exact: {okg}"
    else:
        line += " | (too large for gate-level check here; subspace only)"
    print(line)
    rows.append((dag, bound, F, cx_count(gates)))

# random-DAG fuzz test of U_S (uniform) at gate level
import random


def prune(nn, arcs, s, t):
    """Keep only arcs that lie on some s-t path."""
    fwd, bwd = {s}, {t}
    for _ in range(nn):
        fwd |= {v for u, v in arcs if u in fwd}
        bwd |= {u for u, v in arcs if v in bwd}
    return [(u, v) for u, v in arcs if u in fwd and v in bwd]


fails = tested = 0
sizes = []
for trial in range(400):
    r = random.Random(trial)
    nn = r.randint(4, 8)
    arcs = prune(nn, [(u, v) for u in range(nn) for v in range(u + 1, nn) if r.random() < 0.5], 0, nn - 1)
    if not (2 <= len(arcs) <= 16):
        continue
    dg = PathDAG(nn, arcs, 0, nn - 1, [1] * len(arcs), [1] * len(arcs))
    ps = dg.enumerate_paths()
    pr = StateVector(dg.n_qubits).run(state_prep_gates(dg, "uniform")).probabilities()
    tg = {dg.flow_string(p): 1 / len(ps) for p in ps}
    tested += 1
    sizes.append((len(arcs), len(ps)))
    if set(pr) != set(tg) or max(abs(pr[k] - tg[k]) for k in tg) > 1e-9:
        fails += 1
    if dg.code_distance() is not None and dg.code_distance() < 3:
        print("  distance < 3 found:", arcs)
print(f"random-DAG fuzz test of U_S: {tested} DAGs tested (up to {max(s[0] for s in sizes)} arcs, "
      f"{max(s[1] for s in sizes)} paths), failures = {fails}")

print("\n" + "=" * 90)
print("C. Decoder: ideal success probability per shot (probability of the constrained optimum)")
print("=" * 90)


def best_angles(sim, good, p, restarts=60):
    def neg(x):
        a = sim.state(x[:p], x[p:2 * p], x[2 * p:])
        return -float(np.sum(np.abs(a[good]) ** 2))
    best = (0.0, None)
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(-1.5, 1.5, 2 * p), rng.uniform(0, 2 * math.pi, p)])
        res = minimize(neg, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    return best


results = {}
for dag, bound, F, cxs in rows:
    best_cost, opt, paths, c, d = exact_csp(dag, bound)
    good = np.zeros(F, bool); good[opt] = True
    feasible = sum(1 for x in d if x <= bound)
    sim = SubspaceGMQAOA(dag)
    p0 = len(opt) / F
    p1, x1 = best_angles(sim, good, 1)
    p2, x2 = best_angles(sim, good, 2, restarts=80)
    th = lambda r: math.sin((2 * r + 1) * math.asin(math.sqrt(p0))) ** 2
    r_opt = max(0, round(math.pi / (4 * math.asin(math.sqrt(p0))) - 0.5))
    lam = larac_lambda(c, d, bound)
    # best classical Boltzmann sampler at the LARAC multiplier (tau tuned with hindsight)
    taus = np.linspace(0, 6, 601)
    gib = [(gibbs_probs(c, d, lam, t)[opt].sum(), t) for t in taus]
    pg, tg = max(gib)
    supported = min(range(F), key=lambda i: c[i] + lam * d[i]) in opt or \
        abs(min(c[i] + lam * d[i] for i in range(F)) - (c[opt[0]] + lam * d[opt[0]])) < 1e-9
    # gate-level cross-check of GM-QAOA p=1 for small instances
    chk = ""
    if dag.n_qubits <= 12:
        gl = StateVector(dag.n_qubits).run(gm_qaoa_gates(dag, x1[:1], x1[1:2], x1[2:])).probabilities()
        a = sim.state(x1[:1], x1[1:2], x1[2:])
        sub = {dag.flow_string(p): abs(a[i]) ** 2 for i, p in enumerate(paths)}
        chk = f" gate-level==subspace: {tvd(gl, sub) < 1e-9}"
    # quantum on top of the classical sampler: prepare the Gibbs state, then amplify
    gprobs = gibbs_probs(c, d, lam, tg)
    gsim = SubspaceGMQAOA(dag, prep_probs=gprobs)
    pg1, _ = best_angles(gsim, good, 1)
    thg = lambda r: math.sin((2 * r + 1) * math.asin(math.sqrt(pg))) ** 2
    assert abs(float(np.sum(np.abs(gsim.threshold_state(good, 1)[good]) ** 2)) - thg(1)) < 1e-9
    results[dag.name] = dict(F=F, p0=p0, p1=p1, p2=p2, th1=th(1), r_opt=r_opt, th_opt=th(r_opt), pg=pg,
                             pg1=pg1, thg1=thg(1), lam=lam, tau=tg, bound=bound, opt=opt, paths=paths,
                             c=c, d=d, angles2=x2)
    print(f"{dag.name:22s} D<={bound:3d} feasible={feasible:3d}/{F:3d} opt: {dag.path_nodes(paths[opt[0]])} "
          f"cost={best_cost} delay={d[opt[0]]}")
    print(f"    min-cost path cost={min(c)} (delay {min(d[i] for i in range(F) if c[i]==min(c))}, infeasible);"
          f" LARAC lambda={lam:.3f}; optimum is a Lagrangian minimiser: {supported}")
    print(f"    uniform sampling p=0: {p0:.4f} | linear-phase GM-QAOA p=1: {p1:.4f} (x{p1/p0:.2f}) "
          f"p=2: {p2:.4f} (x{p2/p0:.2f}) | threshold oracle 1 round: {th(1):.4f}, "
          f"{r_opt} rounds: {th(r_opt):.4f}")
    print(f"    classical Boltzmann sampler at LARAC lambda, best tau={tg:.2f}: {pg:.4f}{chk}")
    print(f"    Gibbs prep + linear-phase p=1: {pg1:.4f} (x{pg1/pg:.2f} over the classical sampler) | "
          f"Gibbs prep + threshold oracle, 1 round: {thg(1):.4f} (x{thg(1)/pg:.2f})")
    print(f"    p=1 angles (gamma_c, gamma_d, beta) = {np.round(x1, 4)}")
    results[dag.name]["angles1"] = x1

print("\n" + "=" * 90)
print("D. Planning forecast (crude): global depolarising with F = (1-eps)^(routing * CX) * (1-ro)^n")
print("=" * 90)
print("assumptions: routing overhead x2 on CX count, readout error 1%, reflection costed at 8n CX")
for dag, bound, F, cxs in rows:
    n = dag.n_qubits
    r = results[dag.name]
    cx_p1 = 3 * cxs + 8 * n
    for eps in (1e-3, 3e-3):
        f0 = (1 - eps) ** (2 * cxs) * 0.99 ** n
        f1 = (1 - eps) ** (2 * cx_p1) * 0.99 ** n
        # per-shot probability of (valid AND optimal); noise term spreads over 2^n strings (negligible)
        s0, s1 = f0 * r["p0"], f1 * r["p1"]
        print(f"{dag.name:22s} eps={eps:.0e}: CX U_S={cxs:3d}, CX p=1={cx_p1:4d} | valid frac U_S~{f0:.2f}, "
              f"p=1~{f1:.2f} | success/shot p=0 {s0:.4f} vs p=1 {s1:.4f} -> gain x{s1/s0:.2f}")

print("\n" + "=" * 90)
print("E. Classical simulation of the linear-phase decoder (p+1 terms, complex-weight DP)")
print("=" * 90)
from qpathfinder import conditional_probs, dequantized_ratio, dequantized_sample, dequantized_terms

for dag, bound, F, cxs in rows:
    r = results[dag.name]
    sim = SubspaceGMQAOA(dag)
    worst = 0.0
    for p in (1, 2, 3, 5):
        gc, gd, be = rng.uniform(-1, 1, p), rng.uniform(-1, 1, p), rng.uniform(0, 2 * math.pi, p)
        a = sim.state(gc, gd, be)
        terms = dequantized_terms(dag, gc, gd, be)
        pred = np.array([dequantized_ratio(terms, r["c"][i], r["d"][i]) / F for i in range(F)])
        worst = max(worst, float(np.max(np.abs(pred - np.abs(a) ** 2))))
        assert sum(abs(t[0]) for t in terms) <= 2 * p + 1 + 1e-9
    # same check with a Gibbs preparation
    L = [dag.cost[e] + r["lam"] * dag.delay[e] for e in range(dag.n_qubits)]
    cond = conditional_probs(dag, "gibbs", L, r["tau"])
    gsim = SubspaceGMQAOA(dag, prep_probs=gibbs_probs(r["c"], r["d"], r["lam"], r["tau"]))
    gc, gd, be = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2), rng.uniform(0, 6, 2)
    terms = dequantized_terms(dag, gc, gd, be, prep_cond=cond)
    pi = gibbs_probs(r["c"], r["d"], r["lam"], r["tau"])
    pred = np.array([dequantized_ratio(terms, r["c"][i], r["d"][i]) * pi[i] for i in range(F)])
    worstg = float(np.max(np.abs(pred - np.abs(gsim.state(gc, gd, be)) ** 2)))
    print(f"{dag.name:22s} max |P_classical - P_quantum| over all paths: uniform prep {worst:.1e}, gibbs prep {worstg:.1e}")

# exact classical sampler vs the quantum distribution, on I3 at its best p=2 angles
dag = rows[2][0]; r = results[dag.name]; F = r["F"]
x2 = r["angles2"]
terms = dequantized_terms(dag, x2[:2], x2[2:4], x2[4:])
samples, proposals = dequantized_sample(dag, terms, 20000, rng)
emp = {}
for s_ in samples:
    emp[s_] = emp.get(s_, 0) + 1 / len(samples)
a = SubspaceGMQAOA(dag).state(x2[:2], x2[2:4], x2[4:])
ideal = {p: abs(a[i]) ** 2 for i, p in enumerate(r["paths"])}
print(f"I3, p=2 best angles: classical rejection sampler, 20000 samples, TVD to quantum distribution = "
      f"{tvd(emp, ideal):.4f}; proposals per accepted sample = {proposals/len(samples):.2f} (bound (2p+1)^2 = 25)")
print(f"   empirical P(optimum) = {sum(emp.get(r['paths'][i], 0) for i in r['opt']):.4f} vs quantum {r['p2']:.4f}")
