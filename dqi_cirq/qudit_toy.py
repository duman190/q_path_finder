"""
qudit_toy.py -- DQI on the 5-port toy, Instance(5, 2, 2, seed=3) of ../dqi_paths/dqi_paths.py.

Path problem: a fabric of n = 2 stages with p = 5 ports each (25 paths x = (x0, x1)); m = 4
metrics y = 1..4 with weight row B[y] = (1, y), so metric y's total is T_y(x) = x0 + y*x1 mod 5.
Each metric allows two totals. DQI with l = 1 samples paths that satisfy many metrics.

Registers (all 5-level qudits, big-endian in this order):
  0..3 : metric register e_1..e_4 (digit i holds the "error" on metric y = i+1)
  4, 5 : path register s0, s1 -- holds the syndrome, after the Fourier step it holds the path x.

Circuit (op list, single source of truth):
  1. one 625x625 unitary on qudits 0..3 preparing  w0|0000> + w1 * 1/2 * sum_i sum_u ghat_i(u) |u at i>
     (Householder reflection completing the target vector);
  2. syndrome: for each metric i and stage j, |e_i>|s_j> -> |e_i>|s_j + B[i,j] e_i>  (25x25 perm);
  3. decode-and-erase: for each metric y, on (s0, s1, e_y): if s0 != 0 and s1 == y*s0 then
     e_y -> e_y - s0 (125x125 perm);
  4. F[x, s] = exp(-2 pi i s x / 5)/sqrt(5) on qudits 4 and 5;
  5. measure qudits 4, 5 -> (x0, x1).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dqi_paths"))
from dqi_paths import Instance  # noqa: E402

from ops import Op, apply_ops, is_unitary, marginal, to_cirq  # noqa: E402

P, NST, M = 5, 2, 4
DIMS = [P] * (M + NST)
METRIC = list(range(M))
PATH = [M + j for j in range(NST)]


def instance():
    return Instance(5, 2, 2, seed=3)


def weights(inst):
    """Reference weights dqi()['w'], with the global phase fixed so that w0 is real and >= 0."""
    w = np.asarray(inst.dqi()["w"], complex)
    ph = w[0] / abs(w[0]) if abs(w[0]) > 0 else 1.0
    return w / ph


def target_metric_state(inst):
    w = weights(inst)
    gh = inst.g_hat()
    t = np.zeros((P,) * M, complex)
    t[(0,) * M] = w[0]
    for i in range(M):
        for u in range(1, P):
            idx = [0] * M; idx[i] = u
            t[tuple(idx)] += w[1] * gh[i, u] / np.sqrt(M)
    return t.ravel()


def householder_prep(t):
    """Unitary U with U|0> = t, for t with t[0] real >= 0 (U = I - 2 v v^dag / v^dag v, v = e0 - t)."""
    t = t / np.linalg.norm(t)
    assert abs(t[0].imag) < 1e-15 and t[0].real >= 0
    v = -t.copy(); v[0] += 1.0
    nv = np.vdot(v, v).real
    if nv < 1e-30:
        return np.eye(t.size, dtype=complex)
    return np.eye(t.size, dtype=complex) - 2 * np.outer(v, v.conj()) / nv


def add_gate(b, p=P):
    """|e>|s> -> |e>|s + b e mod p>, a p^2 x p^2 permutation (big-endian over (e, s))."""
    U = np.zeros((p * p, p * p))
    for e in range(p):
        for s in range(p):
            U[e * p + (s + b * e) % p, e * p + s] = 1
    return U


def decode_gate(y, p=P):
    """On (s0, s1, e_y): if s0 != 0 and s1 == y*s0 mod p then e_y -> e_y - s0 mod p."""
    U = np.zeros((p ** 3, p ** 3))
    for s0 in range(p):
        for s1 in range(p):
            for e in range(p):
                e2 = (e - s0) % p if (s0 != 0 and s1 == (y * s0) % p) else e
                U[(s0 * p + s1) * p + e2, (s0 * p + s1) * p + e] = 1
    return U


def fourier(p=P):
    k = np.arange(p)
    return np.exp(-2j * np.pi * np.outer(k, k) / p) / np.sqrt(p)


def build_ops(inst=None, decode=True):
    inst = inst or instance()
    ops = [Op("u", tuple(METRIC), matrix=householder_prep(target_metric_state(inst)))]
    for i in range(M):
        for j in range(NST):
            b = int(inst.B[i, j]) % P
            if b:
                ops.append(Op("u", (METRIC[i], PATH[j]), matrix=add_gate(b)))
    if decode:
        for i, y in enumerate(inst.ys):
            ops.append(Op("u", (PATH[0], PATH[1], METRIC[i]), matrix=decode_gate(int(y))))
    for j in range(NST):
        ops.append(Op("u", (PATH[j],), matrix=fourier()))
    return ops


def numpy_run(decode=True):
    """Returns (path distribution (5,5) indexed [x0, x1], P(metric register = 0000), final state)."""
    state = apply_ops(build_ops(decode=decode), DIMS)
    prob = np.abs(state) ** 2
    p_zero = float(prob[(0,) * M].sum())
    return marginal(prob, PATH), p_zero, state


def checks():
    """All numpy checks for the qudit toy; returns a list of (name, value, passed)."""
    inst = instance()
    ref = inst.dqi()
    ops = build_ops(inst)
    out = []
    unit_err = max(np.abs(op.matrix.conj().T @ op.matrix - np.eye(op.matrix.shape[0])).max() for op in ops)
    out.append(("qudit: every op unitary (max |U^dag U - I|)", unit_err, unit_err < 1e-12))
    perm_ok = all(((decode_gate(y) == 0) | (decode_gate(y) == 1)).all() and
                  (decode_gate(y).sum(0) == 1).all() and (decode_gate(y).sum(1) == 1).all() for y in range(1, P))
    out.append(("qudit: decode gates are permutations", float(perm_ok), perm_ok))
    U0 = ops[0].matrix
    prep_err = np.abs(U0[:, 0] - target_metric_state(inst)).max()
    out.append(("qudit: state-prep column 0 = target state", prep_err, prep_err < 1e-12))
    # metric register zero on the whole support right after decoding (before Fourier)
    st = apply_ops(ops[:-NST], DIMS)
    pr = np.abs(st) ** 2
    mask = np.ones(pr.shape, bool); mask[(0,) * M] = False
    leak = float(pr[mask].sum())
    out.append(("qudit: after decode, P(metric != 0000)", leak, leak < 1e-12))
    dist, p_zero, _ = numpy_run(True)
    out.append(("qudit: final P(metric = 0000)", p_zero, abs(p_zero - 1) < 1e-12))
    dev = float(np.abs(dist - ref["prob"]).max())
    out.append(("qudit: max |P_circuit - dqi()['prob']|", dev, dev < 1e-12))
    dist_nd, _, _ = numpy_run(False)
    dev_u = float(np.abs(dist_nd - 1 / 25).max())
    out.append(("qudit: no decode -> uniform, max |P - 1/25|", dev_u, dev_u < 1e-12))
    s = inst.scores()
    best = s.max()
    out.append(("qudit: P(best path) DQI circuit", float(dist[s == best].sum()), True))
    out.append(("qudit: number of best paths / best score", f"{int((s == best).sum())} / {int(best)}", True))
    out.append(("qudit: P(best path) uniform", float((s == best).mean()), True))
    out.append(("qudit: E[metrics] DQI circuit", float((dist * s).sum()), abs((dist * s).sum() - ref["expected"]) < 1e-12))
    out.append(("qudit: E[metrics] uniform random", float(s.mean()), True))
    return out


# ---- Cirq side (NOT executed here: Cirq is not installed in the authoring environment) -------
def build_qudit_circuit(measure=True):
    """cirq.Circuit on cirq.LineQid(i, dimension=5), i = 0..5, from the same op list.
    MatrixGate's matrix is big-endian over the qids in the order passed to .on(...)."""
    circuit, qids = to_cirq(build_ops(), DIMS, measure=PATH if measure else None, key="path")
    return circuit, qids


def simulate_qudit_with_cirq():
    """Path distribution (5,5) [x0, x1] from cirq.Simulator's final state vector."""
    import cirq
    circuit, qids = build_qudit_circuit(measure=False)
    res = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qids)
    psi = np.asarray(res.final_state_vector).reshape(DIMS)
    return marginal(np.abs(psi) ** 2, PATH)


def sample_qudit_with_cirq(repetitions=20000):
    import cirq
    circuit, _ = build_qudit_circuit(measure=True)
    res = cirq.Simulator().run(circuit, repetitions=repetitions)
    counts = np.zeros((P, P))
    for x0, x1 in res.measurements["path"]:
        counts[x0, x1] += 1
    return counts / repetitions


if __name__ == "__main__":
    for name, val, ok in checks():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {val}")
    try:
        import cirq  # noqa: F401
    except ImportError:
        print("Cirq not installed: Cirq self-test skipped.")
    else:
        ref = instance().dqi()["prob"]
        d = simulate_qudit_with_cirq()
        dev = np.abs(d - ref).max()
        print(f"{'PASS' if dev < 1e-9 else 'FAIL'}  Cirq simulate vs numpy reference: max dev {dev:.2e}")
        smp = sample_qudit_with_cirq()
        tv = 0.5 * np.abs(smp - ref).sum()
        print(f"{'PASS' if tv < 0.03 else 'FAIL'}  Cirq run (20000 shots) TV distance from reference: {tv:.4f}")
