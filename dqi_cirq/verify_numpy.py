"""
verify_numpy.py -- every numpy check of this directory, printed as a PASS/FAIL table and
written to results.txt. Also (re)writes qasm/b1.qasm and qasm/b2.qasm with the own writer and
validates them by parsing back and re-simulating. Needs only numpy (and ../dqi_paths).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qudit_toy  # noqa: E402
from binary_toys import B1, B2, noisy_distribution, score_counts, simulate_numpy, tv  # noqa: E402
from ops import _depolarize2, apply_ops, gate_counts, marginal, op_matrix, parse_qasm  # noqa: E402

rows, notes = [], []


def row(name, val, ok):
    rows.append((name, val, ok))


def fmt(v):
    return f"{v:.3e}" if isinstance(v, float) and (abs(v) < 1e-3 and v != 0) else (f"{v:.6f}" if isinstance(v, float) else str(v))


# ---- A: qudit toy --------------------------------------------------------------------------
for r in qudit_toy.checks():
    row(*r)

# ---- channel check: closed-form depolariser vs explicit Pauli sum (3 qubits, random rho) ------
rng = np.random.default_rng(0)
G = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
rho = G @ G.conj().T; rho /= np.trace(rho)
paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
p = 0.07
expl = (1 - p) * rho
for a in range(4):
    for b in range(4):
        if a or b:
            Pm = np.kron(np.kron(paulis[a], np.eye(2)), paulis[b])          # acts on qubits 0 and 2
            expl = expl + p / 15 * Pm @ rho @ Pm.conj().T
mine = _depolarize2(rho.reshape((2,) * 6), 0, 2, p, 3).reshape(8, 8)
row("channel: depolarize2 vs explicit Pauli sum", float(np.abs(mine - expl).max()), np.abs(mine - expl).max() < 1e-14)

# ---- B: binary instances -------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "qasm"), exist_ok=True)
noise_lines = []
for inst in (B1(), B2()):
    ref = inst.reference()
    s = inst.scores()
    ops = inst.ops()
    u = max(np.abs(op_matrix(o).T @ op_matrix(o) - np.eye(2 ** len(o.targets))).max() for o in ops)
    row(f"{inst.name}: every gate unitary", float(u), u < 1e-12)
    amp, p_clean, _ = simulate_numpy(inst)
    row(f"{inst.name}: P(metric qubits{' + ancilla' if inst.anc else ''} all 0)", p_clean, abs(p_clean - 1) < 1e-12)
    da = float(np.abs(amp - ref["amp"]).max())
    row(f"{inst.name}: max |amplitude - direct definition| (signs included)", da, da < 1e-12)
    dist = np.abs(amp) ** 2
    dp = float(np.abs(dist - ref["prob"]).max())
    row(f"{inst.name}: max |P_circuit - P_direct|", dp, dp < 1e-12)
    st_nd = apply_ops(inst.ops(decode=False), [2] * inst.N)
    d_nd = marginal(np.abs(st_nd) ** 2, inst.path)
    du = float(np.abs(d_nd - 2.0 ** -inst.n).max())
    row(f"{inst.name}: no decode -> uniform, max |P - 2^-n|", du, du < 1e-12)
    row(f"{inst.name}: E[metrics] DQI / uniform / best",
        f"{(dist * s).sum():.4f} / {s.mean():.4f} / {s.max()} of {inst.m}", True)
    row(f"{inst.name}: P(best path) DQI / uniform", f"{dist[s == s.max()].sum():.4f} / {(s == s.max()).mean():.4f}", True)
    row(f"{inst.name}: weights w0, w1", f"{ref['w'][0]:.6f}, {ref['w'][1]:.6f}", True)
    # QASM: own writer -> file -> own parser -> re-simulate
    text = inst.qasm()
    fn = os.path.join(HERE, "qasm", f"{inst.name.lower()}.qasm")
    with open(fn, "w") as fh:
        fh.write(text)
    ops2, N2, meas = parse_qasm(open(fn).read())
    st2 = apply_ops(ops2, [2] * N2)
    d2 = marginal(np.abs(st2) ** 2, meas)
    dq = float(np.abs(d2 - ref["prob"]).max())
    row(f"{inst.name}: qasm/{inst.name.lower()}.qasm parsed back, max |P - P_direct|", dq,
        dq < 1e-12 and len(ops2) == len(ops) and meas == inst.path)
    # score_counts on exact "counts" in both bit orders
    X = inst.paths()
    cnt_c = {"".join(map(str, x)): int(round(1e6 * dist[tuple(x)])) for x in X}
    cnt_q = {k[::-1]: v for k, v in cnt_c.items()}
    rc, rq = score_counts(inst, cnt_c, "cirq"), score_counts(inst, cnt_q, "qiskit")
    ok = rc["tv"] < 1e-5 and rq["tv"] < 1e-5 and abs(rc["mean_satisfied"] - (dist * s).sum()) < 1e-5
    row(f"{inst.name}: score_counts cirq/qiskit order, TV of rounded exact counts", f"{rc['tv']:.1e} / {rq['tv']:.1e}", ok)
    # noise
    gc = gate_counts(ops)
    notes.append(f"{inst.name}: {inst.N} qubits ({inst.m} metric, {inst.n} path{', 1 ancilla' if inst.anc else ''}); "
                 f"gates {gc}; v = {inst.v.tolist()}")
    d0 = noisy_distribution(inst, 0.0, 0.0)
    row(f"{inst.name}: density-matrix sim (no noise) vs state vector", float(np.abs(d0 - dist).max()),
        np.abs(d0 - dist).max() < 1e-12)
    labels = ["".join(map(str, x)) for x in X]
    noise_lines.append(f"{inst.name}  ideal      " + "  ".join(f"{l}:{dist[tuple(x)]:.4f}" for l, x in zip(labels, X))
                       + f"   mean metrics {(dist * s).sum():.4f}")
    for p2, pro in [(0.003, 0.01), (0.01, 0.02)]:
        dn = noisy_distribution(inst, p2, pro)
        noise_lines.append(f"{inst.name}  p2={p2:<5} p_ro={pro:<4} " + "  ".join(f"{l}:{dn[tuple(x)]:.4f}" for l, x in zip(labels, X))
                           + f"   TV {tv(dn, dist):.4f}   mean metrics {(dn * s).sum():.4f}")

# ---- report --------------------------------------------------------------------------------
out = io.StringIO()
w = max(len(r[0]) for r in rows)
print("numpy verification (Cirq not used)\n", file=out)
for name, val, ok in rows:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {fmt(val)}", file=out)
print(f"\n{sum(r[2] for r in rows)}/{len(rows)} checks passed\n", file=out)
for n_ in notes:
    print(n_, file=out)
print("\nnoise prediction (exact density matrix; 2-qubit depolarising p2 after every cx, six per ccx;"
      " readout bit-flip p_ro; bitstring character j = stage j)", file=out)
for l_ in noise_lines:
    print(l_, file=out)
text = out.getvalue()
print(text, end="")
with open(os.path.join(HERE, "results.txt"), "w") as fh:
    fh.write(text)
sys.exit(0 if all(r[2] for r in rows) else 1)
