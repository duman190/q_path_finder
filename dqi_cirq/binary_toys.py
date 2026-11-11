"""
binary_toys.py -- DQI (l = 1) on two binary path problems, as qubit circuits for hardware.

Path problem with p = 2: n stages, two ports per stage, a path is a bit string x = (x_0..x_{n-1}).
Metric i has a 0/1 weight row b_i; its total is the parity T_i(x) = b_i . x mod 2, and it is
satisfied when T_i(x) == v_i. With f_i(t) = +1 if t == v_i else -1 (= (-1)^(t + v_i)), DQI with
l = 1 targets the amplitude  w0 * P0 + w1 * P1(x),  P0 = 2^(-n/2),  P1(x) = sum_i f_i(b_i.x) / sqrt(m 2^n).

Qubit layout (big-endian, qubit 0 most significant, as in Cirq):
  0..m-1      metric qubits e_i
  m..m+n-1    path qubits; qubit m+j is stage j (holds syndrome s_j, then the port x_j)
  m+n         ancilla (only when n = 3)

Circuit (gate list, single source of truth; gates h, x, z, ry, cx, ccx only):
  1. w0|0^m> + w1|W_m>: ry(2 atan2(w1, w0)) on e_0, then for k = 0..m-2 a controlled-RY
     (decomposed ry, cx, ry, cx) from e_k to e_{k+1} and cx(e_{k+1} -> e_k);
  2. z on e_i where v_i = 1   (ghat_i(1) = (-1)^{v_i}; derivation in README);
  3. syndrome s = B^T e: cx(e_i -> s_j) where B[i, j] = 1;
  4. decode-and-erase: x on e_i when s == b_i  (ccx with negated controls; for n = 3 the first
     two syndrome bits are combined into the ancilla, re-used between rows);
  5. h on every path qubit; measure the path qubits (key "path", stage j = j-th bit).
"""
from __future__ import annotations

import itertools

import numpy as np

from ops import (Op, apply_ops, apply_ops_density, ccx_pairs, gate_counts, marginal, parse_qasm,
                 readout_flip, to_cirq, write_qasm)


class BinaryInstance:
    def __init__(self, name, B, v):
        self.name = name
        self.B = np.array(B, int) % 2
        self.v = np.array(v, int) % 2
        self.m, self.n = self.B.shape
        assert len({tuple(r) for r in self.B}) == self.m and self.B.any(1).all(), \
            "l = 1 decoding needs distinct nonzero rows"
        self.anc = self.n == 3
        self.N = self.m + self.n + int(self.anc)
        self.metric = list(range(self.m))
        self.path = [self.m + j for j in range(self.n)]

    # classical side
    def paths(self):
        return np.array(list(itertools.product([0, 1], repeat=self.n)))      # row index = big-endian x

    def scores(self):
        """Number of satisfied metrics per path, shape (2,)*n indexed [x0, ..., x_{n-1}]."""
        X = self.paths()
        return ((X @ self.B.T) % 2 == self.v).sum(1).reshape((2,) * self.n)

    # direct definition of the DQI state
    def reference(self):
        X = self.paths()
        F = np.where((X @ self.B.T) % 2 == self.v, 1.0, -1.0)                # f_i(b_i . x)
        P0 = np.full(len(X), 2 ** (-self.n / 2))
        P1 = F.sum(1); P1 = P1 / np.linalg.norm(P1)
        s = self.scores().ravel().astype(float)
        sec = [P0, P1]
        gram = np.array([[a @ b for b in sec] for a in sec])
        S = np.array([[a @ (s * b) for b in sec] for a in sec])
        ev, vec = np.linalg.eigh(S)
        w = vec[:, -1] * np.sign(vec[0, -1] or 1.0)                           # w0 >= 0
        amp = w[0] * P0 + w[1] * P1
        return dict(w=w, amp=amp.reshape((2,) * self.n), prob=(amp ** 2).reshape((2,) * self.n),
                    gram_err=float(np.abs(gram - np.eye(2)).max()), top_eig=float(ev[-1]))

    # circuit
    def ops(self, decode=True):
        m, n, B = self.m, self.n, self.B
        e, s = self.metric, self.path
        w = self.reference()["w"]
        L = [Op("ry", (e[0],), param=2 * np.arctan2(w[1], w[0]))]
        for k in range(m - 1):
            a = 2 * np.arccos(1 / np.sqrt(m - k))
            L += [Op("ry", (e[k + 1],), param=a / 2), Op("cx", (e[k], e[k + 1])),
                  Op("ry", (e[k + 1],), param=-a / 2), Op("cx", (e[k], e[k + 1])),
                  Op("cx", (e[k + 1], e[k]))]
        L += [Op("z", (e[i],)) for i in range(m) if self.v[i]]
        L += [Op("cx", (e[i], s[j])) for i in range(m) for j in range(n) if B[i, j]]
        if decode:
            L += self._decoder()
        L += [Op("h", (q,)) for q in s]
        return L

    def _decoder(self):
        e, s, B = self.metric, self.path, self.B
        L = []
        if self.n == 2:
            for i, row in enumerate(B):
                neg = [Op("x", (s[j],)) for j in range(2) if row[j] == 0]
                L += neg + [Op("ccx", (s[0], s[1], e[i]))] + neg
            return L
        if self.n != 3:
            raise NotImplementedError("decoder written for n = 2 or 3 stages")
        a = self.N - 1
        # anc walks through the four prefix functions s0 s1, s0 ~s1, ~s0 s1, ~s0 ~s1 using cheap cx/x.
        steps = {(1, 1): [Op("ccx", (s[0], s[1], a))],
                 (1, 0): [Op("cx", (s[0], a))],
                 (0, 1): [Op("cx", (s[1], a)), Op("cx", (s[0], a))],
                 (0, 0): [Op("x", (a,)), Op("cx", (s[0], a))]}
        for prefix, step in steps.items():
            L += step
            for i, row in enumerate(B):
                if tuple(row[:2]) == prefix:
                    neg = [Op("x", (s[2],))] if row[2] == 0 else []
                    L += neg + [Op("ccx", (a, s[2], e[i]))] + neg
        L += [Op("x", (a,)), Op("cx", (s[0], a)), Op("cx", (s[1], a)), Op("ccx", (s[0], s[1], a))]
        return L

    def qasm(self):
        note = (f"DQI l=1, instance {self.name}: m={self.m} metrics, n={self.n} stages, v={self.v.tolist()}\n"
                f"qubits 0..{self.m-1} metric, {self.m}..{self.m+self.n-1} path (stage j = q[{self.m}+j])"
                + (f", {self.N-1} ancilla" if self.anc else "") + "\n"
                f"c[j] = port of stage j; Qiskit count strings print c[{self.n-1}] first (stage j = char n-1-j)")
        return write_qasm(self.ops(), self.N, self.path, comment=note)


def B1():
    return BinaryInstance("B1", [[1, 0], [0, 1], [1, 1]], [1, 0, 0])


def B2(seed=5):
    """Hamming-dual instance. v from default_rng(seed); seed 5 gives v = (1,1,0,1,0,1,1), whose best
    path satisfies 6 of 7 (seeds whose v is the complement of a codeword give the flat case: every path but one scores 4)."""
    rows = [[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(1, 8)]     # all nonzero 3-bit vectors
    rng = np.random.default_rng(seed)
    while True:
        v = rng.integers(0, 2, size=7)
        inst = BinaryInstance("B2", rows, v)
        if inst.scores().max() < 7:
            return inst


def simulate_numpy(inst, decode=True):
    """Ideal gate-level run. Returns (path amplitudes (2,)*n, P(metric+ancilla all 0), full state)."""
    st = apply_ops(inst.ops(decode), [2] * inst.N)
    junk = tuple(range(inst.m)) + ((inst.N - 1,) if inst.anc else ())
    pr = np.abs(st) ** 2
    p_clean = float(marginal(pr, list(junk))[(0,) * len(junk)])
    idx = tuple(0 if q in junk else slice(None) for q in range(inst.N))
    return st[idx], p_clean, st


def noisy_distribution(inst, p2, p_ro):
    """Exact density-matrix prediction (numpy) of the measured path distribution."""
    rho = apply_ops_density(inst.ops(), inst.N, p2)
    diag = np.einsum(f"{'abcdefghijk'[:inst.N]}{'abcdefghijk'[:inst.N]}->{'abcdefghijk'[:inst.N]}", rho)
    return readout_flip(marginal(diag, inst.path), p_ro)


def tv(P, Q):
    return 0.5 * float(np.abs(np.asarray(P) - np.asarray(Q)).sum())


def score_counts(inst, counts, bit_order="cirq"):
    """Score measured counts of the path qubits.

    counts: dict bitstring -> count. bit_order:
      "cirq"   : character j is stage j  (bits in the order of cirq.measure(*path_qubits));
      "qiskit" : Qiskit/IBM string, classical bit c[n-1] printed first, so stage j = character n-1-j
                 (spaces between registers are ignored; there is only one register here).
    Returns dict(dist (2,)*n indexed [x0..x_{n-1}], tv from ideal, mean metrics satisfied, shots)."""
    D = np.zeros((2,) * inst.n)
    for key, c in counts.items():
        b = str(key).replace(" ", "")
        assert len(b) == inst.n, f"bitstring {key!r} has wrong length"
        if bit_order == "qiskit":
            b = b[::-1]
        D[tuple(int(ch) for ch in b)] += c
    shots = D.sum()
    D = D / shots
    return dict(dist=D, tv=tv(D, inst.reference()["prob"]),
                mean_satisfied=float((D * inst.scores()).sum()), shots=int(shots))


# ---- Cirq side (NOT executed here: Cirq is not installed in the authoring environment) -------
def build_binary_circuit(inst, measure=True):
    """cirq.Circuit on cirq.LineQubit(0..N-1) from the same gate list; path qubits measured
    under key "path" in stage order (result bit j = stage j)."""
    circuit, _ = to_cirq(inst.ops(), [2] * inst.N, measure=inst.path if measure else None, key="path")
    return circuit


def _counts_from_result(result, key="path"):
    counts = {}
    for row in result.measurements[key]:
        b = "".join(str(int(x)) for x in row)
        counts[b] = counts.get(b, 0) + 1
    return counts


def simulate_ideal(circuit, repetitions=10000):
    import cirq
    return _counts_from_result(cirq.Simulator().run(circuit, repetitions=repetitions))


def noisy_circuit(circuit, p2, p_ro):
    """Same noise model as noisy_distribution: depolarize(p2, 2) after every 2-qubit gate, six of
    them (on the Toffoli-decomposition pairs) after every CCX, bit_flip(p_ro) before measurement."""
    import cirq
    out = cirq.Circuit()
    for op in circuit.all_operations():
        if cirq.is_measurement(op):
            out.append(cirq.bit_flip(p_ro).on_each(*op.qubits))
            out.append(op)
            continue
        out.append(op)
        if isinstance(op.gate, cirq.CCXPowGate):
            for a, b in ccx_pairs(*op.qubits):
                out.append(cirq.depolarize(p2, n_qubits=2).on(a, b))
        elif len(op.qubits) == 2:
            out.append(cirq.depolarize(p2, n_qubits=2).on(*op.qubits))
    return out


def simulate_noisy(circuit, p2, p_ro, repetitions=10000):
    import cirq
    sim = cirq.DensityMatrixSimulator(dtype=np.complex128)
    return _counts_from_result(sim.run(noisy_circuit(circuit, p2, p_ro), repetitions=repetitions))


def export_qasm(circuit, path):
    text = circuit.to_qasm()
    with open(path, "w") as fh:
        fh.write(text)
    return text


def _cirq_selftest():
    import cirq
    for inst in (B1(), B2()):
        ref = inst.reference()["prob"]
        c = build_binary_circuit(inst, measure=False)
        qs = cirq.LineQubit.range(inst.N)
        psi = np.asarray(cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=qs).final_state_vector)
        d = marginal(np.abs(psi.reshape((2,) * inst.N)) ** 2, inst.path)
        print(f"{'PASS' if np.abs(d - ref).max() < 1e-9 else 'FAIL'}  {inst.name} Cirq simulate vs numpy: "
              f"{np.abs(d - ref).max():.2e}")
        cm = build_binary_circuit(inst)
        r = score_counts(inst, simulate_ideal(cm, 20000))
        print(f"{'PASS' if r['tv'] < 0.03 else 'FAIL'}  {inst.name} Cirq ideal 20000 shots: TV {r['tv']:.4f}")
        for p2, pro in [(0.003, 0.01), (0.01, 0.02)]:
            pred = noisy_distribution(inst, p2, pro)
            r = score_counts(inst, simulate_noisy(cm, p2, pro, 20000))
            d = tv(r["dist"], pred)
            print(f"{'PASS' if d < 0.03 else 'FAIL'}  {inst.name} Cirq noisy p2={p2} p_ro={pro}: "
                  f"TV(Cirq samples, numpy prediction) {d:.4f}")
        import os
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm", f"{inst.name.lower()}_from_cirq.qasm")
        export_qasm(cm, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    for inst in (B1(), B2()):
        amp, pc, _ = simulate_numpy(inst)
        print(inst.name, "v =", inst.v, "P(clean) =", pc, "max |amp - ref| =",
              np.abs(amp - inst.reference()["amp"]).max(), gate_counts(inst.ops()))
    try:
        import cirq  # noqa: F401
    except ImportError:
        print("Cirq not installed: Cirq self-test skipped.")
    else:
        _cirq_selftest()
