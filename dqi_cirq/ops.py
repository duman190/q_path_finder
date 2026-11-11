"""
ops.py -- one backend-neutral description of a circuit, and everything that consumes it.

A circuit is a list of Op(name, targets, matrix, param) acting on registers whose dimensions
are given by a list `dims` (dims[k] = 5 for a 5-port digit, 2 for a qubit).

* name in {"h", "x", "z", "ry", "cx", "ccx"}: a named qubit gate (ry takes `param` = angle);
* name == "u": an explicit unitary `matrix` on `targets` (any dimensions).

Index convention (the same as Cirq's): for an op on targets (t0, t1, ...), the matrix row/column
index is the big-endian ravel of the digits (d_t0, d_t1, ...), i.e. t0 is the MOST significant
digit. The full state vector is likewise big-endian in register order 0, 1, 2, ... (register 0
most significant), which is what cirq.Simulator returns for qubit_order = registers 0..N-1.
So one matrix serves both the numpy applier here and cirq.MatrixGate.

Consumers
---------
apply_ops         : state-vector simulation (numpy), the reference used by every check;
apply_ops_density : exact density-matrix simulation with optional 2-qubit depolarising noise;
to_cirq           : builds a cirq.Circuit from the same list (Cirq is imported inside the
                    function; it is NOT installed in the environment where this was written);
write_qasm / parse_qasm : an own OpenQASM 2 writer and a minimal parser for the six gates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

QUBIT_GATES = ("h", "x", "z", "ry", "cx", "ccx")


@dataclass
class Op:
    name: str
    targets: tuple
    matrix: np.ndarray | None = None
    param: float | None = None


def ry_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def op_matrix(op: Op) -> np.ndarray:
    if op.name == "u":
        return op.matrix
    if op.name == "h":
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    if op.name == "x":
        return np.array([[0, 1], [1, 0]], float)
    if op.name == "z":
        return np.diag([1.0, -1.0])
    if op.name == "ry":
        return ry_matrix(op.param)
    if op.name == "cx":                          # control = targets[0]
        M = np.eye(4); M[2:, 2:] = [[0, 1], [1, 0]]
        return M
    if op.name == "ccx":                         # controls = targets[0], targets[1]
        M = np.eye(8); M[6:, 6:] = [[0, 1], [1, 0]]
        return M
    raise ValueError(op.name)


def is_unitary(M, tol=1e-12):
    return np.abs(M.conj().T @ M - np.eye(M.shape[0])).max() < tol


def _apply_on_axes(T, M, axes, dims):
    """Contract matrix M (big-endian over `axes`) into tensor T on the given axes."""
    k = len(axes)
    sub = [dims[a] for a in axes]
    Mt = M.reshape(sub + sub)
    T = np.tensordot(Mt, T, axes=(list(range(k, 2 * k)), list(axes)))
    return np.moveaxis(T, list(range(k)), list(axes))


def apply_ops(ops, dims, state=None):
    """State-vector simulation. Returns the final state as a tensor of shape dims."""
    if state is None:
        state = np.zeros(dims, complex); state[(0,) * len(dims)] = 1.0
    state = np.asarray(state).reshape(dims)
    for op in ops:
        state = _apply_on_axes(state, op_matrix(op), op.targets, dims)
    return state


def ccx_pairs(a, b, c):
    """The qubit pairs of the six cx in the textbook Toffoli decomposition (controls a,b; target c).
    Used to charge each ccx with six 2-qubit depolarising errors."""
    return [(b, c), (a, c), (b, c), (a, c), (a, b), (a, b)]


def _depolarize2(rho, a, b, p, N):
    """Cirq's depolarize(p, n_qubits=2):  rho -> (1-p) rho + p/15 sum_{P != II} P rho P
       = (1 - 16p/15) rho + (16p/15) Tr_ab(rho) (x) I/4."""
    L = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    r, c = list(L[:N]), list(L[N:2 * N])
    full = "".join(r + c)
    c2 = c.copy(); c2[a], c2[b] = r[a], r[b]
    red_in = "".join(r + c2)
    red_out = "".join(x for i, x in enumerate(r) if i not in (a, b)) + \
              "".join(x for i, x in enumerate(c) if i not in (a, b))
    red = np.einsum(f"{red_in}->{red_out}", rho)
    eye = np.eye(2)
    D = np.einsum(f"{red_out},{r[a]}{c[a]},{r[b]}{c[b]}->{full}", red, eye, eye) / 4
    return (1 - 16 * p / 15) * rho + (16 * p / 15) * D


def apply_ops_density(ops, N, p2=0.0):
    """Exact density-matrix simulation of a qubit op list (real gates only), with a 2-qubit
    depolarising channel of strength p2 after every cx and after each of the six cx-equivalents
    of every ccx (ccx itself applied ideally). Returns rho as a (2,)*2N tensor."""
    dims = [2] * N
    rho = np.zeros(dims * 2); rho[(0,) * (2 * N)] = 1.0
    for op in ops:
        M = op_matrix(op)
        rho = _apply_on_axes(rho, M, op.targets, dims * 2)
        rho = _apply_on_axes(rho, M.conj(), [N + t for t in op.targets], dims * 2)
        if p2 > 0 and op.name == "cx":
            rho = _depolarize2(rho, *op.targets, p2, N)
        if p2 > 0 and op.name == "ccx":
            for a, b in ccx_pairs(*op.targets):
                rho = _depolarize2(rho, a, b, p2, N)
    return rho


def marginal(prob_tensor, keep):
    """Marginal probability over the registers in `keep` (in that order)."""
    N = prob_tensor.ndim
    P = prob_tensor.sum(axis=tuple(i for i in range(N) if i not in keep))
    order = sorted(keep)
    return np.transpose(P, [order.index(k) for k in keep])


def readout_flip(P, p_ro):
    """Independent bit-flip with probability p_ro on every axis of a distribution over bits."""
    K = np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    for ax in range(P.ndim):
        P = np.moveaxis(np.tensordot(K, P, axes=([1], [ax])), 0, ax)
    return P


def gate_counts(ops):
    c = {g: 0 for g in QUBIT_GATES + ("u",)}
    for op in ops:
        c[op.name] += 1
    c = {k: v for k, v in c.items() if v}
    c["total"] = len(ops)
    c["two_qubit_cx_equiv"] = c.get("cx", 0) + 6 * c.get("ccx", 0)
    return c


# ---- Cirq builder (NOT executed in the authoring environment: Cirq not installed) ----------
def to_cirq(ops, dims, measure=None, key="path"):
    """cirq.Circuit from the op list. Registers of dimension 2 become cirq.LineQubit(i), others
    cirq.LineQid(i, dimension=d). `measure`: list of register indices to measure under `key`."""
    import cirq
    qs = [cirq.LineQubit(i) if d == 2 else cirq.LineQid(i, dimension=d) for i, d in enumerate(dims)]
    named = {"h": cirq.H, "x": cirq.X, "z": cirq.Z, "cx": cirq.CNOT, "ccx": cirq.CCX}
    circuit = cirq.Circuit()
    for op in ops:
        tq = [qs[t] for t in op.targets]
        if op.name in named:
            circuit.append(named[op.name].on(*tq))
        elif op.name == "ry":
            circuit.append(cirq.ry(op.param).on(*tq))
        else:
            gate = cirq.MatrixGate(op.matrix, qid_shape=tuple(dims[t] for t in op.targets))
            circuit.append(gate.on(*tq))
    if measure:
        circuit.append(cirq.measure(*[qs[i] for i in measure], key=key))
    return circuit, qs


# ---- own OpenQASM 2 writer / parser for the six qubit gates ---------------------------------
def write_qasm(ops, N, measured, comment=""):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    if comment:
        lines += [f"// {c}" for c in comment.splitlines()]
    lines += [f"qreg q[{N}];", f"creg c[{len(measured)}];"]
    for op in ops:
        if op.name not in QUBIT_GATES:
            raise ValueError("only h,x,z,ry,cx,ccx can be written to QASM")
        args = ",".join(f"q[{t}]" for t in op.targets)
        par = f"({op.param:.17g})" if op.name == "ry" else ""
        lines.append(f"{op.name}{par} {args};")
    for j, q in enumerate(measured):
        lines.append(f"measure q[{q}] -> c[{j}];")
    return "\n".join(lines) + "\n"


def parse_qasm(text):
    """Minimal parser for files written by write_qasm. Returns (ops, N, measured)."""
    ops, N, measured = [], None, {}
    gate_re = re.compile(r"^(h|x|z|ry|cx|ccx)(?:\(([^)]*)\))?\s+(.+);$")
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "creg")):
            continue
        if line.startswith("qreg"):
            N = int(re.search(r"\[(\d+)\]", line).group(1)); continue
        if line.startswith("measure"):
            q, c = map(int, re.findall(r"\[(\d+)\]", line)); measured[c] = q; continue
        m = gate_re.match(line)
        if not m:
            raise ValueError(f"cannot parse: {raw}")
        name, par, args = m.groups()
        targets = tuple(int(t) for t in re.findall(r"q\[(\d+)\]", args))
        ops.append(Op(name, targets, param=float(par) if par else None))
    return ops, N, [measured[j] for j in range(len(measured))]
