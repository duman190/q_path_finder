"""Independent check of the agent's hardware files: my own QASM reader and simulator, and my own
reference computed from the definition of the DQI state. Shares no code with dqi_cirq/."""
import itertools, math, os, re, sys
import numpy as np

def run_qasm(path):
    txt = open(path).read()
    n = int(re.search(r"qreg q\[(\d+)\]", txt).group(1))
    psi = np.zeros((2,) * n, complex); psi[(0,) * n] = 1
    meas = {}
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
    def one(u, q, sub=None, ax=None):
        arr = psi if sub is None else sub
        m = np.moveaxis(arr, q if ax is None else ax, 0); m[...] = np.tensordot(u, m, axes=([1], [0]))
    counts = dict(h=0, x=0, z=0, ry=0, cx=0, ccx=0)
    for line in txt.splitlines():
        line = line.split("//")[0].strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg")): continue
        if line.startswith("measure"):
            q, c = map(int, re.findall(r"\[(\d+)\]", line)); meas[c] = q; continue
        name = re.match(r"[a-z]+", line).group(0); qs = [int(v) for v in re.findall(r"q\[(\d+)\]", line)]
        counts[name] += 1
        if name == "h": one(H, qs[0])
        elif name == "x": one(X, qs[0])
        elif name == "z": one(Z, qs[0])
        elif name == "ry":
            th = float(re.search(r"ry\(([^)]+)\)", line).group(1))
            one(np.array([[math.cos(th/2), -math.sin(th/2)], [math.sin(th/2), math.cos(th/2)]]), qs[0])
        elif name in ("cx", "ccx"):
            ctrl, t = qs[:-1], qs[-1]
            idx = [slice(None)] * n
            for c in ctrl: idx[c] = 1
            sub = psi[tuple(idx)]
            ax = t - sum(1 for c in ctrl if c < t)
            one(X, t, sub=sub, ax=ax)
        else:
            raise ValueError(line)
    prob = np.abs(psi) ** 2
    order = [meas[c] for c in sorted(meas)]                 # qubit measured into c[0], c[1], ...
    others = tuple(i for i in range(n) if i not in order)
    # probability that every unmeasured qubit is 0, and the distribution over the measured ones
    idx = [0] * n
    for q in order: idx[q] = slice(None)
    clean = prob[tuple(idx)]
    clean = np.moveaxis(clean, range(clean.ndim), np.argsort(np.argsort(order)))  # axes -> c[0], c[1], ...
    marg = prob.sum(axis=others)
    marg = np.moveaxis(marg, range(marg.ndim), np.argsort(np.argsort(order)))
    return n, counts, marg, float(clean.sum())

def reference(B, v):
    """DQI l=1 from the definition: amplitude(x) = w0*psi0 + w1*psi1(x), psi1 ~ sum_i f_i."""
    B = np.array(B); m, n = B.shape
    xs = list(itertools.product([0, 1], repeat=n))
    f = np.array([sum(1 if ((B[i] @ np.array(x)) % 2) == v[i] else -1 for i in range(m)) for x in xs], float)
    s = (m + f) / 2
    psi0 = np.ones(len(xs)) / math.sqrt(len(xs)); psi1 = f / np.linalg.norm(f)
    assert abs(psi0 @ psi1) < 1e-12
    S = np.array([[a @ (s * b) for b in (psi0, psi1)] for a in (psi0, psi1)])
    w = np.linalg.eigh(S)[1][:, -1]
    p = (w[0] * psi0 + w[1] * psi1) ** 2
    return np.array(p).reshape((2,) * n), s.reshape((2,) * n), abs(w)

cases = {"b1": ([[1, 0], [0, 1], [1, 1]], [1, 0, 0]),
         "b2": ([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(1, 8)], [1, 1, 0, 1, 0, 1, 1])}
for name, (B, v) in cases.items():
    n, counts, marg, clean = run_qasm(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm", f"{name}.qasm"))
    # the agent's row order for B2 may differ from mine: read it from the file's own module instead
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dqi_paths")); import binary_toys as bt
    inst = [i for i in (bt.B1(), bt.B2())][0 if name == "b1" else 1] if hasattr(bt, "B1") else None
    Bq = np.array(inst.B) if inst is not None and hasattr(inst, "B") else np.array(B)
    vq = list(inst.v) if inst is not None and hasattr(inst, "v") else v
    ref, s, w = reference(Bq, vq)
    print(f"{name}: qubits {n}, gates {counts}, 2q-equivalent {counts['cx'] + 6*counts['ccx']}")
    print(f"   B rows {Bq.tolist()}  v {vq}")
    print(f"   P(all unmeasured qubits end in 0) = {clean:.12f}")
    print(f"   max |P_qasm - P_definition| = {np.abs(marg - ref).max():.2e}   weights {w.round(6)}")
    print(f"   E[metrics]: DQI {float((marg*s).sum()):.4f}  uniform {s.mean():.4f}  best {int(s.max())} ({int((s==s.max()).sum())} path(s)),  P(best) {float(marg[s==s.max()].sum()):.4f}")
    print("   distribution c[0]c[1]..:", {"".join(map(str, k)): round(float(marg[k]), 4) for k in itertools.product([0, 1], repeat=marg.ndim)})
