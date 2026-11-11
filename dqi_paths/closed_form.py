"""Finite-size closed form for the expected score (tridiagonal eigenvalue), checked against the
simulation, then evaluated at sizes that cannot be simulated."""
import math
import numpy as np
from dqi_paths import Instance, summarize


def finite_size(m, l, p, r):
    """m r/p + sqrt(r(p-r))/p * lambda_max(A), A tridiagonal with diagonal k*d and
    off-diagonal sqrt(k(m-k+1)), d = (p-2r)/sqrt(r(p-r)). Valid when 2l+1 < dual distance."""
    d = (p - 2 * r) / math.sqrt(r * (p - r))
    A = np.zeros((l + 1, l + 1))
    for k in range(l + 1):
        A[k, k] = k * d
        if k:
            A[k, k - 1] = A[k - 1, k] = math.sqrt(k * (m - k + 1))
    return m * r / p + math.sqrt(r * (p - r)) / p * np.linalg.eigvalsh(A)[-1]


def semicircle(m, l, p, r):
    a, b = l / m, r / p
    return m * (1.0 if b > 1 - a else (math.sqrt(a * (1 - b)) + math.sqrt(b * (1 - a))) ** 2)


if __name__ == "__main__":
    print("closed form vs simulation")
    for (p, n, r, seed) in [(5, 2, 2, 3), (7, 3, 3, 1), (13, 5, 6, 1), (17, 5, 8, 1), (31, 3, 15, 1), (61, 3, 30, 1)]:
        R = summarize(Instance(p, n, r, seed))
        cf = finite_size(R["m"], R["l"], p, r)
        cond = 2 * R["l"] + 1 < n + 1
        print(f"  p={p:2d} n={n} l={R['l']}: simulated {R['dqi']:.9f}  closed form {cf:.9f}  diff {abs(R['dqi']-cf):.1e}  "
              f"(condition 2l+1 < n+1 holds: {cond})")
    print("closed form at larger sizes (fractions of m)")
    for (p, n, r) in [(101, 11, 50), (521, 53, 260), (521, 261, 260), (4099, 411, 2049)]:
        m, l = p - 1, (n - 1) // 2
        print(f"  p={p} n={n} r={r} l={l}: DQI closed form {finite_size(m, l, p, r)/m:.4f} | semicircle {semicircle(m, l, p, r)/m:.4f} | "
              f"interpolation {(n + (m - n) * r / p)/m:.4f} | random {r/p:.4f}")
    # is the radius floor(n/2) the largest one for which decoding is unique?
    import itertools
    for (p, n) in [(7, 3), (11, 4)]:
        inst = Instance(p, n, p // 2, 0)
        for l in (n // 2, n // 2 + 1):
            syn = set(); total = 0
            for k in range(l + 1):
                for S in itertools.combinations(range(inst.m), k):
                    for vals in itertools.product(range(1, p), repeat=k):
                        s = tuple(int(v) for v in (np.array(vals) @ inst.B[list(S)]) % p) if k else (0,) * n
                        syn.add(s); total += 1
            print(f"  p={p} n={n}: patterns with <= {l} nonzeros: {total}, distinct syndromes: {len(syn)} -> unique decoding: {total == len(syn)}")
    # the toy instance in full
    inst = Instance(5, 2, 2, 3)
    s = inst.scores()
    print("toy instance p=5 n=2: allowed sets per metric y:", {int(y): np.nonzero(inst.allowed[i])[0].tolist() for i, y in enumerate(inst.ys)})
    print("scores over paths (rows x0, cols x1):\n", s)
    q = inst.dqi()
    print("DQI probabilities (rows x0, cols x1):\n", np.round(q["prob"], 4))
    for x in zip(*np.nonzero(s == s.max())):
        print("  optimal path", tuple(int(v) for v in x), "totals", inst.path_total(x).tolist(), "prob", round(float(q['prob'][x]), 4))
    print("B =\n", inst.B, "\ng_hat rows (abs):\n", np.abs(inst.g_hat()).round(3))
