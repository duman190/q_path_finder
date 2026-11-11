"""
dqi_paths.py -- the many-metric path problem and a state-level simulation of decoded quantum
interferometry (DQI) on it.

The problem (path language)
---------------------------
* A fabric with n stages. At stage j the path takes one of p parallel ports x_j in {0..p-1}.
  A path is x = (x_0, ..., x_{n-1}); there are p**n paths. p is prime.
* m = p-1 metrics, indexed y = 1..p-1. Port a at stage j weighs  a * y**j  (mod p) under metric y.
  The metric total of a path is  T_y(x) = sum_j x_j * y**j  (mod p).
* Every metric has a set of allowed residues F_y (r of the p residues).
* Goal: the path whose totals land in the allowed set for as many metrics as possible.

This is the Optimal Polynomial Intersection problem (the path is the coefficient vector of a
polynomial Q, and T_y(x) = Q(y)).

What is simulated
-----------------
Every step of DQI as a state vector, with no shortcut through the answer:
  1. superposition over small sets of metrics (at most l of them), with amplitudes taken from
     the Fourier coefficients of each metric's allowed set;
  2. the syndrome s = B^T e written into the n-digit path register;
  3. decoding: s -> e, which empties the metric register (here a lookup table; it is valid
     because the syndromes of all patterns with at most l nonzero entries are distinct, which
     the code checks);
  4. a Fourier transform over Z_p on each of the n digits;
  5. the measurement distribution over paths.
The result is compared with the direct formula (amplitude = polynomial of the score), with a
uniformly random path, with interpolation ("fix n metrics, leave the rest to chance"), and with
the brute-force optimum.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


class Instance:
    def __init__(self, p, n, r, seed=0, allowed=None):
        self.p, self.n, self.r = p, n, r
        self.m = p - 1
        self.ys = np.arange(1, p)
        self.B = np.array([[pow(int(y), j, p) for j in range(n)] for y in self.ys])   # m x n Vandermonde
        if allowed is None:
            rng = np.random.default_rng(seed)
            allowed = np.zeros((self.m, p), bool)
            for i in range(self.m):
                allowed[i, rng.choice(p, size=r, replace=False)] = True
        self.allowed = np.asarray(allowed, bool)
        assert self.allowed.shape == (self.m, p) and (self.allowed.sum(1) == r).all()
        self.l = n // 2                      # decoding radius: dual code has distance n+1
        self._totals = None

    # ---- classical side ------------------------------------------------------------
    def totals(self):
        """T[i] = array over all paths (shape (p,)*n) of metric i's total mod p."""
        if self._totals is None:
            grids = np.indices((self.p,) * self.n, dtype=np.int16)
            self._totals = [sum(int(self.B[i, j]) * grids[j] for j in range(self.n)) % self.p
                            for i in range(self.m)]
        return self._totals

    def scores(self):
        """Number of satisfied metrics for every path."""
        s = np.zeros((self.p,) * self.n, dtype=np.int16)
        for i, t in enumerate(self.totals()):
            s += self.allowed[i][t]
        return s

    def path_total(self, x):
        return (self.B @ np.asarray(x)) % self.p

    def path_score(self, x):
        t = self.path_total(x)
        return int(sum(self.allowed[i, t[i]] for i in range(self.m)))

    def interpolate(self, metrics, residues):
        """The path whose totals on the n chosen metrics equal the chosen residues
        (solves the n x n Vandermonde system mod p)."""
        p, n = self.p, self.n
        A = [[int(self.B[i, j]) for j in range(n)] + [int(v)] for i, v in zip(metrics, residues)]
        for c in range(n):
            piv = next(k for k in range(c, n) if A[k][c] % p)
            A[c], A[piv] = A[piv], A[c]
            inv = pow(A[c][c], p - 2, p)
            A[c] = [(a * inv) % p for a in A[c]]
            for k in range(n):
                if k != c and A[k][c]:
                    f = A[k][c]
                    A[k] = [(a - f * b) % p for a, b in zip(A[k], A[c])]
        return [A[k][n] for k in range(n)]

    # ---- quantum side: pieces of the DQI state --------------------------------------
    def g(self):
        """g[i, v]: the allowed-set indicator of metric i as +1/-1, shifted to zero mean and
        scaled to unit norm."""
        f = np.where(self.allowed, 1.0, -1.0)
        f = f - f.mean(axis=1, keepdims=True)
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    def g_hat(self):
        """Fourier coefficients over Z_p, with g[i, t] = p**-0.5 * sum_u w**(-u t) g_hat[i, u]."""
        p = self.p
        w = np.exp(2j * np.pi * np.outer(np.arange(p), np.arange(p)) / p)
        return (self.g() @ w.T) / math.sqrt(p)          # g_hat[i,u] = p^-1/2 sum_v w^{uv} g[i,v]

    def sector_syndrome_amplitudes(self, k, check=None):
        """Steps 1-3 for patterns with exactly k nonzero entries: amplitude array over the
        syndrome s in (Z_p)^n. `check` collects syndromes to confirm that decoding is possible."""
        p, n, m = self.p, self.n, self.m
        gh = self.g_hat()
        A = np.zeros(p ** n, dtype=complex)
        place = p ** np.arange(n - 1, -1, -1)                         # ravel index of a syndrome
        vals = np.arange(1, p)
        count = 0
        for S in itertools.combinations(range(m), k):
            if k == 0:
                A[0] += 1.0
                idx = np.zeros(1, dtype=np.int64)
            else:
                grids = np.meshgrid(*[vals] * k, indexing="ij")
                amp = np.ones((p - 1,) * k, dtype=complex)
                syn = np.zeros((p - 1,) * k + (n,), dtype=np.int64)
                for a, i in enumerate(S):
                    amp = amp * gh[i][grids[a]]
                    syn = syn + grids[a][..., None] * self.B[i][None, :]
                idx = ((syn % p) @ place).ravel()
                np.add.at(A, idx, amp.ravel())
            count += idx.size
            if check is not None:
                check.append(idx)
        return A / math.sqrt(math.comb(m, k)), count

    def dqi(self):
        """Runs the five steps. Returns dict with the sector states, the optimal weights w,
        the final distribution over paths, and checks."""
        p, n, m, l = self.p, self.n, self.m, self.l
        shape = (p,) * n
        s = self.scores().astype(float)
        check, sectors, n_patterns = [], [], 0
        for k in range(l + 1):
            A, cnt = self.sector_syndrome_amplitudes(k, check)
            n_patterns += cnt
            psi = np.fft.fftn(A.reshape(shape)) / math.sqrt(p ** n)      # step 4
            sectors.append(psi)
        all_syn = np.concatenate(check)
        decodable = np.unique(all_syn).size == all_syn.size              # step 3 is well defined
        gram = np.array([[np.vdot(a, b) for b in sectors] for a in sectors])
        S = np.array([[np.vdot(a, s * b) for b in sectors] for a in sectors])
        evals, evecs = np.linalg.eigh((S + S.conj().T) / 2)
        w = evecs[:, -1]
        psi = sum(wk * sk for wk, sk in zip(w, sectors))
        prob = np.abs(psi) ** 2
        return dict(sectors=sectors, w=w, prob=prob, expected=float((prob * s).sum()), norm=float(prob.sum()),
                    decodable=bool(decodable), n_patterns=int(n_patterns),
                    gram_error=float(np.abs(gram - np.eye(l + 1)).max()), top_eig=float(evals[-1]))

    def direct_sectors(self):
        """The same sector states from the definition: the k-th elementary symmetric polynomial
        of g_i(T_i(x)) over the metrics, normalised. No Fourier transform, no decoding."""
        g = self.g()
        E = [np.ones((self.p,) * self.n)] + [np.zeros((self.p,) * self.n) for _ in range(self.l)]
        for i, t in enumerate(self.totals()):
            gi = g[i][t]
            for k in range(self.l, 0, -1):
                E[k] = E[k] + gi * E[k - 1]
        return [e / np.linalg.norm(e) for e in E]

    # ---- reference numbers ----------------------------------------------------------
    def semicircle(self, l=None):
        l = self.l if l is None else l
        a, b = l / self.m, self.r / self.p
        return 1.0 if b > 1 - a else (math.sqrt(a * (1 - b)) + math.sqrt(b * (1 - a))) ** 2

    def interpolation_expectation(self):
        return self.n + (self.m - self.n) * self.r / self.p


def summarize(inst: Instance):
    s = inst.scores()
    q = inst.dqi()
    direct = inst.direct_sectors()
    mismatch = max(min(np.abs(a - b).max(), np.abs(a + b).max()) for a, b in zip(q["sectors"], direct))
    hist_random = np.bincount(s.ravel(), minlength=inst.m + 1) / s.size
    hist_dqi = np.bincount(s.ravel(), weights=q["prob"].ravel(), minlength=inst.m + 1)
    best = int(s.max())
    return dict(p=inst.p, n=inst.n, m=inst.m, r=inst.r, l=inst.l, paths=inst.p ** inst.n,
                random=inst.m * inst.r / inst.p, interpolation=inst.interpolation_expectation(),
                dqi=q["expected"], semicircle=inst.semicircle() * inst.m, best=best,
                n_best=int((s == best).sum()), p_best_dqi=float(q["prob"][s == best].sum()),
                p_best_random=float((s == best).mean()),
                decodable=q["decodable"], n_patterns=q["n_patterns"], gram_error=q["gram_error"],
                norm=q["norm"], sector_mismatch=float(mismatch), w=q["w"],
                hist_random=hist_random, hist_dqi=hist_dqi)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for (p, n, r, seed) in [(5, 2, 2, 3), (7, 3, 3, 1), (13, 5, 6, 1), (17, 5, 8, 1), (31, 3, 15, 1), (61, 3, 30, 1)]:
        inst = Instance(p, n, r, seed)
        R = summarize(inst)
        print(f"p={p:2d} stages n={n} metrics m={R['m']:2d} allowed r={r:2d} l={R['l']} paths={R['paths']:8d} "
              f"patterns={R['n_patterns']:7d} decodable={R['decodable']}")
        print(f"   checks: norm={R['norm']:.12f} gram_err={R['gram_error']:.1e} "
              f"pipeline-vs-direct sector mismatch={R['sector_mismatch']:.1e}")
        print(f"   expected metrics satisfied: random {R['random']:.3f} | interpolation {R['interpolation']:.3f} | "
              f"DQI {R['dqi']:.3f} | semicircle formula {R['semicircle']:.3f} | brute-force best {R['best']} "
              f"({R['n_best']} paths)")
        print(f"   as fractions of m: random {R['random']/R['m']:.3f} | interpolation {R['interpolation']/R['m']:.3f} | "
              f"DQI {R['dqi']/R['m']:.3f} | semicircle {R['semicircle']/R['m']:.3f} | best {R['best']/R['m']:.3f}")
        print(f"   P(sample is an optimal path): DQI {R['p_best_dqi']:.4f} vs random {R['p_best_random']:.6f}; w={np.real(R['w']).round(4)}")
