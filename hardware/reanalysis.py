"""Re-analysis of the two hardware runs reported in the paper.

No new hardware data: everything here is computed from runs.json and the instance definitions.

  1. Weights. The circuits that ran use the l = 1 weights (w0, w1) that maximise the expected
     score (the top eigenvector of a 2x2 matrix). This script recomputes that matrix without
     enumerating paths, from the signed number of weight-3 codewords of ker(B^T), and compares
     the tuned reference with the instance-independent weights (1/sqrt 2, 1/sqrt 2).
  2. Gain kept by the hardware against both references, and z-scores with the null (uniform)
     and the empirical standard error.
  3. A one-parameter global-depolarising model on the transpiled circuit: if every transpiled
     CZ fails with probability p and a failure randomises the output, the kept gain is
     (1 - p)^(number of CZ). The script solves for the p that matches each run.
  4. Two- and three-qubit gates that touch each path qubit in the logical circuit of B2.

Run: python3 reanalysis.py  ->  reanalysis.txt
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqi_cirq"))
from binary_toys import B1, B2, score_counts   # noqa: E402

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def dqi_l1(inst, w):
    X = inst.paths()
    F = np.where((X @ inst.B.T) % 2 == inst.v, 1.0, -1.0)
    S = F.sum(1)
    amp = w[0] + w[1] * S / math.sqrt(inst.m)        # times 2^(-n/2)
    prob = amp ** 2 / len(X)
    return prob / prob.sum(), (S + inst.m) / 2


def weight3_matrix(inst):
    """Expected value of S = sum_i f_i under the l = 1 state, as a quadratic form in (w0, w1).
    E[S] = 0 and E[S^2] = m when d_perp >= 3; E[S^3] = 6 sigma, sigma the signed number of
    weight-3 codewords {i, j, k} (b_i + b_j + b_k = 0), sign (-1)^(v_i + v_j + v_k)."""
    m = inst.m
    sigma = 0
    for i, j, k in itertools.combinations(range(m), 3):
        if not ((inst.B[i] + inst.B[j] + inst.B[k]) % 2).any():
            sigma += (-1) ** int(inst.v[i] + inst.v[j] + inst.v[k])
    A = np.array([[0.0, math.sqrt(m)], [math.sqrt(m), 6 * sigma / m]])
    ev, vec = np.linalg.eigh(A)
    w = vec[:, -1] * np.sign(vec[0, -1])
    return sigma, A, ev[-1], w


def main():
    rec = json.load(open(os.path.join(HERE, "runs.json")))
    for run in rec["runs"]:
        inst = {"B1": B1, "B2": B2}[run["instance"]]()
        name, N = run["instance"], run["shots"]
        say("=" * 88)
        say(f"{name}: m = {inst.m} metrics, n = {inst.n} stages, v = {inst.v.tolist()}, backend {run['backend']}")
        sigma, A, lam, w_formula = weight3_matrix(inst)
        w_tuned = inst.reference()["w"]
        say(f"  signed weight-3 codewords of ker(B^T): sigma = {sigma}; matrix {A.round(4).tolist()}; "
            f"top eigenvalue {lam:.4f} -> mean score (m + lambda)/2 = {(inst.m + lam) / 2:.4f}")
        say(f"  weights from that matrix {w_formula.round(6).tolist()}, weights used on hardware "
            f"{np.asarray(w_tuned).round(6).tolist()}")
        hw = np.asarray(score_counts(inst, run["counts"], bit_order="qiskit")["dist"]).ravel()
        s = inst.scores().ravel().astype(float)
        uni = np.full_like(hw, 1 / hw.size)
        mean_u = float((uni * s).sum())
        mean_hw = float((hw * s).sum())
        refs = {}
        for label, w in (("tuned", w_tuned), ("standard", (1 / math.sqrt(2), 1 / math.sqrt(2)))):
            P, sc = dqi_l1(inst, np.asarray(w))
            mean_r = float((P * s).sum())
            refs[label] = mean_r
            best, worst = s == s.max(), s == s.min()
            say(f"  {label:>8} weights ({w[0]:.4f}, {w[1]:.4f}): mean score {mean_r:.4f}, best-path share "
                f"{P[best].sum():.4f}, worst-score share {P[worst].sum():.4f}, per path "
                + ", ".join(f"{''.join(map(str, x))}:{p:.4f}" for x, p in zip(inst.paths(), P)))
        say(f"  hardware mean score {mean_hw:.4f}, uniform {mean_u:.4f}")
        for label, mr in refs.items():
            say(f"  gain kept against the {label} reference: {(mean_hw - mean_u) / (mr - mean_u):.3f}")
        var_null = float((uni * (s - mean_u) ** 2).sum())
        var_emp = float((hw * (s - mean_hw) ** 2).sum())
        say(f"  z-score of the mean against uniform: null SE {(mean_hw - mean_u) / math.sqrt(var_null / N):.1f}, "
            f"empirical SE {(mean_hw - mean_u) / math.sqrt(var_emp / N):.1f}")
        if name == "B2":
            best = s == s.max()
            p0 = 1 / hw.size
            z = (hw[best].sum() - p0) / math.sqrt(p0 * (1 - p0) / N)
            say(f"  best-path share z-score with the binomial null SE: {z:.1f}")
        kept = (mean_hw - mean_u) / (refs["tuned"] - mean_u)
        ncz = run["transpiled_ops"]["cz"]
        p_eff = 1 - kept ** (1 / ncz)
        say(f"  global depolarising per transpiled CZ: {ncz} CZ, kept gain {kept:.3f} -> p_eff = {100 * p_eff:.2f} %; "
            f"(1 - 0.01)^{ncz} = {(0.99) ** ncz:.3f}; median CZ error "
            f"{100 * rec['calibration'][run['backend']]['cz_error_median']:.3f} % -> "
            f"{(1 - rec['calibration'][run['backend']]['cz_error_median']) ** ncz:.3f}")
        if name == "B2":
            ops = inst.ops()
            touch = {j: {"2q": 0, "3q": 0} for j in range(inst.n)}
            for op in ops:
                for j, q in enumerate(inst.path):
                    if q in op.targets and len(op.targets) >= 2:
                        touch[j]["2q" if len(op.targets) == 2 else "3q"] += 1
            say(f"  logical gates touching each path qubit (stage: two-qubit, Toffoli): "
                + "; ".join(f"stage {j}: {t['2q']}, {t['3q']}" for j, t in touch.items()))
    open(os.path.join(HERE, "reanalysis.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
