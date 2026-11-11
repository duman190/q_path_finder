"""Tuned-phase amplitude amplification (one round) on top of uniform and Gibbs preparations."""
import math
import numpy as np
from scipy.optimize import minimize
from instances import suite
from qpathfinder import SubspaceGMQAOA, exact_csp, gibbs_probs, larac_lambda

rng = np.random.default_rng(1)

def one_round(f, good):
    """max over (phi, beta) of P(good) after: phase e^{i phi} on good, reflection with angle beta about f."""
    def neg(x):
        a = f * np.where(good, np.exp(1j * x[0]), 1.0)
        a = a - (1 - np.exp(-1j * x[1])) * f * np.vdot(f, a)
        return -float(np.sum(np.abs(a[good]) ** 2))
    return max(-minimize(neg, rng.uniform(0, 6.28, 2), method="Nelder-Mead").fun for _ in range(30))

for dag, bound in suite():
    best, opt, paths, c, d = exact_csp(dag, bound)
    F = len(paths)
    good = np.zeros(F, bool); good[opt] = True
    lam = larac_lambda(c, d, bound)
    taus = np.linspace(0, 6, 601)
    pg, tau = max((gibbs_probs(c, d, lam, t)[opt].sum(), t) for t in taus)
    fu = np.full(F, 1 / math.sqrt(F), dtype=complex)
    fg = np.sqrt(gibbs_probs(c, d, lam, tau)).astype(complex)
    formula = lambda a: 1.0 if a >= 0.25 else math.sin(3 * math.asin(math.sqrt(a))) ** 2
    print(f"{dag.name:22s} uniform a={1/F:.4f}: tuned 1 round {one_round(fu, good):.4f} (formula {formula(1/F):.4f}) | "
          f"gibbs a={pg:.4f}: tuned 1 round {one_round(fg, good):.4f} (formula {formula(pg):.4f}) | "
          f"classical samples for 90%: uniform {math.ceil(math.log(0.1)/math.log(1-1/F))}, "
          f"gibbs {math.ceil(math.log(0.1)/math.log(1-pg))}")
