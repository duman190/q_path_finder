"""Instance suite for the qPathFinder experiments. Everything is deterministic (fixed seed)."""
import random

from qpathfinder import PathDAG, exact_csp


def fig2_dag():
    """The 2020 Figure 2 graph. Qubit order matches the 2020 Cirq run:
    q0=A->B, q1=A->C, q2=C->B, q3=B->D, q4=C->D."""
    arcs = [(0, 1), (0, 2), (2, 1), (1, 3), (2, 3)]
    #        A->B    A->C    C->B    B->D    C->D
    # A->B->D: cost 3, delay 10 | A->C->B->D: cost 5, delay 6 | A->C->D: cost 8, delay 3
    cost = [1, 2, 1, 2, 6]
    delay = [6, 1, 1, 4, 2]
    return PathDAG(4, arcs, 0, 3, cost, delay, name="I1 fig2 (4 nodes)", node_names=list("ABCD"))


def transit_mesh(m, seed):
    """s, t and m transit nodes: one direct arc plus m one-transit paths (DCN-like)."""
    rng = random.Random(seed)
    arcs, cost, delay = [(0, m + 1)], [rng.randint(6, 10)], [rng.randint(1, 3)]
    for k in range(1, m + 1):
        for a in ((0, k), (k, m + 1)):
            arcs.append(a)
            cst = rng.randint(1, 6)
            cost.append(cst)
            delay.append(max(1, 7 - cst + rng.randint(-1, 1)))
    return PathDAG(m + 2, arcs, 0, m + 1, cost, delay, name=f"I2 transit mesh m={m}")


def layered(width, depth, seed, name):
    """s -> depth fully connected layers of `width` nodes -> t. width**depth paths."""
    rng = random.Random(seed)
    node = lambda layer, i: 1 + layer * width + i
    t = 1 + width * depth
    arcs = [(0, node(0, i)) for i in range(width)]
    for layer in range(depth - 1):
        arcs += [(node(layer, i), node(layer + 1, j)) for i in range(width) for j in range(width)]
    arcs += [(node(depth - 1, i), t) for i in range(width)]
    cost, delay = [], []
    for _ in arcs:
        cst = rng.randint(1, 9)
        cost.append(cst)
        delay.append(max(1, 10 - cst + rng.randint(-2, 2)))   # cheap arcs tend to be slow
    return PathDAG(t + 1, arcs, 0, t, cost, delay, name=name)


def pick_delay_bound(dag):
    """Loosest delay bound for which the min-cost path is infeasible and the constrained
    optimum is unique -- so the constraint actually bites."""
    _, _, paths, c, d = exact_csp(dag, 10 ** 9)
    cmin = min(c)
    dmin_of_cheapest = min(d[i] for i in range(len(paths)) if c[i] == cmin)
    cands = sorted(set(x for x in d if x < dmin_of_cheapest), reverse=True)
    for bound in cands:
        best, opt, *_ = exact_csp(dag, bound)
        if best is not None and len(opt) == 1:
            return bound
    return cands[0] if cands else max(d)


def suite():
    out = [fig2_dag(),
           transit_mesh(4, seed=11),
           layered(2, 3, seed=7, name="I3 layered 2x3"),
           layered(3, 3, seed=5, name="I4 layered 3x3"),
           layered(4, 4, seed=3, name="I5 layered 4x4")]
    return [(g, pick_delay_bound(g)) for g in out]
