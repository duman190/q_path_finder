"""Figures for the Grover part. Run: python3 figures.py -> fig_grover_circuit.png, fig_grover_results.png"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
SURFACE = "#ffffff"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID, AXIS = "#e6e5e0", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_WASH = "#e6f0fd"
BAND = "#f5f4ef"
DPI = 200
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))


# ------------------------------------------------------------------------------------------
def circuit():
    W, H = 6.5, 4.25
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)

    # ---- inset: the DAG ----
    ax = fig.add_axes([0.0, 0.0, 0.2, 0.86]); ax.set_xlim(0, 4); ax.set_ylim(0, 17.2); ax.axis("off")
    pos = {"A": (0.75, 12.4), "C": (2.0, 14.4), "B": (2.0, 10.4), "D": (3.25, 12.4)}
    arcs = [("A", "B", "q0"), ("A", "C", "q1"), ("C", "B", "q2"), ("B", "D", "q3"), ("C", "D", "q4")]
    for u, v, q in arcs:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=1.0, color=TEXT2,
                                     shrinkA=7.5, shrinkB=7.5, zorder=2))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx = {"q0": -0.38, "q1": -0.38, "q2": 0.34, "q3": 0.38, "q4": 0.38}[q]
        dy = {"q0": -0.55, "q1": 0.55, "q2": 0.0, "q3": -0.55, "q4": 0.55}[q]
        ax.text(mx + dx, my + dy, q, ha="center", va="center", fontsize=7, color=BLUE, fontweight="semibold")
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.36, fc=SURFACE, ec=TEXT, lw=1.0, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=7.5, color=TEXT, zorder=4)
    ax.set_aspect("equal")
    ax.text(0.2, 8.3, "One qubit per arc.\nWeights per arc\n(w$_1$ cost, w$_2$ delay, w$_3$):", fontsize=6.6, color=TEXT2, va="top", linespacing=1.35)
    rows = [("q0", "1, 6, 2"), ("q1", "2, 1, 3"), ("q2", "1, 1, 4"), ("q3", "2, 4, 1"), ("q4", "6, 2, 2")]
    for i, (q, w) in enumerate(rows):
        ax.text(0.35, 5.75 - 0.62 * i, q, fontsize=6.6, color=BLUE, fontweight="semibold", va="center")
        ax.text(1.15, 5.75 - 0.62 * i, w, fontsize=6.6, color=TEXT, va="center")
    ax.text(0.2, 2.35, "Ranges: T$_1$ 2 to 4,\nT$_2$ 9 to 11, T$_3$ 2 to 4.\nOnly A→B→D is in range.", fontsize=6.6, color=TEXT2,
            va="top", linespacing=1.35)

    # ---- circuit ----
    ax = fig.add_axes([0.205, 0.0, 0.79, 0.86]); ax.set_xlim(-2.7, 25.4); ax.set_ylim(-0.95, 12.2); ax.axis("off")
    yq = [10.0, 9.0, 8.0, 7.0, 6.0]
    yacc = [4.3, 3.3, 2.3]
    ytest = 1.45
    yflag = 0.6
    x_end = 24.8
    labels = ["q0  A→B", "q1  A→C", "q2  C→B", "q3  B→D", "q4  C→D"]
    for y, lab in zip(yq, labels):
        ax.plot([0, x_end], [y, y], color=TEXT, lw=0.7, zorder=1)
        ax.text(-0.25, y, lab, ha="right", va="center", fontsize=6.8, color=TEXT)
        ax.text(0.12, y + 0.27, "|0⟩", ha="left", va="center", fontsize=5.6, color=MUTED)
    for y, lab in zip(yacc, ["T$_1$", "T$_2$", "T$_3$"]):
        ax.plot([0, x_end], [y - 0.06, y - 0.06], color=TEXT, lw=0.6, zorder=1)
        ax.plot([0, x_end], [y + 0.06, y + 0.06], color=TEXT, lw=0.6, zorder=1)
        ax.text(-0.25, y, f"total {lab}", ha="right", va="center", fontsize=6.8, color=TEXT)
        ax.text(0.12, y + 0.3, "|0⟩", ha="left", va="center", fontsize=5.6, color=MUTED)
    ax.plot([0, x_end], [ytest - 0.06, ytest - 0.06], color=TEXT, lw=0.6, zorder=1)
    ax.plot([0, x_end], [ytest + 0.06, ytest + 0.06], color=TEXT, lw=0.6, zorder=1)
    ax.text(-0.25, ytest, "tests (2K = 6)", ha="right", va="center", fontsize=6.8, color=TEXT)
    ax.text(0.12, ytest + 0.3, "|0⟩", ha="left", va="center", fontsize=5.6, color=MUTED)
    ax.plot([0, x_end], [yflag, yflag], color=TEXT, lw=0.7, zorder=1)
    ax.text(-0.25, yflag, "flag", ha="right", va="center", fontsize=6.8, color=TEXT)
    ax.text(0.12, yflag + 0.27, "|−⟩", ha="left", va="center", fontsize=5.6, color=MUTED)

    def vline(x, y0, y1, color=TEXT):
        ax.plot([x, x], [y0, y1], color=color, lw=0.8, zorder=2)

    def ctrl(x, y, open_=False):
        ax.add_patch(Circle((x, y), 0.13, fc=SURFACE if open_ else TEXT, ec=TEXT, lw=0.8, zorder=4))

    def oplus(x, y):
        ax.add_patch(Circle((x, y), 0.24, fc=SURFACE, ec=TEXT, lw=0.8, zorder=4))
        ax.plot([x - 0.24, x + 0.24], [y, y], color=TEXT, lw=0.8, zorder=5)
        ax.plot([x, x], [y - 0.24, y + 0.24], color=TEXT, lw=0.8, zorder=5)

    def box(x, y, text, w=0.8, h=0.62, fc=SURFACE, fs=6.2, ec=TEXT):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=0.8, zorder=4))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT, zorder=5)

    def cx(x, c, t):
        vline(x, yq[c], yq[t]); ctrl(x, yq[c]); oplus(x, yq[t])

    def cry(x, c, t, text):
        vline(x, yq[c], yq[t]); ctrl(x, yq[c]); box(x, yq[t], text, w=1.2, h=0.78, fs=5.0)

    # state preparation, exact gate list of state_prep_gates(I1)
    box(1.0, yq[0], "X", w=0.6)
    cry(2.15, 0, 1, "Ry\n109.5°")
    cx(3.2, 1, 0)
    cx(4.0, 1, 2)
    cry(5.1, 2, 4, "Ry\n90°")
    cx(6.1, 4, 2)
    cx(7.0, 0, 3)
    cx(7.9, 2, 3)

    # oracle: add, test, AND onto flag, untest, subtract.  A bracket on the arc wires means
    # "one singly-controlled constant addition per arc qubit", not a multi-controlled gate.
    def adder(x, k, text):
        ax.add_patch(Rectangle((x - 0.17, yq[4] - 0.3), 0.34, yq[0] - yq[4] + 0.6, fc=BLUE_WASH, ec=TEXT, lw=0.7, zorder=4))
        vline(x, yq[4] - 0.3, yacc[k])
        box(x, yacc[k], text, w=0.9, fc=BLUE_WASH, fs=5.6)

    adder(9.5, 0, "+w$_1$")
    adder(10.5, 1, "+w$_2$")
    adder(11.5, 2, "+w$_3$")

    def tests(x, inverse=False):
        vline(x, yacc[0], ytest)
        for k, t in enumerate(["2..4", "9..11", "2..4"]):
            box(x, yacc[k], t, w=1.15, fc=BLUE_WASH, fs=5.2)
        box(x, ytest, "C$^†$" if inverse else "C", w=0.6, fc=BLUE_WASH, fs=5.6)

    tests(12.8)
    xa = 13.95
    vline(xa, ytest, yflag); box(xa, ytest, "∧", w=0.6, fc=SURFACE, fs=6.4); oplus(xa, yflag)
    tests(15.1, inverse=True)
    adder(16.3, 2, "−w$_3$")
    adder(17.3, 1, "−w$_2$")
    adder(18.3, 0, "−w$_1$")

    # diffusion
    def big(x, text, w=1.35):
        ax.add_patch(Rectangle((x - w / 2, yq[4] - 0.45), w, yq[0] - yq[4] + 0.9, fc=SURFACE, ec=TEXT, lw=0.8, zorder=4))
        ax.text(x, (yq[0] + yq[4]) / 2, text, ha="center", va="center", fontsize=7, color=TEXT, zorder=5)

    big(19.5, "U$_S$$^†$")
    xz = 20.95
    vline(xz, yq[0], yflag)
    for y in yq:
        ctrl(xz, y, open_=True)
    oplus(xz, yflag)
    big(22.4, "U$_S$")

    # measurement
    for y in yq:
        ax.add_patch(Rectangle((23.75, y - 0.31), 0.8, 0.62, fc=SURFACE, ec=TEXT, lw=0.8, zorder=4))
        ax.add_patch(Arc((24.15, y - 0.14), 0.5, 0.5, theta1=0, theta2=180, color=TEXT, lw=0.7, zorder=5))
        ax.plot([24.15, 24.36], [y - 0.14, y + 0.2], color=TEXT, lw=0.7, zorder=5)

    # section bands and labels
    def band(x0, x1, title, sub):
        ax.add_patch(Rectangle((x0, -0.25), x1 - x0, 11.05, fc=BAND, ec="none", zorder=0))
        ax.text((x0 + x1) / 2, 11.75, title, ha="center", va="center", fontsize=7.4, fontweight="semibold", color=TEXT)
        ax.text((x0 + x1) / 2, 11.2, sub, ha="center", va="center", fontsize=6.2, color=TEXT2)

    band(0.5, 8.5, "State preparation U$_S$", "all three paths, equal amplitude")
    band(8.95, 18.85, "Oracle", "add, test, AND, untest, subtract")
    band(19.0, 23.25, "Diffusion", "reflect about U$_S$|0⟩")
    ax.annotate("", xy=(23.2, -0.45), xytext=(8.95, -0.45), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.plot([8.95, 8.95], [-0.45, -0.3], color=MUTED, lw=0.8); ax.plot([23.2, 23.2], [-0.45, -0.3], color=MUTED, lw=0.8)
    ax.text(16.1, -0.47, "  repeat k times  ", ha="center", va="center", fontsize=6.4, color=TEXT2,
            bbox=dict(fc=SURFACE, ec="none", pad=0.5))
    fig.savefig(os.path.join(HERE, "fig_grover_circuit.png"), dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=TEXT2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def results():
    B = json.load(open(os.path.join(HERE, "baselines.json")))
    fig = plt.figure(figsize=(6.5, 2.75), dpi=DPI, facecolor=SURFACE)

    # left: success against rounds for I4 and I5 (path-subspace simulation and sine law)
    ax = fig.add_axes([0.075, 0.17, 0.37, 0.78]); style(ax)
    for name, color, lab_xy in (("I4", ORANGE, (6.45, 0.40)), ("I5", BLUE, (8.0, 0.72))):
        row = next(r for r in R["instances"] if r["name"] == name)
        n, m = row["paths"], row["feasible"]
        th = math.asin(math.sqrt(m / n))
        ks = np.linspace(0, len(row["curve"]) - 1, 300)
        ax.plot(ks, np.sin((2 * ks + 1) * th) ** 2, color=color, lw=2, zorder=3)
        ax.plot(range(len(row["curve"])), row["curve"], "o", ms=5.2, mfc=color, mec=SURFACE, mew=1.2, zorder=4)
        ax.text(lab_xy[0], lab_xy[1], f"{name}, {n} paths", fontsize=7.8, color=TEXT, ha="left", va="center")
    ax.set_xlim(-0.3, 11.3); ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, 12, 2)); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Grover rounds k", fontsize=8, color=TEXT2)
    ax.set_ylabel("P(path in range)", fontsize=8, color=TEXT2)

    # right: evaluations to reach 90 % success, same target for every method
    ax = fig.add_axes([0.585, 0.17, 0.39, 0.78]); style(ax)
    S = B["scaling"]
    x = np.array([r["paths"] / r["feasible"] for r in S])
    xs = np.geomspace(10, 20000, 100)
    GRAY = "#5c5b57"
    ax.plot(xs, np.log(0.1) / np.log(1 - 1 / xs), color=ORANGE, lw=1.4, zorder=2)
    ax.plot(x, [r["uniform_90"] for r in S], "o", ms=4.6, mfc=ORANGE, mec=SURFACE, mew=1.0, zorder=4, label="Uniform sampling")
    ax.plot(x, [r["mitm_sub_90"] for r in S], "s", ms=4.2, mfc=GRAY, mec=SURFACE, mew=1.0, zorder=4, label="Sub-sampled MITM")
    ax.plot(x, [r["evals_90"] for r in S], "o", ms=4.6, mfc=BLUE, mec=SURFACE, mew=1.0, zorder=5, label="Grover")
    ax.legend(loc="upper left", frameon=False, fontsize=7.2, handletextpad=0.3, borderaxespad=0.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(10, 20000); ax.set_ylim(1, 60000)
    ax.set_xticks([10, 100, 1000, 10000]); ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_xlabel("Paths per path in range (N / M)", fontsize=8, color=TEXT2)
    ax.set_ylabel("Evaluations for 90%", fontsize=8, color=TEXT2)
    fig.savefig(os.path.join(HERE, "fig_grover_results.png"), dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


circuit()
results()
print("ok")
