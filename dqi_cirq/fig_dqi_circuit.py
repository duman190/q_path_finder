"""Gate-level drawing of the DQI circuit for hardware instance B1, in the style of the Grover
circuit figure (grover_mcp/figures.py). Run: python3 fig_dqi_circuit.py -> fig_dqi_circuit_b1.png

The drawn gate list is the one in qasm/b1.qasm (= binary_toys.B1().ops()); a controlled RY is drawn
as one gate (it is ry, cx, ry, cx in the file) and a Toffoli with an X before and after a control
is drawn with an open control. Before drawing, the same gate list is simulated with numpy and the
path distribution is checked against the ideal output reported in the paper (1/3, 1/3, 1/3, 0).
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_dqi_circuit_b1.png")

# ------------------------------------------------------------------------------------------
# 1. check the gate list (qubit 0 = e1 most significant; 0..2 metric, 3..4 path)
def check():
    I2, X = np.eye(2), np.array([[0, 1], [1, 0]])
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    Z = np.diag([1, -1])

    def ry(t):
        return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]])

    n = 5
    st = np.zeros(2 ** n); st[0] = 1
    st = st.reshape([2] * n)

    def one(U, q):
        nonlocal st
        st = np.moveaxis(np.tensordot(U, st, axes=([1], [q])), 0, q)

    def ctrl(U, cs, t):
        nonlocal st
        idx = [slice(None)] * n
        for c in cs:
            idx[c] = 1
        sub = st[tuple(idx)].copy()
        tt = t - sum(1 for c in cs if c < t)
        sub = np.moveaxis(np.tensordot(U, sub, axes=([1], [tt])), 0, tt)
        st[tuple(idx)] = sub

    a1, a2 = 2 * math.acos(1 / math.sqrt(3)), 2 * math.acos(1 / math.sqrt(2))
    one(ry(math.pi / 3), 0)
    ctrl(ry(a1), [0], 1); ctrl(X, [1], 0)
    ctrl(ry(a2), [1], 2); ctrl(X, [2], 1)
    one(Z, 0)
    for c, t in [(0, 3), (1, 4), (2, 3), (2, 4)]:
        ctrl(X, [c], t)
    one(X, 4); ctrl(X, [3, 4], 0); one(X, 4)
    one(X, 3); ctrl(X, [3, 4], 1); one(X, 3)
    ctrl(X, [3, 4], 2)
    one(H, 3); one(H, 4)
    p = np.abs(st) ** 2
    assert abs(p[0, 0, 0].sum() - 1) < 1e-12, "metric register not erased"
    dist = {f"{x0},{x1}": float(p[0, 0, 0, x0, x1]) for x0 in (0, 1) for x1 in (0, 1)}
    want = {"0,0": 1 / 3, "1,0": 1 / 3, "1,1": 1 / 3, "0,1": 0.0}
    assert all(abs(dist[k] - want[k]) < 1e-12 for k in want), dist
    print("gate list check PASS:", {k: round(v, 4) for k, v in dist.items()})
    print(f"angles: {math.degrees(math.pi / 3):.1f}, {math.degrees(a1):.1f}, {math.degrees(a2):.1f} deg")


# ------------------------------------------------------------------------------------------
# 2. draw
names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("Inter", "Arial", "Helvetica Neue", "Helvetica") if f in names] + ["DejaVu Sans"]
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
SURFACE = "#ffffff"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
BLUE = "#2a78d6"
BLUE_WASH = "#e6f0fd"
BAND = "#f5f4ef"
DPI = 200


def circuit():
    W, H = 6.5, 3.55
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)

    # ---- inset: the fabric and its metrics ----
    ax = fig.add_axes([0.0, 0.0, 0.215, 0.84]); ax.set_xlim(0, 4); ax.set_ylim(0, 13.0); ax.axis("off")
    pos = {"s": (0.3, 11.0), "m": (2.0, 11.0), "t": (3.7, 11.0)}
    for (u, v, j) in (("s", "m", 0), ("m", "t", 1)):
        (x0, y0), (x1, y1) = pos[u], pos[v]
        for port, rad in ((0, -0.45), (1, 0.45)):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5, lw=1.0,
                                         color=TEXT2, shrinkA=7.5, shrinkB=7.5,
                                         connectionstyle=f"arc3,rad={rad}", zorder=2))
            ax.text((x0 + x1) / 2, y0 + (0.62 if port else -0.62), str(port), ha="center", va="center",
                    fontsize=6.8, color=BLUE, fontweight="bold")
        ax.text((x0 + x1) / 2, y0 - 1.25, f"stage {j}", ha="center", va="center", fontsize=5.6, color=TEXT2)
    for nme, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.3, fc=SURFACE, ec=TEXT, lw=1.0, zorder=3))
        ax.text(x, y, nme, ha="center", va="center", fontsize=6.6, color=TEXT, zorder=4)
    ax.set_aspect("equal")
    ax.text(0.2, 8.9, "Path x = (x0, x1).\nThree metrics, totals\nmod 2, one allowed\nvalue each:",
            fontsize=6.6, color=TEXT2, va="top", linespacing=1.35)
    rows = [("y1", "x0", "1"), ("y2", "x1", "0"), ("y3", "x0 ⊕ x1", "0")]
    for i, (y, tot, ok) in enumerate(rows):
        yy = 5.55 - 0.62 * i
        ax.text(0.35, yy, y, fontsize=6.6, color=BLUE, fontweight="bold", va="center")
        ax.text(1.05, yy, f"{tot} = {ok}", fontsize=6.6, color=TEXT, va="center")
    ax.text(0.2, 3.5, "Paths 00, 10, 11 satisfy\n2 of 3, path 01 none.\nIdeal output: 1/3, 1/3,\n1/3 and 0.",
            fontsize=6.6, color=TEXT2, va="top", linespacing=1.35)

    # ---- circuit ----
    ax = fig.add_axes([0.215, 0.0, 0.78, 0.84]); ax.set_xlim(-1.6, 19.4); ax.set_ylim(-0.4, 9.9); ax.axis("off")
    ye = [7.6, 6.6, 5.6]          # metric qubits e1..e3
    yx = [3.6, 2.6]               # path qubits x0, x1
    x_end = 18.2
    for y, lab in zip(ye, ["e1", "e2", "e3"]):
        ax.plot([0, x_end - 1.0], [y, y], color=TEXT, lw=0.7, zorder=1)
        ax.text(-0.25, y, lab, ha="right", va="center", fontsize=6.8, color=TEXT)
        ax.text(0.12, y + 0.27, "|0⟩", ha="left", va="center", fontsize=5.6, color=MUTED)
        ax.text(x_end - 0.85, y, "|0⟩", ha="left", va="center", fontsize=6.2, color=MUTED)
    for y, lab in zip(yx, ["x0", "x1"]):
        ax.plot([0, x_end], [y, y], color=TEXT, lw=0.7, zorder=1)
        ax.text(-0.25, y, lab, ha="right", va="center", fontsize=6.8, color=TEXT)
        ax.text(0.12, y + 0.27, "|0⟩", ha="left", va="center", fontsize=5.6, color=MUTED)
    Y = ye + yx                   # qubit index -> y

    def vline(x, y0, y1):
        ax.plot([x, x], [y0, y1], color=TEXT, lw=0.8, zorder=2)

    def ctrl(x, y, open_=False):
        ax.add_patch(Circle((x, y), 0.13, fc=SURFACE if open_ else TEXT, ec=TEXT, lw=0.8, zorder=4))

    def oplus(x, y):
        ax.add_patch(Circle((x, y), 0.24, fc=SURFACE, ec=TEXT, lw=0.8, zorder=4))
        ax.plot([x - 0.24, x + 0.24], [y, y], color=TEXT, lw=0.8, zorder=5)
        ax.plot([x, x], [y - 0.24, y + 0.24], color=TEXT, lw=0.8, zorder=5)

    def box(x, y, text, w=0.62, h=0.6, fc=SURFACE, fs=6.2):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=TEXT, lw=0.8, zorder=4))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT, zorder=5)

    def cx(x, c, t):
        vline(x, Y[c], Y[t]); ctrl(x, Y[c]); oplus(x, Y[t])

    def cry(x, c, t, text):
        vline(x, Y[c], Y[t]); ctrl(x, Y[c]); box(x, Y[t], text, w=1.2, h=0.74, fs=5.0)

    def ccx(x, c0, c1, t, open0=False, open1=False):
        vline(x, Y[t], Y[c1]); ctrl(x, Y[c0], open0); ctrl(x, Y[c1], open1); oplus(x, Y[t])

    # 1. superposition over patterns with at most one picked metric: w0|000> + w1|W3>
    box(1.0, Y[0], "Ry\n60°", w=1.0, h=0.74, fs=5.0)
    cry(2.3, 0, 1, "Ry\n109.5°")
    cx(3.4, 1, 0)
    cry(4.5, 1, 2, "Ry\n90°")
    cx(5.6, 2, 1)
    # 2. phase of the allowed sets: v = (1, 0, 0)
    box(7.0, Y[0], "Z", fc=BLUE_WASH)
    # 3. syndrome s = B^T e
    cx(8.4, 0, 3)
    cx(9.2, 1, 4)
    cx(10.0, 2, 3)
    cx(10.8, 2, 4)
    # 4. decode and erase: e_i flipped back when s equals row i of B
    ccx(12.2, 3, 4, 0, open1=True)    # row (1, 0)
    ccx(13.0, 3, 4, 1, open0=True)    # row (0, 1)
    ccx(13.8, 3, 4, 2)                # row (1, 1)
    # 5. Fourier transform over Z_2
    for y in yx:
        box(15.65, y, "H")
    # measurement of the path register only
    for y in yx:
        ax.add_patch(Rectangle((16.85, y - 0.31), 0.8, 0.62, fc=SURFACE, ec=TEXT, lw=0.8, zorder=4))
        ax.add_patch(Arc((17.25, y - 0.14), 0.5, 0.5, theta1=0, theta2=180, color=TEXT, lw=0.7, zorder=5))
        ax.plot([17.25, 17.46], [y - 0.14, y + 0.2], color=TEXT, lw=0.7, zorder=5)

    def band(x0, x1, title, sub):
        ax.add_patch(Rectangle((x0, 1.75), x1 - x0, 6.75, fc=BAND, ec="none", zorder=0))
        ax.text((x0 + x1) / 2, 9.45, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEXT)
        ax.text((x0 + x1) / 2, 8.9, sub, ha="center", va="center", fontsize=5.9, color=TEXT2)

    band(0.35, 6.25, "1. Pick at most one metric", r"$w_0|000\rangle + w_1|W_3\rangle$, $W_3$: one 1 of three")
    band(6.35, 7.65, "2. Phase", "v = 100")
    band(7.75, 11.35, "3. Syndrome", r"$s = B^{\mathsf{T}}e$")
    band(11.45, 14.45, "4. Decode", "erase e to 000")
    band(14.75, 16.55, "5. Fourier", "H per stage")

    ax.text(0.35, 1.15, "Open circles are controls on 0. A controlled Ry is two CNOTs in b1.qasm.", fontsize=6.0, color=TEXT2, va="center")
    ax.text(0.35, 0.55, "Rows of B: (1, 0), (0, 1), (1, 1). Only the path register is measured.", fontsize=6.0, color=TEXT2, va="center")
    fig.savefig(OUT, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    check()
    circuit()
