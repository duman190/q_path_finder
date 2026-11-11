"""Figures for the DQI many-metric path problem. Run: python3 figures.py  ->  fig1_problem.png, fig2_pipeline.png, fig3_scores.png"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

from dqi_paths import Instance, summarize

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

SURFACE = "#ffffff"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_WASH, BLUE_SOFT = "#e6f0fd", "#cde2fb"
GRAY_LINE = "#c9c8c2"
DPI = 200

toy = Instance(5, 2, 2, 3)
PATH = (2, 1)


# ------------------------------------------------------------------------------------------
def crop_whitespace(path, pad=8):
    """Trim the blank margin left where the in-figure title used to be."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))).save(path)


def fig1():
    fig = plt.figure(figsize=(6.5, 3.05), dpi=DPI, facecolor=SURFACE)
    # ---- left: the fabric ----
    ax = fig.add_axes([0.0, 0.0, 0.52, 0.86]); ax.set_xlim(0, 10); ax.set_ylim(0.1, 5.2); ax.axis("off")
    fig.text(0.02, 0.937, "A path picks one port per stage", fontsize=10, fontweight="semibold", color=TEXT)
    fig.text(0.02, 0.862, "2 stages, 5 ports each, 25 paths", fontsize=8.5, color=TEXT2)
    nodes = {"S": (1.0, 3.05), "M": (5.0, 3.05), "T": (9.0, 3.05)}
    offsets = [1.5, 0.75, 0.0, -0.75, -1.5]                       # port 0 at the top
    for stage, (a, b) in enumerate([("S", "M"), ("M", "T")]):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        for port, off in enumerate(offsets):
            chosen = PATH[stage] == port
            verts = [(x0, y0), ((x0 + x1) / 2, y0 + 2 * off), (x1, y1)]
            ax.add_patch(PathPatch(Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fill=False,
                                   lw=2.4 if chosen else 1.1, ec=BLUE if chosen else GRAY_LINE,
                                   zorder=3 if chosen else 1, capstyle="round"))
            ax.text((x0 + x1) / 2, y0 + off + 0.0, str(port), ha="center", va="center", fontsize=8,
                    color=TEXT if chosen else TEXT2, fontweight="semibold" if chosen else "normal", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.22", fc=SURFACE, ec=BLUE if chosen else GRAY_LINE,
                              lw=1.4 if chosen else 0.8))
        ax.text((x0 + x1) / 2, 0.92, f"stage {stage}", ha="center", fontsize=8.5, color=TEXT2)
        ax.text((x0 + x1) / 2, 0.50, f"port {PATH[stage]} chosen", ha="center", fontsize=8.5, color=TEXT)
    for name, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.42, fc=SURFACE, ec=TEXT, lw=1.2, zorder=6))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, color=TEXT, zorder=7)
    ax.set_aspect("equal")

    # ---- right: totals against allowed sets ----
    ax = fig.add_axes([0.60, 0.205, 0.37, 0.56]); ax.set_facecolor(SURFACE)
    fig.text(0.535, 0.937, "Each metric total must land in an allowed cell", fontsize=10, fontweight="semibold", color=TEXT)
    fig.text(0.535, 0.862, "total for metric y  =  x₀ + x₁·y  (mod 5)", fontsize=8.5, color=TEXT2)
    totals = toy.path_total(PATH)
    for i, y in enumerate(toy.ys):
        for v in range(5):
            if toy.allowed[i, v]:
                ax.add_patch(Rectangle((y - 0.44, v - 0.44), 0.88, 0.88, fc=BLUE_SOFT, ec="none", zorder=1))
    for i, y in enumerate(toy.ys):
        hit = toy.allowed[i, totals[i]]
        ax.scatter([y], [totals[i]], s=70, zorder=4, facecolor=BLUE if hit else SURFACE,
                   edgecolor=SURFACE if hit else TEXT, linewidth=1.6)
        if not hit:
            ax.annotate("miss", (y, totals[i]), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=7.5, color=TEXT2)
    ax.set_xlim(0.4, 4.6); ax.set_ylim(-0.6, 4.6)
    ax.set_xticks(range(1, 5)); ax.set_yticks(range(5))
    ax.set_xlabel("metric y", color=TEXT2, fontsize=8.5); ax.set_ylabel("total (mod 5)", color=TEXT2, fontsize=8.5)
    ax.tick_params(length=0, colors=TEXT2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for v in np.arange(-0.5, 5, 1):
        ax.axhline(v, color=GRID, lw=0.6, zorder=0)
    for v in np.arange(0.5, 5, 1):
        ax.axvline(v, color=GRID, lw=0.6, zorder=0)
    # legend (shape and fill, not colour alone)
    lx = 0.60
    fig.patches.append(Rectangle((lx, 0.035), 0.016, 0.034, transform=fig.transFigure, fc=BLUE_SOFT, ec="none"))
    fig.text(lx + 0.022, 0.040, "allowed", fontsize=7.5, color=TEXT2)
    fig.lines.append(plt.Line2D([lx + 0.125], [0.052], marker="o", ms=6, mfc=BLUE, mec=SURFACE, transform=fig.transFigure))
    fig.text(lx + 0.14, 0.040, "inside", fontsize=7.5, color=TEXT2)
    fig.lines.append(plt.Line2D([lx + 0.225], [0.052], marker="o", ms=6, mfc=SURFACE, mec=TEXT, mew=1.2, transform=fig.transFigure))
    fig.text(lx + 0.24, 0.040, "outside", fontsize=7.5, color=TEXT2)
    fig.text(0.26, 0.040, "totals (3, 4, 0, 1): 3 of 4 metrics satisfied", fontsize=8.5, color=TEXT, ha="center", fontweight="semibold")
    fig.savefig("fig1_problem.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, title, body, fc=SURFACE, ec=GRAY_LINE, title_color=TEXT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top", fontsize=7.6, color=title_color, zorder=3)
    ax.text(x + w / 2, y + 0.24, body, ha="center", va="bottom", fontsize=9, color=TEXT, zorder=3,
            fontweight="semibold")


def arrow(ax, p0, p1, color=TEXT2, rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=2, shrinkB=2))


def fig2():
    fig = plt.figure(figsize=(6.5, 3.55), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 7.1); ax.axis("off")
    cols = [1.55, 3.85, 6.15, 8.45, 10.75]
    w, h = 2.0, 1.12
    y_top, y_bot = 3.72, 1.55
    heads = ["1  Pick a few\nmetrics", "2  Write the\nsyndrome", "3  Decode and\nerase", "4  Fourier\ntransform", "5  Measure"]
    for c, t in zip(cols, heads):
        ax.text(c + w / 2, 6.45, t, ha="center", va="center", fontsize=8.6, color=TEXT, fontweight="semibold", linespacing=1.25)
    # lane labels
    ax.text(0.12, y_top + h / 2, "metric\nregister\n(4 digits)", ha="left", va="center", fontsize=7.6, color=TEXT2, linespacing=1.3)
    ax.text(0.12, y_bot + h / 2, "path\nregister\n(2 digits)", ha="left", va="center", fontsize=7.6, color=TEXT2, linespacing=1.3)
    # lane bands
    ax.add_patch(Rectangle((1.35, y_top - 0.22), 11.6, h + 0.44, fc="#f6f5f1", ec="none", zorder=0))
    ax.add_patch(Rectangle((1.35, y_bot - 0.22), 11.6, h + 0.44, fc="#f6f5f1", ec="none", zorder=0))
    top = ["metric 2 carries 3", "unchanged", "emptied", "idle", "idle"]
    topv = ["e = (0, 3, 0, 0)", "e = (0, 3, 0, 0)", "(0, 0, 0, 0)", "(0, 0, 0, 0)", "(0, 0, 0, 0)"]
    bot = ["empty", "s = Bᵀe", "unchanged", "now holds paths", "one path"]
    botv = ["(0, 0)", "s = (3, 1)", "s = (3, 1)", "Σ amp(x) |x⟩", "x = (2, 1)"]
    for i, c in enumerate(cols):
        active_top = i in (0, 2)
        active_bot = i in (1, 3, 4)
        box(ax, c, y_top, w, h, top[i], topv[i], fc=BLUE_WASH if active_top else SURFACE, ec=BLUE if active_top else GRAY_LINE)
        box(ax, c, y_bot, w, h, bot[i], botv[i], fc=BLUE_WASH if active_bot else SURFACE, ec=BLUE if active_bot else GRAY_LINE)
        if i < 4:
            arrow(ax, (c + w, y_top + h / 2), (cols[i + 1], y_top + h / 2), color=GRAY_LINE)
            arrow(ax, (c + w, y_bot + h / 2), (cols[i + 1], y_bot + h / 2), color=GRAY_LINE)
    # information flow between lanes
    arrow(ax, (cols[1] + 0.55, y_top), (cols[1] + 0.55, y_bot + h), color=BLUE, lw=1.6)
    ax.text(cols[1] + 0.68, (y_top + y_bot + h) / 2, "3·(1, 2)\nmod 5", fontsize=7.2, color=TEXT2, va="center", linespacing=1.25)
    arrow(ax, (cols[2] + 0.55, y_bot + h), (cols[2] + 0.55, y_top), color=BLUE, lw=1.6)
    ax.text(cols[2] + 0.68, (y_top + y_bot + h) / 2, "value = 3,\nmetric = 1÷3 = 2", fontsize=7.2, color=TEXT2, va="center", linespacing=1.25)
    # notes under the lanes
    notes = ["amplitudes: Fourier\ncoefficients of each\nmetric's allowed set",
             "one multiply-and-add\nper metric and stage",
             "an error-correcting\ndecoder; without it the\nnext step gives noise",
             "one transform over\nZ₅ per stage",
             "amplitude grows with\nthe number of metrics\nsatisfied"]
    for c, t in zip(cols, notes):
        ax.text(c + w / 2, 1.05, t, ha="center", va="top", fontsize=7.2, color=TEXT2, linespacing=1.3)
    ax.text(0.12, 5.42, "One branch of the superposition, on the instance of Figure 1", fontsize=7.6, color=MUTED, style="italic")
    fig.savefig("fig2_pipeline.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def rounded_bar(ax, x_left, x_right, height, color, radius_px=4):
    """Bar with square base and rounded data end, drawn in pixel space."""
    (px0, py0), (px1, py1) = ax.transData.transform([(x_left, 0), (x_right, height)])
    r = min(radius_px * DPI / 96, (px1 - px0) / 2, max(py1 - py0, 0))
    if py1 - py0 < 0.5:
        return
    verts = [(px0, py0), (px0, py1 - r), (px0, py1), (px0 + r, py1), (px1 - r, py1), (px1, py1), (px1, py1 - r), (px1, py0), (px0, py0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", transform=None, zorder=3))


def fig3():
    R = summarize(Instance(17, 5, 8, 1))
    m = R["m"]
    fig = plt.figure(figsize=(6.5, 3.0), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0.085, 0.17, 0.895, 0.60]); ax.set_facecolor(SURFACE)
    ymax = 0.30
    ax.set_xlim(-0.6, m + 0.6); ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.10, 0.20, 0.30]); ax.set_yticklabels(["0", "10%", "20%", "30%"])
    ax.set_xticks(range(0, m + 1))
    ax.tick_params(length=0, colors=TEXT2, labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRAY_LINE)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Metrics satisfied (of 16)", color=TEXT2, fontsize=8.5)
    fig.canvas.draw()
    bw, gap = 0.30, 0.035
    for s in range(m + 1):
        rounded_bar(ax, s - bw - gap / 2, s - gap / 2, R["hist_random"][s], ORANGE)
        rounded_bar(ax, s + gap / 2, s + bw + gap / 2, R["hist_dqi"][s], BLUE)
    # legend: swatch + text in text colour, means as the only direct numbers
    for k, (col, lab) in enumerate([(ORANGE, f"Random path, mean {R['random']:.1f}"), (BLUE, f"DQI sample, mean {R['dqi']:.1f}")]):
        x = 0.085 + 0.27 * k
        fig.patches.append(Rectangle((x, 0.782), 0.014, 0.03, transform=fig.transFigure, fc=col, ec="none"))
        fig.text(x + 0.021, 0.786, lab, fontsize=8, color=TEXT)
    ax.annotate("best path\nsatisfies all 16", xy=(16, 0.004), xytext=(15.2, 0.105), fontsize=7.5, color=TEXT2, ha="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_LINE, lw=0.8), linespacing=1.25)
    fig.savefig("fig3_scores.png", dpi=DPI, facecolor=SURFACE)
    crop_whitespace("fig3_scores.png")
    plt.close(fig)
    return R


if __name__ == "__main__":
    fig1(); fig2(); R = fig3()
    print("fig3 data: random", np.round(R["hist_random"], 4).tolist())
    print("fig3 data: dqi   ", np.round(R["hist_dqi"], 4).tolist())
