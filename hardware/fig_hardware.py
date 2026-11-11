"""Figure: ideal simulation, pre-run noise model and hardware, for B1 and B2."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from score_runs import analyse

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams.update({"font.size": 9})
SURFACE, TEXT, TEXT2, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"
COLORS = {"ideal": "#2a78d6", "model": "#eb6834", "hw": "#1baf7a"}
LABELS = {"ideal": "Ideal simulation", "model": "Noise model set before the runs", "hw": "Hardware"}
DPI = 200

def rounded_bar(ax, x0, x1, h, color, radius_px=4):
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, 0), (x1, h)])
    if py1 - py0 < 0.5:
        return
    r = min(radius_px * DPI / 96, (px1 - px0) / 2, py1 - py0)
    v = [(px0, py0), (px0, py1 - r), (px0, py1), (px0 + r, py1), (px1 - r, py1), (px1, py1), (px1, py1 - r), (px1, py0), (px0, py0)]
    c = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), fc=color, ec="none", transform=None, zorder=3))

def crop_whitespace(path, pad=8):
    """Trim the blank margin left where the in-figure title used to be."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))).save(path)


R = analyse()
fig = plt.figure(figsize=(6.5, 3.35), dpi=DPI, facecolor=SURFACE)
for k, key in enumerate(("ideal", "model", "hw")):
    x = 0.02 + [0, 0.19, 0.50][k]
    fig.patches.append(Rectangle((x, 0.795), 0.013, 0.028, transform=fig.transFigure, fc=COLORS[key], ec="none"))
    fig.text(x + 0.019, 0.799, LABELS[key], fontsize=8, color=TEXT)
panels = [("B1", [0.075, 0.185, 0.27, 0.46], 0.42, [0, 0.1, 0.2, 0.3, 0.4]), ("B2", [0.43, 0.185, 0.555, 0.46], 0.62, [0, 0.2, 0.4, 0.6])]
for name, rect, ymax, ticks in panels:
    r = R[name]; n = len(r["paths"])
    ax = fig.add_axes(rect); ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(0, ymax)
    ax.set_yticks(ticks); ax.set_yticklabels([f"{int(round(t * 100))}%" if t else "0" for t in ticks])
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{p}\n{s} of {r['m']}" for p, s in zip(r["paths"], r["score"])], fontsize=7.6, linespacing=1.35)
    ax.tick_params(length=0, colors=TEXT2, labelsize=7.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.0, 1.13, f"{name} on {r['backend']}", transform=ax.transAxes, fontsize=8.4, color=TEXT, fontweight="semibold")
    ax.text(0.0, 1.045, f"{r['qubits']} qubits, {r['gates']['cz']} CZ gates after transpiling", transform=ax.transAxes, fontsize=7.4, color=TEXT2)
    ax.axhline(r["uniform"], color=AXIS, lw=0.9, zorder=1)
    ax.text(n - 0.42, r["uniform"] + ymax * 0.012, "uniform" if name == "B1" else "uniform output", fontsize=6.8, color=TEXT2,
            ha="right", va="bottom", zorder=6)
    fig.canvas.draw()
    bw, gap = 0.24, 0.03
    for i in range(n):
        for j, key in enumerate(("ideal", "model", "hw")):
            x0 = i + (j - 1.5) * (bw + gap) + gap / 2
            rounded_bar(ax, x0, x0 + bw, r[key][i], COLORS[key])
    # direct labels on the two hardware values the text discusses
    i = n - 1 if name == "B1" else 0
    x = i + 0.5 * (bw + gap) + gap / 2
    ax.text(x + (bw / 2 if name == "B1" else 0.02), r["hw"][i] + ymax * 0.015, f"{100 * r['hw'][i]:.1f}%",
            ha="center" if name == "B1" else "left", va="bottom", fontsize=7, color=TEXT, zorder=6)
fig.savefig("fig4_hardware.png", dpi=DPI, facecolor=SURFACE)
crop_whitespace("fig4_hardware.png")
print("saved")
