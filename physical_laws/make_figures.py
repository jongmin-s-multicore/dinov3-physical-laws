"""Figures for the README.

usage: python physical_laws/make_figures.py out/viewer physical_laws/results/sep.json physical_laws/assets

laws_tokens.png  one ID video per law (ID 1 slot of extract.py), frames t = 0, 6, 12, 18 with the PCA RGB map of
                 their patch tokens next to each frame (stored frames).
separation.png   2-D PCA of the mean patch token on cleaned frames: state, first and second difference, coloured by law.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("viewer_data", help="output folder of extract.py")
ap.add_argument("sep_json")
ap.add_argument("out")
args = ap.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.unicode_minus": False, "axes.formatter.use_mathtext": True, "font.size": 15})
INK = "#222222"
LAW = [("uniform", "Uniform motion", "#2b6c8a"), ("parabola", "Parabola", "#c9692c"), ("collision", "Collision", "#6b6b6b")]
T, NT, G = 32, 256, 16

# ---- frames and tokens
meta = json.load(open(f"{args.viewer_data}/meta.json"))
K = meta["k"]
STEPS, P, GAP, ROWGAP = (0, 6, 12, 18), 256, 22, 22
L, TOP = 330, 74
W = L + len(STEPS) * 2 * P + (len(STEPS) - 1) * GAP
H = TOP + len(LAW) * P + (len(LAW) - 1) * ROWGAP
canvas = np.full((H, W, 3), 255, np.uint8)
for r, (key, name, _) in enumerate(LAW):
    vid = next(v for v in next(s for s in meta["scenarios"] if s["key"] == key)["videos"] if v["slot"] == 0)["id"]
    sprite = np.asarray(Image.open(f"{args.viewer_data}/frames/{vid}.png").convert("RGB"))
    rgb = np.frombuffer(open(f"{args.viewer_data}/feat/{vid}.bin", "rb").read(T * NT * 3), np.uint8).reshape(T, G, G, 3)
    y = TOP + r * (P + ROWGAP)
    for c, t in enumerate(STEPS):
        x = L + c * (2 * P + GAP)
        canvas[y : y + P, x : x + P] = sprite[(t // 8) * 256 : (t // 8 + 1) * 256, (t % 8) * 256 : (t % 8 + 1) * 256]
        canvas[y : y + P, x + P : x + 2 * P] = np.kron(rgb[t], np.ones((16, 16, 1), np.uint8))
dpi = 100
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(canvas, interpolation="nearest")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
for r in range(len(LAW)):
    y = TOP + r * (P + ROWGAP)
    for c in range(len(STEPS)):
        x = L + c * (2 * P + GAP)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 2 * P, P, fill=False, ec="#9a9a9a", lw=1.2))
        ax.plot([x + P - 0.5, x + P - 0.5], [y - 0.5, y + P - 0.5], color="#9a9a9a", lw=1.2)
for c, t in enumerate(STEPS):
    x = L + c * (2 * P + GAP)
    ax.text(x + P, TOP - 22, rf"$t = {t}$", ha="center", va="baseline", fontsize=26, color=INK)
    for dx, word in ((0, "frame"), (P, "tokens")):
        ax.text(x + dx + 8, TOP + 8, word, ha="left", va="top", fontsize=15, color="white",
                bbox=dict(boxstyle="square,pad=0.25", fc=(0, 0, 0, 0.72), ec="none"))
for r, (_, name, _) in enumerate(LAW):
    ax.text(L - 28, TOP + r * (P + ROWGAP) + P / 2, name, ha="right", va="center", fontsize=23, color=INK)
fig.savefig(f"{args.out}/laws_tokens.png", dpi=dpi)
plt.close(fig)

# ---- separation scatter
sep = json.load(open(args.sep_json))
sc = sep["clean"]["mean"]["scatter"]
fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6), dpi=110)
fig.subplots_adjust(left=0.03, right=0.99, bottom=0.08, top=0.9, wspace=0.12)
rng = np.random.default_rng(0)
for ax, (k, sym) in zip(axes, (("z", r"$z_t$"), ("d1", r"$\Delta z$"), ("d2", r"$\Delta^2 z$"))):
    pts = [(np.array(sc[k]["pts"][key]), col, name) for key, name, col in LAW]
    allp = np.concatenate([p for p, _, _ in pts])
    lo, hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0)
    pad = (hi - lo) * 0.06
    order = rng.permutation(sum(len(p) for p, _, _ in pts))
    xy = np.concatenate([p for p, _, _ in pts])[order]
    cols = np.concatenate([[c] * len(p) for p, c, _ in pts])[order]
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c=cols, alpha=0.55, linewidths=0)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9a9a"); s.set_linewidth(0.8)
    ax.text(0.03, 0.96, sym, transform=ax.transAxes, ha="left", va="top", fontsize=22, color=INK)
    v = sc[k]["var"]
    ax.set_xlabel(f"PC1 ({v[0] * 100:.0f} %)", fontsize=14, color="#555555", labelpad=6)
    ax.set_ylabel(f"PC2 ({v[1] * 100:.0f} %)", fontsize=14, color="#555555", labelpad=6)
handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=c, label=n) for _, n, c in LAW]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=15, handletextpad=0.3, columnspacing=2.2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{args.out}/separation.png")
plt.close(fig)
print("wrote", f"{args.out}/laws_tokens.png", f"{args.out}/separation.png")
