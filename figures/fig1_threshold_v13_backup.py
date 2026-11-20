# -*- coding: utf-8 -*-
"""Figure 1 (Nature-style layout): (a) learning with a noisy quantum memory, schematic; (b) memory-noise phase diagram;
(c) per-site factor on the dephasing family: ideal Bell pairs (optimal and standard memory frame), ideal three-qubit code
(theory/memory_factor.py) and the readout-mitigated hardware points of the second-day chain run (MS_origin_bias2, DATA_DIR)."""
import os, sys, json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc, Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "pdf.fonttype": 42})
C_SYS, C_NOISE, C_MEAS, C_BLUE, C_RED = "#cfe8dc", "#f6d2bf", "#e4e4e4", "#3b6fb6", "#c4452c"
fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.0], width_ratios=[1.0, 1.0], hspace=0.26, wspace=0.32,
                      left=0.075, right=0.985, top=0.965, bottom=0.075)
def label(ax, s, x=-0.1, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
# ------------------------------------------------------------------ a: schematic
a = fig.add_subplot(gs[0, :]); a.set_xlim(-0.45, 10); a.set_ylim(-0.08, 3.25); a.axis("off"); label(a, "a", x=-0.035, y=0.97)
def box(x, y, w, h, fc, text, fs=7.5, ec="0.25"):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.8, zorder=3))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
def meter(x, y, s=0.17):
    a.add_patch(FancyBboxPatch((x - s, y - s * 0.8), 2 * s, 1.6 * s, boxstyle="round,pad=0.01,rounding_size=0.04", fc="white", ec="0.25", lw=0.8, zorder=3))
    a.add_patch(Arc((x, y - s * 0.35), 1.3 * s, 1.3 * s, theta1=20, theta2=160, lw=0.8, zorder=4))
    a.plot([x, x + 0.45 * s], [y - s * 0.35, y + 0.4 * s], color="0.2", lw=0.8, zorder=4)
rows = [2.45, 1.55, 0.35]
a.text(-0.3, 3.1, "Learning with a noisy quantum memory", fontsize=7.5, fontweight="bold")
for k, y in enumerate(rows):
    ys, ym = y + 0.2, y - 0.2
    a.plot([0.55, 3.45], [ys, ys], color="0.2", lw=0.9, zorder=1); a.plot([0.55, 3.45], [ym, ym], color="0.2", lw=0.9, zorder=1)
    a.plot([0.55, 0.55], [ym, ys], color="0.2", lw=0.9); a.plot([0.55], [y], "o", color="0.2", ms=3.2)
    box(1.45, ys - 0.15, 0.75, 0.3, C_SYS, r"$\Lambda_P$")
    box(1.3, ym - 0.15, 1.05, 0.3, C_NOISE, r"$\mathcal{N}_{\boldsymbol{\lambda}}$")
    box(2.85, ym - 0.17, 0.62, ys - ym + 0.34, C_MEAS, "Bell\nmeas.", fs=6.3)
    a.annotate("", (4.15, y), (3.5, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.3"))
    if k == 0:
        a.text(0.42, ys, "system", ha="right", va="center", fontsize=6.3, color="0.3"); a.text(0.42, ym, "memory", ha="right", va="center", fontsize=6.3, color="0.3")
a.text(1.8, 0.98, r"$\vdots$", ha="center", va="center", fontsize=10)
a.text(-0.33, 1.4, "$n$ sites", rotation=90, ha="center", va="center", fontsize=7)
a.annotate("", (-0.2, 0.05), (-0.2, 2.75), arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3"))
box(4.2, 0.12, 1.25, 2.6, "white", "decode\n" + r"$P$" + "\nfrom all\nsites", fs=6.8)
# right part: the two regimes
x0 = 6.05
a.text(x0, 3.08, "Without memory", fontsize=7.5, fontweight="bold")
a.plot([x0 + 0.15, x0 + 1.75], [2.6, 2.6], color="0.2", lw=0.9, zorder=1); a.plot([x0 + 0.15], [2.6], "o", color="0.2", ms=3.2)
box(x0 + 0.55, 2.45, 0.7, 0.3, C_SYS, r"$\Lambda_P$"); meter(x0 + 1.9, 2.6)
a.text(x0 + 2.35, 2.6, r"$\Omega(2^n)$ uses", va="center", fontsize=7)
a.text(x0, 1.95, "With memory", fontsize=7.5, fontweight="bold")
a.text(x0, 1.62, r"uses $\propto\Pi_j\,\nu_j^{-1}$;  advantage $A_n=\Pi_j\,2\nu_j$", fontsize=7)
a.add_patch(FancyBboxPatch((x0, 0.2), 3.85, 1.15, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f7f7f7", ec="0.7", lw=0.6))
a.text(x0 + 0.12, 1.12, r"$\sum\lambda^2\leq1$:", fontsize=7, fontweight="bold"); a.text(x0 + 1.15, 1.12, "no protocol has a growing advantage", fontsize=7)
a.text(x0 + 0.12, 0.78, r"$\sum\lambda^2>3/2$:", fontsize=7, fontweight="bold"); a.text(x0 + 1.15, 0.78, r"Bell pairs: $A_n$ grows", fontsize=7)
a.text(x0 + 0.12, 0.44, r"in between:", fontsize=7, fontweight="bold"); a.text(x0 + 1.15, 0.44, "only codes can make it grow", fontsize=7)
# ------------------------------------------------------------------ b: phase diagram
b = fig.add_subplot(gs[1, 0]); label(b, "b", x=-0.17)
L = np.linspace(0, 1, 500); X, Z = np.meshgrid(L, L)
notch = X > (1 + Z) / 2; eb = 2 * X + Z <= 1; S = 2 * X ** 2 + Z ** 2
img = np.ones(X.shape + (3,))
img[(S <= 1) & ~eb & ~notch] = (0.86, 0.90, 0.97); img[eb & ~notch] = (0.70, 0.78, 0.92); img[notch] = (0.88, 0.88, 0.88)
img[(S > 1) & (S <= 1.5) & ~notch] = (0.99, 0.93, 0.85); img[(S > 1.5) & ~notch] = (0.97, 0.80, 0.70)
b.imshow(img, origin="lower", extent=[0, 1, 0, 1], aspect="auto", interpolation="bilinear")
th = np.linspace(0, np.pi / 2, 300)
b.plot(np.sin(th) / np.sqrt(2), np.cos(th), "k-", lw=1.4)
b.plot(np.sqrt(1.5) * np.sin(th) / np.sqrt(2), np.sqrt(1.5) * np.cos(th), color=C_RED, lw=1.0)
xb = np.linspace(0.59, 1, 200); b.plot(xb, np.sqrt(1 / 8) / xb ** 2, color=C_RED, lw=0.7, ls="--")
b.plot(L, 1 - 2 * L, "k--", lw=0.7); b.plot(L, L, ":", color="0.35", lw=0.6)
for v, t in ((1 / 3, "1/3"), (1 / np.sqrt(3), r"$1/\sqrt{3}$"), (1 / np.sqrt(2), r"$1/\sqrt{2}$")):
    b.plot(v, v, "ko", ms=2.8); b.annotate(t, (v, v), (v + 0.03, v - 0.065), fontsize=6.3)
b.plot(0.580, 0.580, "D", color="#2a7f62", ms=3.8)
b.annotate("[[32,20,4]] code", (0.58, 0.58), (0.30, 0.45), fontsize=6, color="#2a7f62", arrowprops=dict(arrowstyle="-", color="#2a7f62", lw=0.5))
b.annotate("", (0.03, 0.975), (0.93, 0.975), arrowprops=dict(arrowstyle="<-", color="k", lw=0.9)); b.text(0.92, 0.93, r"$(a,a,1)$", fontsize=6.3, ha="right")
b.text(0.13, 0.22, "entanglement\nbreaking\n(proved)", fontsize=6.3)
b.text(0.33, 0.66, r"$\sum\lambda^2\leq1$" + "\nno growing\nadvantage", fontsize=6.3, ha="center")
b.text(0.87, 0.84, "Bell\nprobes", fontsize=6.3, color="#7a2a14", ha="center")
b.text(0.82, 0.15, "not a\nchannel", fontsize=6.3, color="0.4", ha="center")
b.set_xlabel(r"$\lambda_X=\lambda_Y$"); b.set_ylabel(r"$\lambda_Z$"); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.set_xticks([0, 0.25, 0.5, 0.75, 1]); b.set_yticks([0, 0.25, 0.5, 0.75, 1])
# ------------------------------------------------------------------ c: per-site factor on the dephasing family
from memory_factor import nu as nu_ideal
c = fig.add_subplot(gs[1, 1]); label(c, "c", x=-0.17)
aa = np.linspace(0.3, 1.0, 141)
c.axhspan(0.5, 1.03, color="#fdf0ea", zorder=0, lw=0); c.axhline(0.5, color="k", lw=0.6)
c.text(0.99, 0.515, "advantage grows", ha="right", fontsize=6.3, color="#7a2a14"); c.text(0.99, 0.46, "advantage shrinks", ha="right", fontsize=6.3, color="0.35")
c.plot(aa, (1 + 2 * aa ** 2) / 3, "-", color=C_BLUE, lw=1.3, label="Bell pair, optimal frame")
c.plot(aa, aa ** (4 / 3), "--", color=C_BLUE, lw=1.0, label="Bell pair, standard frame")
c.plot(aa, [nu_ideal("rep3", (x, x, 1.0)) for x in aa], "-", color=C_RED, lw=1.3, label="3-qubit code")
try:
    d = json.load(open(os.path.join(DATA, "MS_origin_bias2_mitigate.json")))
    for kind, col in (("bare", C_BLUE), ("rep3", C_RED)):
        pts = sorted((int(k.split("dz")[1]) / 100, v[0]) for k, v in d.items() if k.startswith(f"mit_{kind}_dz") and not k.endswith("dz100"))
        x = np.array([p[0] for p in pts]); y = np.array([p[1]["nu"] for p in pts]); ci = np.array([p[1]["ci"] for p in pts])
        c.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", ms=3.5, mfc="white", mew=0.9, color=col, capsize=1.2, lw=0.8,
                   label="hardware, unencoded" if kind == "bare" else "hardware, 3-qubit code")
except FileNotFoundError:
    pass
c.set_xlim(0.3, 1.0); c.set_ylim(0, 1.03); c.set_xlabel(r"$a$  (memory noise $(a,a,1)$)"); c.set_ylabel(r"per-site factor $\nu$")
c.legend(loc="lower right", frameon=False, handlelength=1.8)
for ax in (b, c):
    for sp in ("top", "right"): ax.spines[sp].set_visible(ax is b)
fig.savefig(os.path.join(HERE, "fig1_threshold.pdf")); fig.savefig(os.path.join(HERE, "fig1_threshold.png"), dpi=220)
print("ok")
