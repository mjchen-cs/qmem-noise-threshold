# -*- coding: utf-8 -*-
"""Figure 1: memory-noise phase diagram (a) and the per-site factor on the dephasing family (b): ideal Bell pairs in the
optimal and the standard memory frame, the ideal three-qubit code (theory/memory_factor.py), and the readout-mitigated
hardware points of the second-day chain run (MS_origin_bias2, DATA_DIR)."""
import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw=dict(width_ratios=[1.05, 0.8]))
a = ax[0]; L = np.linspace(0, 1, 400)
X, Z = np.meshgrid(L, L)
notch = X > (1 + Z) / 2
eb = 2 * X + Z <= 1
inside = (2 * X**2 + Z**2 <= 1) & ~eb & ~notch
bell = (2 * X**2 + Z**2 > 1.5) & ~notch          # Bell probes in the optimal memory frame: nu = sum(lambda^2)/3 > 1/2
img = np.zeros(X.shape + (3,)) + 1.0
img[inside] = (0.86, 0.90, 0.97); img[eb] = (0.70, 0.78, 0.92); img[notch] = (0.85, 0.85, 0.85)
img[(2 * X**2 + Z**2 > 1) & ~notch & ~bell] = (0.99, 0.93, 0.85); img[bell] = (0.97, 0.80, 0.70)
a.imshow(img, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
th = np.linspace(0, np.pi / 2, 300); a.plot(np.sin(th) / np.sqrt(2), np.cos(th), "k-", lw=1.8)
a.plot(L, 1 - 2 * L, "k--", lw=1)
a.plot(np.sqrt(1.5) * np.sin(th) / np.sqrt(2), np.sqrt(1.5) * np.cos(th), color="#b5482b", lw=1.3)
xb = np.linspace(0.59, 1, 200); a.plot(xb, np.sqrt(1 / 8) / xb**2, color="#b5482b", lw=0.8, ls="--")   # standard frame, (lx ly lz)^(2/3) = 1/2
d = [(1 / 3, "1/3"), (1 / np.sqrt(3), r"$1/\sqrt{3}$"), (1 / np.sqrt(2), r"$1/\sqrt{2}$")]
a.plot(L, L, ":", color="0.3", lw=0.8)
for v, t in d: a.plot(v, v, "ko", ms=4); a.annotate(t, (v, v), (v + 0.03, v - 0.07), fontsize=8)
a.plot(0.580, 0.580, "D", color="#2a7f62", ms=5); a.annotate("[[32,20,4]]\ncode 0.580", (0.58, 0.58), (0.36, 0.42), fontsize=7, color="#2a7f62",
                                                            arrowprops=dict(arrowstyle="-", color="#2a7f62", lw=0.6))
a.annotate("", (0.02, 0.985), (0.95, 0.985), arrowprops=dict(arrowstyle="<-", color="k", lw=1.2))
a.text(0.93, 0.935, r"$(a,a,1)$", fontsize=7.5, ha="right")
a.text(0.12, 0.25, "entanglement\nbreaking:\nno advantage\n(proved)", fontsize=7.5)
a.text(0.33, 0.66, r"$\sum\lambda^2\leq1$" + "\nno advantage\n(conjecture)", fontsize=7.5, ha="center")
a.text(0.88, 0.74, "Bell probes\n" + r"$\nu>1/2$", fontsize=7.5, color="#7a2a14", ha="center")
a.text(0.80, 0.16, "not a\nchannel", fontsize=7.5, color="0.4", ha="center")
a.set_xlabel(r"$\lambda_X=\lambda_Y$"); a.set_ylabel(r"$\lambda_Z$"); a.set_title("(a) memory-noise phase diagram", fontsize=10)
a.set_xlim(0, 1); a.set_ylim(0, 1)
b = ax[1]
# (b) per-site factor on the dephasing family: ideal protocols (lines) and hardware (points); the advantage grows iff nu > 1/2
import sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from memory_factor import nu as nu_ideal
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
aa = np.linspace(0.3, 1.0, 141)
b.axhspan(0.5, 1.02, color="#fdf0ea", zorder=0, lw=0); b.axhline(0.5, color="k", lw=0.8)
b.text(0.985, 0.52, "advantage grows", ha="right", fontsize=7.5, color="#7a2a14"); b.text(0.985, 0.455, "advantage shrinks", ha="right", fontsize=7.5, color="0.35")
b.plot(aa, (1 + 2 * aa ** 2) / 3, "-", color="#3b6fb6", lw=1.6, label="Bell pair, optimal frame")
b.plot(aa, aa ** (4 / 3), "--", color="#3b6fb6", lw=1.3, label="Bell pair, standard frame")
b.plot(aa, [nu_ideal("rep3", (x, x, 1.0)) for x in aa], "-", color="#c4452c", lw=1.6, label="3-qubit code")
try:      # hardware, second day, one chain, readout-mitigated (memory level)
    d = json.load(open(os.path.join(DATA, "MS_origin_bias2_mitigate.json")))
    for kind, c in (("bare", "#3b6fb6"), ("rep3", "#c4452c")):
        pts = sorted((int(k.split("dz")[1]) / 100, v[0]) for k, v in d.items() if k.startswith(f"mit_{kind}_dz") and not k.endswith("dz100"))
        x = np.array([p[0] for p in pts]); y = np.array([p[1]["nu"] for p in pts]); ci = np.array([p[1]["ci"] for p in pts])
        b.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", ms=4.5, mfc="white", color=c, capsize=1.5, lw=1,
                   label=("hardware, unencoded" if kind == "bare" else "hardware, 3-qubit code"))
except FileNotFoundError:
    pass
b.set_xlim(0.3, 1.0); b.set_ylim(0, 1.02); b.set_xlabel(r"$a$  (memory noise $(a,a,1)$)"); b.set_ylabel(r"per-site factor $\nu$")
b.legend(fontsize=6.8, loc="lower right", frameon=False)
b.set_title("(b) per-site factor on the dephasing family", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_threshold.pdf")); fig.savefig(os.path.join(HERE, "fig1_threshold.png"), dpi=170)
