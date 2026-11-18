# -*- coding: utf-8 -*-
"""Figure 1: memory-noise phase diagram (a) and crossing points on the dephasing family (b).
Hardware crossings are those reported in the paper (fig3_code.py and the slope interpolation of Fig. 2)."""
import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw=dict(width_ratios=[1.05, 1]))
a = ax[0]; L = np.linspace(0, 1, 400)
X, Z = np.meshgrid(L, L)
notch = X > (1 + Z) / 2
eb = 2 * X + Z <= 1
inside = (2 * X**2 + Z**2 <= 1) & ~eb & ~notch
bell = (X**4 * Z**2 > 1 / 8) & ~notch
img = np.zeros(X.shape + (3,)) + 1.0
img[inside] = (0.86, 0.90, 0.97); img[eb] = (0.70, 0.78, 0.92); img[notch] = (0.85, 0.85, 0.85)
img[(2 * X**2 + Z**2 > 1) & ~notch & ~bell] = (0.99, 0.93, 0.85); img[bell] = (0.97, 0.80, 0.70)
a.imshow(img, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
th = np.linspace(0, np.pi / 2, 300); a.plot(np.sin(th) / np.sqrt(2), np.cos(th), "k-", lw=1.8)
a.plot(L, 1 - 2 * L, "k--", lw=1)
xb = np.linspace(0.59, 1, 200); a.plot(xb, np.sqrt(1 / 8) / xb**2, color="#b5482b", lw=1.3)
d = [(1 / 3, "1/3"), (1 / np.sqrt(3), r"$1/\sqrt{3}$"), (1 / np.sqrt(2), r"$1/\sqrt{2}$")]
a.plot(L, L, ":", color="0.3", lw=0.8)
for v, t in d: a.plot(v, v, "ko", ms=4); a.annotate(t, (v, v), (v + 0.03, v - 0.07), fontsize=8)
a.plot(0.580, 0.580, "D", color="#2a7f62", ms=5); a.annotate("[[32,20,4]]\ncode 0.580", (0.58, 0.58), (0.36, 0.42), fontsize=7, color="#2a7f62",
                                                            arrowprops=dict(arrowstyle="-", color="#2a7f62", lw=0.6))
a.annotate("", (0.02, 0.985), (0.95, 0.985), arrowprops=dict(arrowstyle="<-", color="k", lw=1.2))
a.text(0.93, 0.935, r"$(a,a,1)$", fontsize=7.5, ha="right")
a.text(0.12, 0.25, "entanglement\nbreaking:\nno advantage\n(proved)", fontsize=7.5)
a.text(0.33, 0.66, r"$\sum\lambda^2\leq1$" + "\nno advantage\n(conjecture)", fontsize=7.5, ha="center")
a.text(0.80, 0.86, "Bell probes\n" + r"$\nu>1/2$", fontsize=7.5, color="#7a2a14", ha="center")
a.text(0.80, 0.16, "not a\nchannel", fontsize=7.5, color="0.4", ha="center")
a.set_xlabel(r"$\lambda_X=\lambda_Y$"); a.set_ylabel(r"$\lambda_Z$"); a.set_title("(a) memory-noise phase diagram", fontsize=10)
a.set_xlim(0, 1); a.set_ylim(0, 1)
b = ax[1]
rows = [("fundamental limit (conjecture)", 0.0, None, "k"),
        ("ideal Bell probe", 0.595, None, "#3b6fb6"), ("ideal 3-qubit code", 0.39, None, "#c4452c"),
        ("hardware Bell, 9 pairs, end to end", 0.747, None, "#3b6fb6"),
        ("hardware Bell, memory level", 0.626, (0.614, 0.638), "#3b6fb6"),
        ("hardware 3-qubit code, memory level", 0.536, (0.524, 0.547), "#c4452c"),
        ("hardware 3-qubit code, end to end", 0.741, (0.732, 0.750), "#c4452c")]
for i, (t, x, ci, c) in enumerate(rows):
    y = len(rows) - i
    b.plot([x], [y], "o" if "hardware" in t else ("s" if "ideal" in t else "*"), color=c, ms=7 if "fund" not in t else 11,
           mfc=c if "memory" not in t else "white")
    if ci: b.plot(ci, [y, y], color=c, lw=2)
    b.text(x + 0.025 if x < 0.7 else x - 0.025, y + 0.18, t, fontsize=7.5, ha="left" if x < 0.7 else "right")
b.set_yticks([]); b.set_xlim(-0.03, 0.9); b.set_ylim(0.3, len(rows) + 0.8)
b.set_xlabel(r"crossing $a^*$ ($\nu=1/2$) on the family $\lambda=(a,a,1)$")
b.set_title("(b) where the advantage stops growing", fontsize=10)
b.axvspan(-0.03, 0.0, color="0.85")
fig.tight_layout(); HERE = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(HERE, "fig1_threshold.pdf")); fig.savefig(os.path.join(HERE, "fig1_threshold.png"), dpi=170)
