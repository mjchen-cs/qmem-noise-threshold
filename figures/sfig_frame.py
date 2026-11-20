# -*- coding: utf-8 -*-
"""Supplementary figure: Bell pair in the standard and in the best memory frame, and the three-qubit code, on one chain
(111-120-129-138, Wukong 180, third session, run MS_origin_frame, readout-mitigated = memory level, 4,000 circuits per point).
(a) per-site factor versus the dephasing parameter a (noise (a, a, 1)); lines: zero-parameter models nu(1) a^(4/3) (standard
frame) and nu(1) ((1+2a)/3)^2 (best frame) with nu(1) from the interleaved noiseless circuits, and a power-law fit for the code;
vertical lines: a = (3/sqrt2 - 1)/2, the limit of Bell sampling in any frame on any processor, and a = 0.5 (sum lambda^2 = 3/2).
(b) the three letter factors of the unencoded pair: the best frame equalizes them."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import nstyle  # noqa: F401
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
d = json.load(open(os.path.join(DATA, "MS_origin_frame_mitigate.json")))
col = {"bare": "#3b6fb6", "bareF": "#2a7f62", "rep3": "#c4452c"}
lab = {"bare": "unencoded, standard frame", "bareF": "unencoded, best frame", "rep3": "3-qubit code"}
def series(kind):
    pts = sorted((int(k.split("dz")[1]) / 100, v[0]) for k, v in d.items() if k.startswith(f"mit_{kind}_dz") and not k.endswith("dz100"))
    a = np.array([p[0] for p in pts]); nu = np.array([p[1]["nu"] for p in pts]); ci = np.array([p[1]["ci"] for p in pts])
    L = np.array([p[1]["letters"] for p in pts]); return a, nu, ci, L
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9))
aa = np.linspace(0.47, 0.77, 200)
for kind in ("bare", "bareF", "rep3"):
    a, nu, ci, L = series(kind); n1 = d[f"mit_{kind}_dz100"][0]["nu"]
    ax.errorbar(a, nu, yerr=[nu - ci[:, 0], ci[:, 1] - nu], fmt="o", ms=3.5, capsize=1.5, color=col[kind], label=lab[kind])
    if kind == "bare": ax.plot(aa, n1 * aa ** (4 / 3), "-", color=col[kind], lw=1)
    elif kind == "bareF": ax.plot(aa, n1 * ((1 + 2 * aa) / 3) ** 2, "-", color=col[kind], lw=1)
    else:
        p, c = np.polyfit(np.log(a), np.log(nu), 1); ax.plot(aa, np.exp(c) * aa ** p, "-", color=col[kind], lw=1)
ax.axhline(0.5, color="k", ls="--", lw=0.7)
ax.axvline((3 / np.sqrt(2) - 1) / 2, color="0.2", ls="-.", lw=0.9, label="Bell sampling, any frame (ideal)")
ax.axvline(0.5, color="0.6", ls="--", lw=0.6, label=r"Bell-input bound ($\sum\lambda^2=3/2$)")
ax.set_xlabel(r"$a$   (memory noise $(a,a,1)$)"); ax.set_ylabel(r"per-site factor $\nu$"); ax.set_title("a   memory frame and code", loc="left")
ax.legend(fontsize=5.5, frameon=False, loc="upper left")
for kind, mk in (("bare", "o"), ("bareF", "s")):
    a, nu, ci, L = series(kind)
    for i, (ls, name) in enumerate(((":", "X"), ("--", "Y"), ("-", "Z"))):
        bx.plot(a, L[:, i], marker=mk, ls=ls, color=col[kind], ms=3, lw=0.9, label=f"{'standard' if kind == 'bare' else 'best'} frame, {name}")
n1 = d["mit_bareF_dz100"][0]["nu"]; bx.plot(aa, n1 * aa ** 2, color="0.6", lw=0.6); bx.plot(aa, n1 * ((1 + 2 * aa) / 3) ** 2, color="0.6", lw=0.6)
bx.set_xlabel(r"$a$"); bx.set_ylabel(r"letter factor $G_\ell$"); bx.set_title("b   letter factors of the unencoded pair", loc="left")
bx.set_ylim(0.15, 1.12); bx.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left")
fig.savefig(os.path.join(HERE, "sfig_frame.pdf")); fig.savefig(os.path.join(HERE, "sfig_frame.png"), dpi=200); print("ok")
