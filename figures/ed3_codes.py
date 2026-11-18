# -*- coding: utf-8 -*-
"""Extended Data Fig. 3: per-site factor nu_soft (geometric mean over full-weight logical letters) of coded
memories under depolarizing noise, versus the fidelity eta. Crossing of 1/2: threshold of the code; dotted:
1/sqrt(3) (sum lambda^2 = 1) and 1/sqrt(2) (Bell probes). Uses theory/code_thresholds.py."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from code_thresholds import code_lib, logicals, figures
C = code_lib(); show = ["[[4,2,2]]", "[[6,4,2]]", "[[12,10,2]]", "[[7,1,3]]", "[[15,7,3]]", "[[31,21,3]]", "[[16,6,4]]", "[[32,20,4]]", "[[64,50,4]]"]
eta = np.linspace(0.45, 0.85, 41)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(eta, eta ** 2, "k-", lw=1.6, label="Bell (unencoded)")
cm = plt.get_cmap("tab10")
for i, name in enumerate(show):
    S = C[name]; m = S.shape[1] // 2; LX, LZ = logicals(S, m)
    v = [np.exp(np.mean(np.log(figures(S, LX, LZ, e, rng=np.random.default_rng(0))[0]))) for e in eta]
    ax[0].plot(eta, v, color=cm(i % 10), lw=1, label=name); ax[1].plot(eta, v, color=cm(i % 10), lw=1)
ax[1].plot(eta, eta ** 2, "k-", lw=1.6)
for a in ax:
    a.axhline(0.5, color="r", lw=0.8); a.axvline(1 / np.sqrt(3), color="0.4", ls=":", lw=1); a.axvline(1 / np.sqrt(2), color="0.4", ls="--", lw=1)
    a.set_xlabel(r"depolarizing fidelity $\eta$"); a.set_ylabel(r"$\nu_{\rm soft}$")
ax[0].set_title("a, coded memories", fontsize=10, loc="left"); ax[0].legend(fontsize=7, ncol=2)
ax[1].set_xlim(0.55, 0.72); ax[1].set_ylim(0.35, 0.65); ax[1].set_title(r"b, near the threshold ($1/\sqrt{3}$ dotted, $1/\sqrt{2}$ dashed)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ed3_codes.pdf")); fig.savefig(os.path.join(HERE, "ed3_codes.png"), dpi=150)
