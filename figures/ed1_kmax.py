# -*- coding: utf-8 -*-
"""Extended Data Fig. 1: maximum of K over all memory marginals versus memory noise.
(a) depolarizing noise, (b) dephasing-biased family lambda = (t/2, t/2, t); per-qubit value max_W K_m^(1/m)
for m = 1, 2, 3 against max(1, kappa). Uses theory/maximize_K.py."""
import os, sys, json
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nstyle  # Nature-style formatting (figures/nstyle.py)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from maximize_K import setup, K_of_W
from kfunction import pauli_probs
rng = np.random.default_rng(1)


def kmax(lam, m, restarts):
    R, Ps, Ss = setup(lam, m); d = 2 ** m
    def f(v):
        A = (v[:d * d] + 1j * v[d * d:]).reshape(d, d); W = A @ A.conj().T; W /= np.trace(W).real
        return -K_of_W(W, R, Ps, Ss)
    best = max(K_of_W(np.eye(d) / d, R, Ps, Ss), 1.0)
    for r in range(restarts):
        res = minimize(f, rng.normal(size=2 * d * d) * (0.3 if r % 2 else 1), method="L-BFGS-B", options=dict(maxiter=300))
        best = max(best, -res.fun)
    return best


fams = {"depolarizing  $(\\eta,\\eta,\\eta)$": lambda t: (t, t, t),
        "dephasing-biased  $(t/2,t/2,t)$": lambda t: (t / 2, t / 2, t)}
out = {}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (name, fam) in enumerate(fams.items()):
    ts = [t for t in (np.linspace(0.35, 0.8, 10) if i == 0 else np.linspace(0.5, 1.0, 10)) if pauli_probs(fam(t)).min() >= 0]; S = [sum(x * x for x in fam(t)) for t in ts]
    ss = np.linspace(min(S), max(S), 200); ax[i].plot(ss, np.maximum(1, (1 + ss) / 2), "k-", lw=1.2, label=r"$\max(1,\kappa)$")
    for m, mk, rs in ((1, "o", 6), (2, "s", 6), (3, "^", 3)):
        v = [kmax(fam(t), m, rs) ** (1 / m) for t in ts]
        ax[i].plot(S, v, mk, ms=5, mfc="none", label=f"$m={m}$"); out[f"{name}|m={m}"] = dict(sum=S, kmax_per_qubit=v)
        print(name, m, np.round(v, 4), flush=True)
    ax[i].axvline(1, color="r", ls=":", lw=1); ax[i].set_xlabel(r"$\lambda_X^2+\lambda_Y^2+\lambda_Z^2$")
    ax[i].set_ylabel(r"$(\max_W K_m)^{1/m}$"); ax[i].set_title(("a", "b")[i] + ", " + name, fontsize=10, loc="left")
    ax[i].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ed1_kmax.pdf")); fig.savefig(os.path.join(HERE, "ed1_kmax.png"), dpi=150)
json.dump(out, open(os.path.join(HERE, "ed1_kmax.json"), "w"), indent=1)
