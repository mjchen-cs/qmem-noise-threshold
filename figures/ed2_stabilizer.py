# -*- coding: utf-8 -*-
"""Extended Data Fig. 2: stabilizer-code memories. (a) K = 2^k 2^{-H2(L|E)} for random [[m,k]] codes (m = 2..7) and
random Pauli channels on and inside the sphere sum lambda^2 = 1, and outside it (per-logical-qubit value K^(1/k)
against kappa). (b) Depolarizing noise at eta = 1/sqrt(3): K (average over syndromes) and the largest
syndrome-conditioned value 2^k sum_L P(L|e)^2. Uses theory/stabilizer_K.py."""
import os, sys, json
import numpy as np, stim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nstyle  # Nature-style formatting (figures/nstyle.py)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from stabilizer_K import pvec, symp, code_from_tableau, K_code, joint_table
rng = np.random.default_rng(5); NC = int(os.environ.get("NCODES", 300))


def rand_lam(r):
    while True:
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * r
        if pvec(v).min() >= -1e-12: return tuple(v)


A = {"S": [], "K": [], "m": [], "kap": [], "Kk": []}; B = {}
for m in range(2, 8):
    digs, X, Zb = symp(m); eta = 1 / np.sqrt(3); wK = 0; wE = 0
    for t in range(NC):
        k = int(rng.integers(1, m + 1)); T = stim.Tableau.random(m); stabs, logs = code_from_tableau(T, m, k)
        r = rng.uniform(0.6, 1.25); lam = rand_lam(r); S = r * r
        K = K_code(lam, m, k, stabs, logs, digs, X, Zb)
        A["S"].append(S); A["K"].append(K); A["m"].append(m); A["kap"].append((1 + S) / 2); A["Kk"].append(K ** (1 / k))
        if k < m:
            P = joint_table((eta, eta, eta), m, k, stabs, logs, digs, X, Zb); q = P.sum(1); ok = q > 0
            col = (P[ok] ** 2).sum(1) / q[ok] ** 2
            wK = max(wK, 2 ** k * np.sum(q[ok] * col)); wE = max(wE, 2 ** k * col.max())
    B[m] = (wK, wE); print(m, wK, wE, flush=True)
S = np.array(A["S"]); K = np.array(A["K"]); Kk = np.array(A["Kk"]); M = np.array(A["m"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cm = plt.get_cmap("viridis")
for m in range(2, 8):
    s = M == m; ax[0].scatter(S[s], Kk[s], s=6, color=cm((m - 2) / 5), label=f"$m={m}$", alpha=0.7)
ss = np.linspace(S.min(), S.max(), 100); ax[0].plot(ss, np.maximum(1, (1 + ss) / 2), "k-", lw=1, label=r"$\max(1,\kappa)$")
ax[0].axvline(1, color="r", ls=":", lw=1); ax[0].set_xlabel(r"$\sum\lambda^2$"); ax[0].set_ylabel(r"$K^{1/k}$ (per logical qubit)")
ax[0].set_title(f"a, random stabilizer codes ({NC} per $m$)", fontsize=10, loc="left"); ax[0].legend(fontsize=7, ncol=2)
ms = list(B); ax[1].plot(ms, [B[m][0] for m in ms], "o-", label="$K$ (averaged over syndromes)")
ax[1].plot(ms, [B[m][1] for m in ms], "s--", label=r"max over syndromes of $2^k\sum_L P(L|e)^2$")
ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xlabel("physical qubits $m$"); ax[1].set_ylabel("largest value over random codes")
ax[1].set_title(r"b, depolarizing $\eta=1/\sqrt{3}$", fontsize=10, loc="left"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ed2_stabilizer.pdf")); fig.savefig(os.path.join(HERE, "ed2_stabilizer.png"), dpi=150)
inside = S <= 1 + 1e-9
print("max K on/inside sphere:", K[inside].max(), " outside max K^(1/k)/kappa:", (Kk[~inside] / np.array(A["kap"])[~inside]).max())
json.dump(dict(max_K_inside=float(K[inside].max()), depol={str(m): B[m] for m in B}), open(os.path.join(HERE, "ed2_stabilizer.json"), "w"), indent=1)
