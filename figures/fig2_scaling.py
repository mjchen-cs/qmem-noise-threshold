# -*- coding: utf-8 -*-
"""Figure: multi-site scaling on Wukong 180 (run MS_origin_pairs_raw: 9 Bell pairs, raw counts).
(a) joint log2 advantage vs number of sites w (all-shot product estimator, 95% bootstrap CI) and the product of
    marginals (lines); (b) per-site factor nu vs dephasing parameter a with the zero-parameter model
    nu_j(a) = nu_j(1) * a^(4/3) (nu_j(1) from the interleaved noiseless calibration), threshold nu = 1/2;
(c) factorization ratio R_w."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
S = json.load(open(os.path.join(DATA, "MS_summary_origin_pairs_raw.json")))   # from: python ../hardware/analyze.py --step MS --mode origin_pairs_raw --B 200
A = [0.40, 0.56, 0.64, 0.72, 0.80, 0.88]
key = lambda a: f"bare_dz{int(round(100 * a)):02d}"
cal = np.array(list(S["bare_dz100"]["nu_sites"].values()))
cm = plt.get_cmap("coolwarm")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
out = {}
for i, a in enumerate(A):
    d = S[key(a)]["by_weight"]; w = np.array(sorted(int(k) for k in d))
    y = np.array([d[str(k)]["log2A"] for k in w]); lo = np.array([d[str(k)]["log2A_ci"][0] for k in w]); hi = np.array([d[str(k)]["log2A_ci"][1] for k in w])
    ym = np.array([d[str(k)]["log2A_marg"] for k in w]); c = cm(i / (len(A) - 1))
    ax[0].errorbar(w, y, yerr=[y - lo, hi - y], fmt="o", ms=4, color=c, capsize=2, label=f"a = {a:.2f}")
    ax[0].plot(w, ym, "-", color=c, lw=1, alpha=0.7)
    R = np.array([d[str(k)]["R"] for k in w]); Rl = np.array([d[str(k)]["R_ci"][0] for k in w]); Rh = np.array([d[str(k)]["R_ci"][1] for k in w])
    ax[2].errorbar(w + (i - 2.5) * 0.06, R, yerr=[R - Rl, Rh - R], fmt="o", ms=3, color=c, capsize=1.5)
    nu = np.array(list(S[key(a)]["nu_sites"].values()))
    ax[1].plot(np.full(len(nu), a), nu, "o", ms=3.5, color=c, alpha=0.8)
    slope = np.polyfit(w, y, 1, w=1 / np.maximum(hi - lo, 1e-3))[0]
    out[a] = dict(slope_joint=float(slope), slope_marg=float(np.mean(np.log2(2 * nu))), log2A9=float(y[-1]), ci9=[float(lo[-1]), float(hi[-1])])
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylim(-11, 4); ax[0].set_xlabel("number of sites w"); ax[0].set_ylabel(r"$\log_2 A_w$ (quantum / classical limit)")
ax[0].set_title("(a) joint advantage vs system size"); ax[0].legend(fontsize=7, ncol=2)
aa = np.linspace(0.35, 1.0, 200)
for v in cal: ax[1].plot(aa, v * aa ** (4 / 3), color="0.75", lw=0.7)
ax[1].plot(aa, np.exp(np.mean(np.log(cal))) * aa ** (4 / 3), "k-", lw=1.5, label=r"model $\nu_j(1)\,a^{4/3}$ (no fit)")
ax[1].plot(aa, aa ** (4 / 3), "k--", lw=1, label="ideal hardware")
ax[1].axhline(0.5, color="r", lw=1); ax[1].text(0.36, 0.52, r"$\nu = 1/2$", color="r", fontsize=8)
astar = (0.5 / np.exp(np.mean(np.log(cal)))) ** 0.75
ax[1].axvline(astar, color="r", ls=":", lw=1); ax[1].set_xlabel(r"dephasing parameter $a$  ($\lambda = (a, a, 1)$)")
ax[1].set_ylabel(r"per-site factor $\nu$"); ax[1].set_title("(b) per-site factor, 9 Bell pairs"); ax[1].legend(fontsize=7)
ax[2].axhline(1, color="k", lw=0.8); ax[2].set_xlabel("number of sites w"); ax[2].set_ylabel(r"$R_w$ (joint / product of marginals)")
ax[2].set_title("(c) factorization"); ax[2].set_ylim(0.6, 1.6)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_scaling.pdf")); fig.savefig(os.path.join(HERE, "fig2_scaling.png"), dpi=200)
print(f"model crossing a* = {astar:.3f} (geo-mean calibration nu = {np.exp(np.mean(np.log(cal))):.3f})")
for a, v in out.items():
    print(f"a={a:.2f}: slope joint {v['slope_joint']:+.3f}, marginal {v['slope_marg']:+.3f} bits/site; log2 A_9 {v['log2A9']:+.2f} [{v['ci9'][0]:+.2f}, {v['ci9'][1]:+.2f}]")
json.dump(dict(astar=astar, rows=out), open(os.path.join(HERE, "fig2_scaling_numbers.json"), "w"), indent=1)
