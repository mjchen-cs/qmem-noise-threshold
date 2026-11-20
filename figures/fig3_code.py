# -*- coding: utf-8 -*-
"""Figure: coding moves the threshold (chain 111-120-129-138, Wukong 180).
bare = Bell pair 111-120, rep3 = phase-flip repetition memory on the whole chain; dephasing family lambda = (a, a, 1).
Runs MS_origin_code_raw (3000 circuits / point, readout calibration 1 h later) and MS_origin_code_ro (4000 / point,
interleaved readout calibration). (a) raw counts (end to end); (b) linear tensor-product readout mitigation (memory level).
Power-law fits nu = c a^p per memory and layer (both runs pooled, weighted by the bootstrap CI), crossings nu = 1/2 with
parametric-bootstrap 95% intervals."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import nstyle  # Nature-style formatting (figures/nstyle.py)
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
# inputs from: python ../hardware/mitigate.py --run MS_origin_code_raw --ro RO_origin_ro ; python ../hardware/mitigate.py --run MS_origin_code_ro
R = {"run 1": json.load(open(os.path.join(RUNS, "MS_origin_code_raw_mitigate.json"))),
     "run 2": json.load(open(os.path.join(RUNS, "MS_origin_code_ro_mitigate.json")))}
rng = np.random.default_rng(4)
def pts(layer, kind):
    A, V, S, tag = [], [], [], []
    for rn, d in R.items():
        for k, v in d.items():
            ly, pol = k.split("_", 1)
            if ly != layer or not pol.startswith(kind + "_dz") or pol.endswith("dz100"): continue
            a = int(pol.split("dz")[1]) / 100; x = v[0]
            A.append(a); V.append(x["nu"]); S.append((x["ci"][1] - x["ci"][0]) / 3.92 / x["nu"]); tag.append(rn)
    return np.array(A), np.array(V), np.array(S), np.array(tag)
def fit(A, V, S):
    p0, c0 = np.polyfit(np.log(A), np.log(V), 1, w=1 / S)          # inflate errors by the misfit (reduced chi^2 > 1)
    chi2 = np.sum(((np.log(V) - (c0 + p0 * np.log(A))) / S) ** 2) / max(len(A) - 2, 1); S = S * np.sqrt(max(chi2, 1.0))
    def cr(lv):
        p, c = np.polyfit(np.log(A), lv, 1, w=1 / S); return np.exp((np.log(0.5) - c) / p), p, c
    x, p, c = cr(np.log(V)); bs = np.array([cr(np.log(V) + rng.normal(0, S))[0] for _ in range(5000)])
    return x, p, c, bs
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
col = {"bare": "#3b6fb6", "rep3": "#c4452c"}; mk = {"run 1": "o", "run 2": "s"}
out = {}
for i, (layer, title) in enumerate((("raw", "(a) raw counts (end to end)"), ("mit", "(b) readout-mitigated (memory level)"))):
    aa = np.linspace(0.44, 0.84, 200); xs = {}
    for kind in ("bare", "rep3"):
        A, V, S, T = pts(layer, kind)
        for rn in R:
            m = T == rn
            ax[i].errorbar(A[m] + (0.004 if rn == "run 2" else -0.004), V[m], yerr=1.96 * S[m] * V[m], fmt=mk[rn], ms=4.5,
                           color=col[kind], mfc=col[kind] if rn == "run 1" else "white", capsize=2,
                           label=f"{'unencoded' if kind == 'bare' else 'rep3'} ({rn})")
        x, p, c, bs = fit(A, V, S); lo, hi = np.percentile(bs, [2.5, 97.5]); xs[kind] = bs
        ax[i].plot(aa, np.exp(c) * aa ** p, "-", color=col[kind], lw=1.2)
        if 0.44 < x < 0.84: ax[i].plot([x], [0.5], "v", color=col[kind], ms=8)
        out[f"{layer}_{kind}"] = dict(astar=float(x), ci=[float(lo), float(hi)], p=float(p))
        print(f"{layer:3s} {kind:4s}: a* = {x:.3f} [{lo:.3f}, {hi:.3f}], exponent p = {p:.2f}")
    d = xs["bare"] - xs["rep3"]; dl, dh = np.percentile(d, [2.5, 97.5])
    out[f"{layer}_shift"] = dict(shift=float(np.median(d)), ci=[float(dl), float(dh)])
    print(f"{layer:3s} shift bare - rep3: {np.median(d):+.3f} [{dl:+.3f}, {dh:+.3f}]")
    if layer == "mit":
        ax[i].axvspan(out["mit_rep3"]["astar"], out["mit_bare"]["astar"], color="0.85", zorder=0)
        ax[i].text(0.5 * (out["mit_rep3"]["astar"] + out["mit_bare"]["astar"]), 0.27, "unencoded shrinks\nrep3 grows",
                   ha="center", fontsize=7.5)
    ax[i].axhline(0.5, color="k", lw=0.8, ls="--"); ax[i].set_xlabel(r"dephasing parameter $a$  ($\lambda = (a, a, 1)$)")
    ax[i].set_title(title); ax[i].set_xlim(0.44, 0.84); ax[i].set_ylim(0.2, 0.75)
ax[0].set_ylabel(r"per-site factor $\nu$  (advantage grows with $n$ iff $\nu > 1/2$)")
ax[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_code.pdf")); fig.savefig(os.path.join(HERE, "fig3_code.png"), dpi=200)
json.dump(out, open(os.path.join(HERE, "fig3_code_numbers.json"), "w"), indent=1)
