# -*- coding: utf-8 -*-
"""Figure: the coding gain and the noise bias (chain 111-120-129-138, Wukong 180, readout-mitigated = memory level).
Runs MS_origin_bias2 (lambda_z = 1.00, 0.98) and MS_origin_bias (0.95, 0.90), 19 Nov, 4,000 circuits per point, interleaved
readout calibration; and the 18 Nov runs at lambda_z = 1 (MS_origin_code_raw, MS_origin_code_ro).
Injected noise lambda = (t, t, lambda_z) on every memory qubit. (a-d) per-site factor of the unencoded (Bell pair) and the
repetition-code memory with power-law fits nu = c t^p; (e) shift of the crossing nu = 1/2 by the code, measured and
predicted: ideal processor (exact, theory/memory_factor.py), processor factors nu(1) from the noiseless circuits, and the
same with an additional 12% loss of the coded memory in noisy circuits (phenomenological). Vertical lines: crossing of an
ideal Bell pair on a perfect processor in the standard memory frame (dotted, nu = (lx ly lz)^(2/3)) and in the best memory
frame (dash-dot, nu = ((lx+ly+lz)/3)^2, the largest value of Bell sampling in any frame on any processor); thin dashed: the
bound sum(lambda^2) = 3/2 for any protocol with a Bell input."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import nstyle  # Nature-style formatting (figures/nstyle.py)
import matplotlib.pyplot as plt
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "theory"))
from memory_factor import nu as nu_ideal
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data")); rng = np.random.default_rng(4)
J = lambda f: json.load(open(os.path.join(DATA, f)))
B2, B1 = J("MS_origin_bias2_mitigate.json"), J("MS_origin_bias_mitigate.json")
OLD = {"run 1": J("MS_origin_code_raw_mitigate.json"), "run 2": J("MS_origin_code_ro_mitigate.json")}
SETS = [(1.00, "dz", B2), (0.98, "zs", B2), (0.95, "zh", B1), (0.90, "zm", B1)]
def pts(d, layer, kind, fam):
    A, V, S = [], [], []
    for k, v in d.items():
        ly, pol = k.split("_", 1)
        if ly == layer and pol.startswith(f"{kind}_{fam}") and not pol.endswith("dz100"):
            A.append(int(pol.split(fam)[1]) / 100); V.append(v[0]["nu"]); S.append((v[0]["ci"][1] - v[0]["ci"][0]) / 3.92 / v[0]["nu"])
    o = np.argsort(A); return np.array(A)[o], np.array(V)[o], np.array(S)[o]
def fit(A, V, S):
    p0, c0 = np.polyfit(np.log(A), np.log(V), 1, w=1 / S)
    chi2 = np.sum(((np.log(V) - (c0 + p0 * np.log(A))) / S) ** 2) / max(len(A) - 2, 1); S = S * np.sqrt(max(chi2, 1.0))
    def cr(lv):
        p, c = np.polyfit(np.log(A), lv, 1, w=1 / S); return np.exp((np.log(0.5) - c) / p), p, c
    x, p, c = cr(np.log(V)); bs = np.array([cr(np.log(V) + rng.normal(0, S))[0] for _ in range(4000)]); return x, p, c, bs
col = {"bare": "#3b6fb6", "rep3": "#c4452c"}
fig = plt.figure(figsize=(12.5, 5.0)); gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.12, 1.55])
OUT = {"measured": {}}
from memory_factor import crossing as ideal_cross
for i, (lz, fam, d) in enumerate(SETS):
    ax = fig.add_subplot(gs[i // 2, i % 2]); xs = {}
    tb = ideal_cross("bare", fam)                     # ideal Bell pair, standard frame: nu = (lx ly lz)^(2/3)
    to = (3 / np.sqrt(2) - lz) / 2                     # Bell sampling, best memory frame: nu = ((2t+lz)/3)^2 = 1/2
    tbd = np.sqrt((1.5 - lz ** 2) / 2)                 # bound for any Bell input: sum(lambda^2) = 3/2
    ax.axvline(tb, color="0.45", ls=":", lw=1.1, label="Bell, standard frame" if i == 0 else None)
    ax.axvline(to, color="0.2", ls="-.", lw=1.0, label="Bell, best frame" if i == 0 else None)
    ax.axvline(tbd, color="0.6", ls="--", lw=0.6, label=r"Bell-input bound ($\sum\lambda^2=3/2$)" if i == 0 else None)
    for kind in ("bare", "rep3"):
        A, V, S = pts(d, "mit", kind, fam); x, p, c, bs = fit(A, V, S); xs[kind] = bs
        ax.errorbar(A, V, yerr=1.96 * S * V, fmt="o", ms=3.5, capsize=1.5, color=col[kind], label="unencoded" if kind == "bare" else "rep3")
        aa = np.linspace(0.46, 0.86, 100); ax.plot(aa, np.exp(c) * aa ** p, "-", color=col[kind], lw=1)
        if 0.46 < x < 0.86: ax.plot([x], [0.5], "v", color=col[kind], ms=6)
        OUT["measured"].setdefault(str(lz), {})[kind] = dict(astar=float(x), ci=list(np.percentile(bs, [2.5, 97.5])), p=float(p))
    xr = OUT["measured"][str(lz)]["rep3"]["astar"]
    OUT["measured"][str(lz)]["ideal_bell_standard"] = float(tb); OUT["measured"][str(lz)]["ideal_bell_bestframe"] = float(to); OUT["measured"][str(lz)]["bell_input_bound"] = float(tbd)
    OUT["measured"][str(lz)]["p_rep3_below_standard"] = float(np.mean(xs["rep3"] < tb))
    OUT["measured"][str(lz)]["p_rep3_below_bestframe"] = float(np.mean(xs["rep3"] < to))
    OUT["measured"][str(lz)]["p_rep3_below_bound"] = float(np.mean(xs["rep3"] < tbd))
    dd = xs["bare"] - xs["rep3"]; OUT["measured"][str(lz)]["shift"] = dict(median=float(np.median(dd)), ci=list(np.percentile(dd, [2.5, 97.5])))
    ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_xlim(0.46, 0.86); ax.set_ylim(0.3, 0.72)
    ax.set_title(f"{'abcd'[i]}   $\\lambda_z$ = {lz:.2f}", loc="left", fontsize=9)
    if i % 2 == 0: ax.set_ylabel(r"$\nu$")
    if i // 2 == 1: ax.set_xlabel(r"$t$   ($\lambda=(t,t,\lambda_z)$)")
    else: ax.set_xticklabels([])
    if i == 0: H_, L_ = ax.get_legend_handles_labels()
# 18 Nov, lambda_z = 1 (both runs pooled, as in the earlier analysis)
A, V, S = [], [], []
for d in OLD.values():
    for kind in ("bare", "rep3"): pass
xs = {}
for kind in ("bare", "rep3"):
    A, V, S = [np.concatenate(z) for z in zip(*[pts(d, "mit", kind, "dz") for d in OLD.values()])]
    xs[kind] = fit(A, V, S)[3]
OUT["p18_rep3_below_standard"] = float(np.mean(xs["rep3"] < ideal_cross("bare", "dz"))); OUT["p18_rep3_below_bestframe"] = float(np.mean(xs["rep3"] < (3 / np.sqrt(2) - 1) / 2)); OUT["p18_rep3_below_bound"] = float(np.mean(xs["rep3"] < 0.5))
dd18 = xs["bare"] - xs["rep3"]; OUT["measured_18Nov"] = dict(median=float(np.median(dd18)), ci=list(np.percentile(dd18, [2.5, 97.5])))
# predictions
f_b = np.mean([B2["mit_bare_dz100"][0]["nu"], B1["mit_bare_dz100"][0]["nu"]]); f_r = np.mean([B2["mit_rep3_dz100"][0]["nu"], B1["mit_rep3_dz100"][0]["nu"]])
def cross(kind, lz, fac):
    f = lambda t: fac * nu_ideal(kind, (t, t, lz)) - 0.5
    return brentq(f, 0.05, 1.0) if f(0.05) * f(1.0) < 0 else np.nan
LZ = np.linspace(0.88, 1.0, 25); pred = {}
for name, fb, fr in (("ideal processor", 1.0, 1.0), (r"factors $\nu(1)$ from noiseless circuits", f_b, f_r), ("same, rep3 x 0.88 in noisy circuits", f_b, 0.88 * f_r)):
    pred[name] = [cross("bare", lz, fb) - cross("rep3", lz, fr) for lz in LZ]
ax = fig.add_subplot(gs[:, 3])
for (name, y), ls in zip(pred.items(), ("-", "--", ":")):
    ax.plot(LZ, y, ls, color="0.35", lw=1.2, label=name)
m = OUT["measured"]; L = np.array([float(k) for k in m]); y = np.array([m[k]["shift"]["median"] for k in m])
lo = np.array([m[k]["shift"]["ci"][0] for k in m]); hi = np.array([m[k]["shift"]["ci"][1] for k in m])
ax.errorbar(L, y, yerr=[y - lo, hi - y], fmt="o", ms=6, color="#8a2be2", capsize=3, label="measured, 19 Nov")
o = OUT["measured_18Nov"]; ax.errorbar([1.003], [o["median"]], yerr=[[o["median"] - o["ci"][0]], [o["ci"][1] - o["median"]]], fmt="s", ms=5,
                                         mfc="white", color="#8a2be2", capsize=3, label="measured, 18 Nov (two runs)")
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(r"bias $\lambda_z$ of the memory noise $(t,t,\lambda_z)$")
ax.set_ylabel(r"shift of the crossing by the code, $t^*_{\rm bare}-t^*_{\rm rep3}$"); ax.set_xlim(0.885, 1.01); ax.set_ylim(-0.27, 0.25)
ax.set_title("e", loc="left", fontweight="bold"); ax.legend(fontsize=7, loc="lower right")
fig.legend(H_, L_, loc="upper center", bbox_to_anchor=(0.3, 0.0), ncol=5, fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_bias.pdf")); fig.savefig(os.path.join(HERE, "fig4_bias.png"), dpi=200)
OUT["factors"] = dict(bare=float(f_b), rep3=float(f_r)); OUT["pred"] = {k: dict(zip([f"{z:.3f}" for z in LZ], map(float, v))) for k, v in pred.items()}
lzc = [float(k) for k in m]; ys = [m[k]["shift"]["median"] for k in m]
o_ = np.argsort(lzc); OUT["lz_zero"] = float(np.interp(0, np.array(ys)[o_][::-1] * -1 * -1 if False else np.array(ys)[o_], np.array(lzc)[o_]))
json.dump(OUT, open(os.path.join(HERE, "fig4_bias_numbers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "pred"}, indent=1))
for k, v in pred.items(): print(k, {f"{z:.2f}": round(x, 3) for z, x in zip(LZ[::6], v[::6])})
