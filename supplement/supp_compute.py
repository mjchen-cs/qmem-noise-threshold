# -*- coding: utf-8 -*-
"""Supplementary analysis: full result tables and robustness checks. Writes supp_results.json.
Usage: DATA_DIR=<figshare folder> python supp_compute.py"""
import os, json
import numpy as np
from supp_core import *
rng = np.random.default_rng(11); B = 500; res = {}
# ======================================================== A. nine Bell pairs, raw counts: per site, per a, per letter
run = Run("MS_origin_pairs_raw"); pols = run.policies("bare"); A_ = sorted(a_of(p) for p in pols)
key = lambda a: f"bare_dz{int(round(100 * a)):02d}"
U = {}; site = {}
for j in range(run.N):
    H0, L0, coded, nb = run.hist("bare_dz100", j); U[j] = frame_U(H0, L0, nb, coded)
for pol in pols:
    for j in range(run.N):
        H, Lt, coded, nb = run.hist(pol, j); Uj = np.eye(4) if pol.endswith("dz100") else U[j]
        r = site_nu(H, Lt, nb, coded, Uj, B=B, rng=rng)
        site[(pol, j)] = dict(nu=r["nu"], ci=list(np.percentile(r["boot"], [2.5, 97.5])), sd=float(np.std(r["boot"])),
                              G=r["G"], G_ci=[list(np.percentile(g, [2.5, 97.5])) for g in r["Gboot"]],
                              n_circ=[int((Lt == l).sum()) for l in (1, 2, 3)])
nu1 = np.array([site[("bare_dz100", j)]["nu"] for j in range(run.N)])
rows = []
for pol in pols:
    a = a_of(pol)
    for j in range(run.N):
        s = site[(pol, j)]; m = nu1[j] * a ** (4 / 3)
        rows.append(dict(a=a, site=j, path=run.lay["paths"][j], **s, model=float(m), z=float((s["nu"] - m) / s["sd"]) if a < 1 else None))
res["pairs_sites"] = rows
chi2 = [r["z"] ** 2 for r in rows if r["z"] is not None]
res["pairs_model_chi2"] = dict(chi2=float(np.sum(chi2)), dof=len(chi2), rms_z=float(np.sqrt(np.mean(chi2))),
                               per_a={str(a): float(np.sqrt(np.mean([r["z"] ** 2 for r in rows if r["a"] == a]))) for a in A_ if a < 1})
print("A done; model rms z", res["pairs_model_chi2"]["rms_z"])
# ======================================================== B. joint estimator: baseline + robustness
def full(frame=True, crossfit=True, sites=None, idx_fn=None, seed=5, Bj=200, n_subsets=30):
    out = {}
    for pol in pols:
        chi, L, jr = CH[(pol, frame)]
        idx = idx_fn(jr) if idx_fn else None
        out[pol] = joint(chi, L, sites=sites, crossfit=crossfit, B=Bj, seed=seed, idx=idx, n_subsets=n_subsets)
    noisy = [p for p in pols if not p.endswith("dz100")]
    sl = {p: slope(out[p]["by_weight"]) for p in pols}
    ast = zero_crossing([a_of(p) for p in noisy], [sl[p] for p in noisy])
    return out, sl, ast
CH = {(p, f): per_shot(run, p, frame=f) for p in pols for f in (True, False)}
def summ(tag, out, sl, ast, sites=None):
    n1 = np.array(out["bare_dz100"]["nu"])
    d = dict(astar_joint=ast, astar_model=model_astar(n1), nu1_geo=float(np.exp(np.mean(np.log(n1)))),
             slopes={str(a_of(p)): sl[p] for p in pols},
             R_last={str(a_of(p)): [out[p]["by_weight"][max(out[p]["by_weight"])]["R"]] + out[p]["by_weight"][max(out[p]["by_weight"])]["R_ci"] for p in pols})
    print(f"{tag:28s} a*_joint {ast:.3f}  a*_model {d['astar_model']:.3f}  R_last(a=1) {d['R_last']['1.0'][0]:.3f} "
          f"[{d['R_last']['1.0'][1]:.3f},{d['R_last']['1.0'][2]:.3f}]")
    return d
base, sl0, a0 = full(); res["pairs_joint"] = {p: base[p]["by_weight"] for p in pols}
res["rob_baseline"] = summ("baseline", base, sl0, a0)
o = full(frame=False); res["rob_noframe"] = summ("no frame alignment", *o)
res["rob_noframe"]["nu"] = {str(a_of(p)): o[0][p]["nu"] for p in pols}
o = full(crossfit=False); res["rob_nocrossfit"] = summ("no cross-fitting", *o)
res["rob_nocrossfit"]["R"] = {str(a_of(p)): {w: v["R"] for w, v in o[0][p]["by_weight"].items()} for p in pols}
res["rob_loo"] = {}
for k in range(run.N):
    o = full(sites=[j for j in range(run.N) if j != k], Bj=100); res["rob_loo"][str(k)] = summ(f"leave out site {k}", *o)
res["rob_halves"] = {}
for h, fn in (("first", lambda jr: np.where(jr <= np.median(jr))[0]), ("second", lambda jr: np.where(jr > np.median(jr))[0])):
    o = full(idx_fn=fn, Bj=100); d = summ(f"{h} half (submission order)", *o)
    d["nu"] = {str(a_of(p)): o[0][p]["nu"] for p in pols}; res["rob_halves"][h] = d
res["rob_seeds"] = {}
for sd in (1, 2, 3, 4, 5, 6):
    o = full(seed=sd, Bj=100); res["rob_seeds"][str(sd)] = summ(f"subset/half seed {sd}", *o)
o = full(n_subsets=126, Bj=100); res["rob_allsubsets"] = summ("126 subsets per w", *o)
# amend (platform readout correction) for comparison
Sm = json.load(open(os.path.join(DATA, "MS_summary_origin_pairs.json"))); res["pairs_amend_joint"] = {p: Sm[p]["by_weight"] for p in Sm if "by_weight" in Sm[p]}
res["pairs_amend_nu"] = {p: Sm[p]["nu_sites"] for p in Sm}
# ======================================================== C. code runs: raw and linear readout mitigation, class tables
res["code"] = {}
for rn, ro in (("MS_origin_code_raw", "RO_origin_ro"), ("MS_origin_code_ro", "MS_origin_code_ro")):
    cr = Run(rn); conf, err = readout_conf(Run(ro)); res["code"][rn] = dict(readout={str(q): e for q, e in err.items()}, pts=[])
    for layer, cf in (("raw", None), ("mit", conf)):
        Uk = {}
        for k in ("bare", "rep3"):
            H0, L0, c0, nb0 = cr.hist(f"{k}_dz100", 0, cf); Uk[k] = frame_U(H0, L0, nb0, c0)
        for pol in cr.policies():
            H, Lt, coded, nb = cr.hist(pol, 0, cf); k = pol.split("_")[0]
            r = site_nu(H, Lt, nb, coded, np.eye(4) if pol.endswith("dz100") else Uk[k], B=B, rng=rng)
            res["code"][rn]["pts"].append(dict(layer=layer, kind=k, a=a_of(pol), nu=r["nu"], ci=list(np.percentile(r["boot"], [2.5, 97.5])),
                                               G=r["G"], G_ci=[list(np.percentile(g, [2.5, 97.5])) for g in r["Gboot"]], tabs=r["tabs"],
                                               n_circ=int(len(H)), shots=float(H.sum())))
    print(rn, "done")
json.dump(res, open(os.path.join(HERE, "supp_results.json"), "w"), indent=1, default=float)
print("saved supp_results.json")
