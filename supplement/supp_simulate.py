# -*- coding: utf-8 -*-
"""Validation of the analysis pipeline on simulated data with the design of the nine-pair experiment (same letters per
circuit, 40 shots, 4,000 circuits per setting). Ground truth: independent sites with per-letter visibilities
mu_{j,l}(a) = f_{j,l} lambda_l(a), f_{j,l} = sqrt(G_{j,l}(1)) from the measured noiseless data, lambda = (a, a, 1), so that
nu_j(a) = nu_j(1) a^(4/3) exactly. Tests: (V1) bias and CI coverage of nu_j, R_w and log2 A_w; (V2) response of R_w to a
common-mode modulation of all visibilities, mu -> mu (1 + eps_c), eps_c = +-delta per circuit (drift); (V3) sampling
distribution of the reversal point a* from the joint slopes. Usage: python supp_simulate.py --test V1|V2|V3 --reps R --seed S   (several seeds, then --test summary)"""
import os, json, argparse
import numpy as np
from supp_core import *
ap = argparse.ArgumentParser(); ap.add_argument("--test", default="V1"); ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0); a_ = ap.parse_args(); rng = np.random.default_rng(1000 + a_.seed)
if a_.test == "summary":                                   # aggregate sim_V*_<seed>.json into sim_summary.json
    import glob
    ld = lambda t: sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, f"sim_{t}_[0-9].json")))], [])
    V1, V2, V3 = ld("V1"), ld("V2"), ld("V3"); out = {}
    for a in (1.0, 0.64):
        r = [x for x in V1 if x["a"] == a]; nu = np.array([x["nu"] for x in r]); tr = np.array(r[0]["nu_true"])
        out[str(a)] = dict(n=len(r), maxbias=float(np.abs(100 * (nu.mean(0) / tr - 1)).max()), nucov=float(np.mean([x["nu_cover"] for x in r])),
                           R9=float(np.mean([x["R9"] for x in r])), R9sd=float(np.std([x["R9"] for x in r])), R9cov=float(np.mean([x["R9_cover"] for x in r])),
                           A9bias=float(np.mean([x["A9"] - x["A9_true"] for x in r])), A9sd=float(np.std([x["A9"] for x in r])),
                           A9cov=float(np.mean([x["A9_cover"] for x in r])), relsd=float(np.mean(nu.std(0) / tr)))
    s3 = np.array([x["astar"] for x in V3]); out["V3"] = dict(n=len(s3), mean=float(s3.mean()), sd=float(s3.std()), ci=list(np.percentile(s3, [2.5, 97.5])))
    out["V2"] = {str(d): dict(R9=float(np.mean([x["R"]["9"] for x in V2 if x["delta"] == d])), sd=float(np.std([x["R"]["9"] for x in V2 if x["delta"] == d])),
                              true=((1 + d) ** 9 + (1 - d) ** 9) / 2) for d in (0.02, 0.05, 0.1)}
    json.dump(out, open(os.path.join(HERE, "sim_summary.json"), "w"), indent=1); print(json.dumps(out, indent=1)); raise SystemExit
R = json.load(open(os.path.join(HERE, "supp_results.json")))
f = np.zeros((9, 4)); f[:, 0] = 1
for r in R["pairs_sites"]:
    if r["a"] == 1.0: f[r["site"], 1:] = np.sqrt(r["G"])
run = Run("MS_origin_pairs_raw"); T = 40
def lam(a): return np.array([0, a, a, 1])          # letters: 1 X, 2 Y, 3 Z; lambda = (a, a, 1)
def simulate(pol, a, delta=0.0):
    _, _, _, _, L, jr = run.arrays(pol); C = len(L)
    mu = f[np.arange(9)[None, :], L] * lam(a)[L]                                  # C x 9
    if delta: mu = mu * (1 + delta * rng.choice([-1, 1], size=C))[:, None]
    chi = np.where(rng.random((C, T, 9)) < (1 + mu[:, None, :]) / 2, 1.0, -1.0)
    return chi, L, jr
def nus_of(chi, L, B=0):
    out = []
    for j in range(9):
        c = chi[:, :, j]; lj = L[:, j][:, None]; r1 = rng.choice([-1, 1], size=c.shape); r2 = rng.choice([-1, 1], size=c.shape)
        z = np.where(lj == 1, c, r1); x = np.where(lj == 1, r2, np.where(lj == 2, z * c, c))   # bits consistent with the letter
        y = ((1 - z) // 2).astype(int) * 2 + ((1 - x) // 2).astype(int)
        H = np.stack([(y == k).sum(1) for k in range(4)], 1).astype(float)
        r = site_nu(H, L[:, j], 2, False, np.eye(4), B=B, rng=rng); out.append((r["nu"], np.percentile(r["boot"], [2.5, 97.5]) if B else None))
    return out
res = []
if a_.test == "V1":
    for rep in range(a_.reps):
        for pol in ("bare_dz100", "bare_dz64"):
            a = a_of(pol); chi, L, _ = simulate(pol, a); tr = np.sqrt(np.prod(f[:, 1:] ** 2 * lam(a)[1:] ** 2, 1)) ** (2 / 3)
            ns = nus_of(chi, L, B=200); jt = joint(chi, L, B=200, seed=rep)
            b9 = jt["by_weight"][9]; t9 = float(np.sum(np.log2(2 * tr)))
            res.append(dict(a=a, nu=[n for n, _ in ns], nu_true=list(tr), nu_cover=[bool(c[0] <= t <= c[1]) for (_, c), t in zip(ns, tr)],
                            R9=b9["R"], R9_cover=bool(b9["R_ci"][0] <= 1 <= b9["R_ci"][1]), A9=b9["log2A"], A9_true=t9,
                            A9_cover=bool(b9["log2A_ci"][0] <= t9 <= b9["log2A_ci"][1])))
elif a_.test == "V2":
    for rep in range(a_.reps):
        for d in (0.02, 0.05, 0.10):
            chi, L, _ = simulate("bare_dz100", 1.0, delta=d); jt = joint(chi, L, B=100, seed=rep)
            res.append(dict(delta=d, R={w: v["R"] for w, v in jt["by_weight"].items()},
                            R_true={w: ((1 + d) ** w + (1 - d) ** w) / 2 for w in range(1, 10)}))
elif a_.test == "V3":
    pols = run.policies("bare")
    for rep in range(a_.reps):
        sl = {}
        for pol in pols:
            chi, L, _ = simulate(pol, a_of(pol)); sl[pol] = slope(joint(chi, L, B=40, seed=rep)["by_weight"])
        noisy = [p for p in pols if not p.endswith("dz100")]
        res.append(dict(astar=zero_crossing([a_of(p) for p in noisy], [sl[p] for p in noisy]), slopes=sl))
json.dump(res, open(os.path.join(HERE, f"sim_{a_.test}_{a_.seed}.json"), "w"), default=float)
print("done", a_.test, a_.seed, len(res))
