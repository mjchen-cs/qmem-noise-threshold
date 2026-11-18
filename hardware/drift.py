# -*- coding: utf-8 -*-
"""Common-mode drift check for MS runs. Per cloud job (time block), per-site visibility of the noiseless
calibration policy (bare_dz100 / rep3_dz100): v_j(t) = mean chi (letters pooled; ideal +1). If the v_j(t) of different
sites move together in time, E_t[prod v] > prod E_t[v] and the factorization ratio R_w rises above 1 without any
interaction between sites. Prints the temporal spread, the cross-site correlation and the R_w predicted by the drift.
Usage: python drift.py --mode origin_pairs [--cal bare_dz100]"""
import os, json, argparse, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
ap = argparse.ArgumentParser(); ap.add_argument("--step", default="MS"); ap.add_argument("--mode", default="origin_pairs")
ap.add_argument("--cal", default="bare_dz100"); a = ap.parse_args()
d_ = os.path.join(DATA, f"{a.step}_{a.mode}")
man = json.load(open(os.path.join(d_, "manifest.json"))); raw = json.load(open(os.path.join(d_, "raw.json")))
jobs = json.load(open(os.path.join(d_, "jobs.json")))
order = sorted(jobs, key=lambda j: jobs[j].get("submitted_unix", 0))
lay = man["layout"]; N = len(lay["system"]); NQ = len(lay["physical"])
S = {j: np.zeros(N) for j in order}; C = {j: np.zeros(N) for j in order}
for r in man["records"]:
    if r["meta"]["policy"] != a.cal or r["key"] not in raw: continue
    jid = raw[r["key"]]["job"]
    for k, v in raw[r["key"]]["counts"].items():
        x, v = int(k), int(v)
        if v <= 0: continue
        for i in range(N):
            z = 1 - 2 * ((x >> (NQ - 1 - lay["system"][i])) & 1); xx = 1 - 2 * ((x >> (NQ - 1 - lay["m0"][i])) & 1)
            l = r["P"][i]; chi = z if l == 1 else (z * xx if l == 2 else xx)
            S[jid][i] += v * chi; C[jid][i] += v
J = [j for j in order if C[j].min() > 0]
V = np.array([S[j] / C[j] for j in J])                     # jobs x sites
se = np.sqrt(np.maximum(1 - V ** 2, 0) / np.array([C[j] for j in J]))
print(f"{len(J)} time blocks (jobs), {N} sites, calibration policy {a.cal}")
print("site   mean v   temporal sd   shot-noise sd   excess sd")
for i in range(N):
    ex = np.sqrt(max(V[:, i].var() - (se[:, i] ** 2).mean(), 0))
    print(f" {i:2d}    {V[:, i].mean():.3f}     {V[:, i].std():.3f}         {np.sqrt((se[:, i] ** 2).mean()):.3f}          {ex:.3f}")
cc = np.corrcoef(V.T); off = cc[~np.eye(N, dtype=bool)]
print(f"cross-site correlation of v_j(t): mean {off.mean():+.2f} (min {off.min():+.2f}, max {off.max():+.2f})")
g = V.mean(1); print(f"site-averaged visibility over the run: first 10% {g[:len(g)//10].mean():.3f}, last 10% {g[-len(g)//10:].mean():.3f}, "
                     f"min {g.min():.3f}, max {g.max():.3f}")
print("R_w predicted by the drift alone (E_t prod v / prod E_t v, averaged over subsets):")
for w in range(1, N + 1):
    subs = list(itertools.combinations(range(N), w))[:200]
    R = np.mean([np.prod(V[:, list(s)], 1).mean() / np.prod(V[:, list(s)].mean(0)) for s in subs])
    print(f"  w={w}: {R:.3f}")
