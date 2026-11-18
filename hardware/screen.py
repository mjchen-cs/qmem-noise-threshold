# -*- coding: utf-8 -*-
"""Empirical screening of sites. The platform's calibration data did not predict which qubits work, so every candidate
is measured directly: candidates (4-qubit chains with --size 4, Bell pairs with --size 2) are split into groups of
disjoint, coupler-separated sites; each group is one noiseless MS run (a = 1.00) with tag _scr<g> (_scrp<g> for pairs).
Quality q = geometric mean over letters of |row|^2 of the letter -> parity response, counting off-diagonal elements
only when they exceed three standard errors.
  python screen.py --make --size 2
  python screen.py --size 2 --paths g        # sites of group g, for:  python run.py --step MS --paths <...> --a 1.00 --circ 150 --tag _scrp<g>
  python screen.py --size 2 --collect --qmin 0.7"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
ap = argparse.ArgumentParser(); ap.add_argument("--make", action="store_true"); ap.add_argument("--paths", type=int)
ap.add_argument("--collect", action="store_true"); ap.add_argument("--qmin", type=float, default=0.6)
ap.add_argument("--size", type=int, default=4, help="4: rep3 chains (RUN_TAG _scr<g>); 2: Bell pairs for bare (RUN_TAG _scrp<g>)")
ap.add_argument("--chip", default=os.path.join(DATA, "chip.json")); ap.add_argument("--mode", default="origin")
a = ap.parse_args()
c = json.load(open(a.chip)); avail = set(c["available_qubits"]); sq = {int(s["qubit"]): s for s in c["single_qubits"]}
E = {}
for d in c["double_qubits"]:
    x, y = d["qubits"]; E[(x, y)] = E[(y, x)] = d["fidelity"]
adj = {q: set() for q in avail}
for (x, y) in E:
    if x in adj and y in adj: adj[x].add(y)
def blocked(p):
    s = set(p)
    for q in p: s |= adj[q]
    return s
SUF = "" if a.size == 4 else "p"
GF = os.path.join(DATA, f"screen_groups{SUF}.json")
if a.make:
    ok = {q for q in avail if q in sq and sq[q]["gate_fidelity"] > 0.99 and sq[q]["readout_fidelity"] > 0.85 and sq[q]["t2"] > 0}
    cands = set()
    if a.size == 2:
        cands = {(x, y) for x in ok for y in adj[x] & ok if x < y and E[(x, y)] >= 0.97}
    for x in (ok if a.size == 4 else []):
        for y in adj[x] & ok:
            for z in (adj[y] & ok) - {x}:
                for w in (adj[z] & ok) - {x, y}:
                    if min(E[(x, y)], E[(y, z)], E[(z, w)]) < 0.97: continue
                    cands.add((x, y, z, w) if x < w else (w, z, y, x))   # one orientation (as path_screen.py)
    cands = sorted(cands); groups = []
    left = list(cands); rng = np.random.default_rng(0); rng.shuffle(left)
    while left:
        g, used, rest = [], set(), []
        for p in left:
            if used.isdisjoint(p) and len(g) < (16 if a.size == 4 else 24): g.append(p); used |= blocked(p)
            else: rest.append(p)
        groups.append(g); left = rest
    json.dump(dict(chip_fetched=c.get("fetched"), groups=[[list(p) for p in g] for g in groups]), open(GF, "w"), indent=1)
    print(f"{len(cands)} candidate chains in {len(groups)} groups (sizes {[len(g) for g in groups]})")
    print(len(groups))
elif a.paths is not None:
    g = json.load(open(GF))["groups"][a.paths]
    print(",".join("-".join(map(str, p)) for p in g))
elif a.collect:
    groups = json.load(open(GF))["groups"]; res = {}
    for gi in range(len(groups)):
        d_ = os.path.join(DATA, f"MS_{a.mode}_scr{SUF}{gi}")
        if not os.path.exists(os.path.join(d_, "raw.json")): continue
        mm = json.load(open(os.path.join(d_, "manifest.json"))); rr = json.load(open(os.path.join(d_, "raw.json")))
        ly = mm["layout"]; n_ = len(ly["system"]); nq = len(ly["physical"]); acc = np.zeros((n_, 4, 4)); cn = np.zeros((n_, 4)); ncl = np.zeros((n_, 4))
        for r in mm["records"]:
            if r["key"] not in rr: continue
            for j in range(n_): ncl[j, r["P"][j]] += 1
            for k, v in rr[r["key"]]["counts"].items():
                x, v = int(k), int(v)
                if v <= 0: continue
                for j in range(n_):
                    z = 1 - 2 * ((x >> (nq - 1 - ly["system"][j])) & 1); xx = 1 - 2 * ((x >> (nq - 1 - ly["m0"][j])) & 1)
                    l = r["P"][j]; acc[j, l, 1:] += v * np.array([z, z * xx, xx]); cn[j, l] += v
        for j in range(n_):
            M = acc[j, 1:, 1:] / np.maximum(cn[j, 1:, None], 1)
            thr = 3 / np.sqrt(np.maximum(ncl[j, 1:], 1))[:, None]; off = ~np.eye(3, dtype=bool)
            Ms = np.where(off & (np.abs(M) < thr), 0.0, M); vis = np.linalg.norm(Ms, axis=1)
            zmix = max(abs(Ms[0, 2]), abs(Ms[1, 2]), abs(Ms[2, 0]), abs(Ms[2, 1]))
            broken = bool(abs(M[2, 2]) < 0.2 or min(vis[0], vis[1]) < 0.2 or zmix > 0.3)
            res[tuple(ly["paths"][j])] = dict(q=float(np.prod(np.minimum(vis, 1) ** 2) ** (1 / 3)), vis=vis.tolist(), broken=bool(broken), group=gi)
    good = sorted([(p, v) for p, v in res.items() if not v["broken"]], key=lambda kv: -kv[1]["q"])
    print(f"measured {len(res)} oriented chains from {len({v['group'] for v in res.values()})} groups; "
          f"{sum(v['broken'] for v in res.values())} broken")
    print(" top chains (s-m1-m2-m3)      q     |row| X Y Z")
    for p, v in good[:30]:
        print(f"  {str(p):26s} {v['q']:.3f}   {' '.join(f'{x:.2f}' for x in v['vis'])}")
    chosen, used = [], set()
    for p, v in good:
        if v["q"] < a.qmin: break
        if used.isdisjoint(p): chosen.append(p); used |= blocked(p)
    print(f"\ndisjoint, buffered set with q >= {a.qmin}: {len(chosen)} chains")
    print("--paths " + ",".join("-".join(map(str, p)) for p in chosen))
    json.dump(dict(chains={"-".join(map(str, p)): v for p, v in res.items()}, chosen=[list(p) for p in chosen]),
              open(os.path.join(DATA, f"screen_result{SUF}.json"), "w"), indent=1)
