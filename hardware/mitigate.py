# -*- coding: utf-8 -*-
"""Per-site factor nu with and without linear readout mitigation .
Readout model: independent per-qubit confusion matrices from an RO run (|0..0> and |1..1>, raw counts, AMEND=0).
Mitigation is linear (tensor-product inverse applied to each circuit's site-marginal outcome counts, no clipping), so
the class sums S_k and counts n_k that enter nu stay unbiased; frames are recomputed from the noiseless policy with the
same treatment. Usage: python mitigate.py --run MS_origin_code_raw --ro RO_origin_ro [--B 300]"""
import os, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--ro", default=None, help="RO run; default: the ro0/ro1 records interleaved in --run (MS_RO)")
ap.add_argument("--B", type=int, default=300); a = ap.parse_args(); rng = np.random.default_rng(9)
def load(run):
    d = os.path.join(DATA, run)
    return json.load(open(os.path.join(d, "manifest.json"))), json.load(open(os.path.join(d, "raw.json")))
# ---- readout confusion
rm, rr = load(a.ro or a.run); rphys = rm["layout"]["physical"]; nr = len(rphys)
c01 = np.zeros((nr, 2, 2))                       # [qubit, prepared, read]
for r in rm["records"]:
    if r["key"] not in rr or not r["meta"]["policy"].startswith("ro"): continue
    b = int(r["meta"]["policy"][2])
    for k, v in rr[r["key"]]["counts"].items():
        x, v = int(k), int(v)
        for i in range(nr): c01[i, b, (x >> (nr - 1 - i)) & 1] += v
conf = {}
print("readout (raw):  qubit  P(1|0)   P(0|1)")
for i, q in enumerate(rphys):
    e0 = c01[i, 0, 1] / c01[i, 0].sum(); e1 = c01[i, 1, 0] / c01[i, 1].sum()
    conf[q] = np.linalg.inv(np.array([[1 - e0, e1], [e0, 1 - e1]]))      # M[true, measured]
    print(f"                {q:4d}   {e0:.4f}   {e1:.4f}")
# ---- data
man, raw = load(a.run); lay = man["layout"]; NQ = len(lay["physical"]); N = len(lay["system"])
pols = sorted({r["meta"]["policy"] for r in man["records"] if not r["meta"]["policy"].startswith("ro")})
def site_bits(j, coded):
    b = [lay["system"][j], lay["m0"][j]] + (list(lay["syndrome3"][j]) if coded else [])
    return b
def tables(pol, j, mitigate):
    coded = pol.startswith("rep3"); bits = site_bits(j, coded); L = len(bits)
    W = np.ones((1, 1))
    for q in bits:
        Mq = conf[lay["physical"][q]] if mitigate else np.eye(2)
        W = np.kron(W, Mq)                                   # W[y, x], bit order = bits order (first = most significant)
    H, Lt = [], []
    for r in man["records"]:
        if r["meta"]["policy"] != pol or r["key"] not in raw: continue
        o = np.zeros(2 ** L)
        for k, v in raw[r["key"]]["counts"].items():
            x, v = int(k), int(v)
            if v <= 0: continue
            idx = 0
            for q in bits: idx = (idx << 1) | ((x >> (NQ - 1 - q)) & 1)
            o[idx] += v
        H.append(W @ o); Lt.append(r["P"][j])
    H = np.array(H); Lt = np.array(Lt)
    y = np.arange(2 ** L); zb = (y >> (L - 1)) & 1; xb = (y >> (L - 2)) & 1
    par = np.stack([np.zeros_like(y), 1 - 2 * zb, (1 - 2 * zb) * (1 - 2 * xb), 1 - 2 * xb], 1).astype(float)   # [y, letter]
    cls = (((y >> 1) & 1) | ((y & 1) << 1)) if coded else np.zeros_like(y)   # syndrome3 bits (s0 | s1 << 1)
    return H, Lt, par, cls
def frame(pol0, j, mitigate):
    H, Lt, par, _ = tables(pol0, j, mitigate); U = np.eye(4)
    for l in (1, 2, 3):
        hh = H[Lt == l]; M = (hh @ par[:, 1:]).sum(0) / hh.sum(); thr = 3 / np.sqrt(max(len(hh), 1))
        row = M.copy()
        for k in range(3):
            if k != l - 1 and abs(row[k]) < thr: row[k] = 0.0
        U[l, 1:] = row / np.linalg.norm(row)
    return U
def nu_of(H, Lt, par, cls, U, idx):
    G = []
    for l in (1, 2, 3):
        cc = idx[Lt[idx] == l]; chi = par @ U[l]; tot = H[cc].sum(); g = 0.0
        for k in np.unique(cls):
            m = (cls == k)
            Sc = H[cc][:, m] @ chi[m]; nc = H[cc][:, m].sum(1)
            Nn = nc.sum(); den = Nn ** 2 - (nc ** 2).sum()
            if den > 0: g += (Nn / tot) * (Sc.sum() ** 2 - (Sc ** 2).sum()) / den
        G.append(max(g, 1e-6))
    return float(np.exp(np.mean(np.log(G)))), G
res = {}
for mit in (False, True):
    print(f"\n=== {'readout-mitigated (linear, tensor product)' if mit else 'raw counts'} ===")
    for pol in pols:
        kind = pol.split("_")[0]; out = []
        for j in range(N):
            H, Lt, par, cls = tables(pol, j, mit)
            U = frame(f"{kind}_dz100", j, mit) if not pol.endswith("dz100") else np.eye(4)
            allidx = np.arange(len(H)); v, G = nu_of(H, Lt, par, cls, U, allidx)
            bt = [nu_of(H, Lt, par, cls, U, rng.integers(len(H), size=len(H)))[0] for _ in range(a.B)]
            lo, hi = np.percentile(bt, [2.5, 97.5]); out.append(dict(nu=v, ci=[lo, hi], letters=G))
            print(f"  {pol:12s} site {j}: nu {v:.3f} [{lo:.3f}, {hi:.3f}]   letters X/Y/Z {'/'.join(f'{g:.2f}' for g in G)}")
        res[f"{'mit' if mit else 'raw'}_{pol}"] = out
json.dump(res, open(os.path.join(DATA, f"{a.run}_mitigate.json"), "w"), indent=1)
print(f"\nsaved data/{a.run}_mitigate.json")
