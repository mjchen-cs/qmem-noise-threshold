# -*- coding: utf-8 -*-
"""Multi-site scaling experiment (steps MS / MSI).
Per site j and letter l: decoded per-shot value chi_j (Pauli parity of system / first memory qubit) and the syndrome class
s_j (2 bits for rep3, none for bare).  Marginal soft factor G_{j,l} = sum_s q_s lambda_s^2 (unbiased, pairs of shots from
different circuits); per-site nu_j = geometric mean over letters (as the single-site analysis); criterion nu > 1/2.
Joint test for a subset S of sites (all from the same shots): Y_c = mean_shots prod_{j in S} chi_j w_j(l_j, s_j), with
weights w = lambda estimated on the other half of the circuits (cross-fitting).  E[Y_c] = prod_j sum_s q lambda w if the
sites are independent; R_S = sum_c Y_c / sum_c M_c (M_c = that product, from marginals) is the factorization ratio (1 if
independent; < 1 = correlated loss, e.g. crosstalk).  Joint advantage over the classical limit at weight w:
log2 A_w = sum_{j in S} log2(2 nu_j) + log2 R_S  (grows with w iff nu > 1/2 and R_S does not decay).
Usage: python analyze.py [--step MS] [--mode origin] [--B 100] [--subsets 30]"""
import os, json, argparse, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
ap = argparse.ArgumentParser(); ap.add_argument("--step", default="MS"); ap.add_argument("--mode", default="origin")
ap.add_argument("--B", type=int, default=100); ap.add_argument("--subsets", type=int, default=30)
ap.add_argument("--frame", default="auto", help="frame alignment: 'auto' = use the noiseless policy rep3_dz100 of this run "
                "if present; 'none'; or a run folder name (e.g. MS_origin_cal) whose rep3_dz100 data calibrate sites by path")
a = ap.parse_args(); rng = np.random.default_rng(5)
d_ = os.path.join(DATA, f"{a.step}_{a.mode}")
man = json.load(open(os.path.join(d_, "manifest.json"))); raw = json.load(open(os.path.join(d_, "raw.json")))
lay = man["layout"]; N = len(lay["system"]); NQ = len(lay["physical"])
pol_recs = {}
for r in man["records"]:
    if r["key"] in raw and not r["meta"]["policy"].startswith("ro"):
        pol_recs.setdefault(r["meta"]["policy"], []).append(r)
def frame_from(run, cal="rep3_dz100"):
    """Per path: unit row vectors u_l (letter l -> combination of the X / Y / Z parities) from noiseless rep3_dz100 data.
    Only the direction is used (|u| = 1), so the correction can never raise a visibility above what was measured."""
    dd = os.path.join(DATA, run); mm = json.load(open(os.path.join(dd, "manifest.json"))); rr = json.load(open(os.path.join(dd, "raw.json")))
    ly = mm["layout"]; n_ = len(ly["system"]); nq = len(ly["physical"]); acc = np.zeros((n_, 4, 4)); cn = np.zeros((n_, 4)); nc_ = np.zeros((n_, 4))
    for r in mm["records"]:
        if r["meta"]["policy"] != cal or r["key"] not in rr: continue
        for j in r["meta"]["active"]: nc_[j, r["P"][j]] += 1
        for k, v in rr[r["key"]]["counts"].items():
            x, v = int(k), int(v)
            if v <= 0: continue
            for j in r["meta"]["active"]:
                z = 1 - 2 * ((x >> (nq - 1 - ly["system"][j])) & 1); xx = 1 - 2 * ((x >> (nq - 1 - ly["m0"][j])) & 1)
                l = r["P"][j]; acc[j, l, 1:] += v * np.array([z, z * xx, xx]); cn[j, l] += v
    out, rep = {}, {}
    for j in range(n_):
        M = acc[j, 1:, 1:] / np.maximum(cn[j, 1:, None], 1); u = np.eye(4)
        for l in range(3):
            row = M[l].copy(); thr = 3 / np.sqrt(max(nc_[j, l + 1], 1))     # keep only significant off-diagonal parts
            for k_ in range(3):
                if k_ != l and abs(row[k_]) < thr: row[k_] = 0.0
            nrm = np.linalg.norm(row); u[l + 1, 1:] = row / nrm if nrm > 0.05 else np.eye(3)[l]
        out[tuple(ly["paths"][j])] = u; rep[tuple(ly["paths"][j])] = M
    return out, rep, nc_
FR = {}; BROKEN = set()
if a.frame != "none":
    run = f"{a.step}_{a.mode}" if a.frame == "auto" else a.frame
    pols_run = {r["meta"]["policy"] for r in json.load(open(os.path.join(DATA, run, "manifest.json")))["records"]}
    for kind_ in ("bare", "rep3"):
        if f"{kind_}_dz100" not in pols_run: continue
        FR[kind_], REP, NC = frame_from(run, f"{kind_}_dz100")
        print(f"frame calibration from {run} ({kind_}_dz100): letter -> parity matrices, visibility |row| per letter")
        for jj, (path, M) in enumerate(REP.items()):
            thr = 3 / np.sqrt(np.maximum(NC[jj, 1:], 1))[:, None]; offd = ~np.eye(3, dtype=bool)
            Ms = np.where(offd & (np.abs(M) < thr), 0.0, M); vis = np.linalg.norm(Ms, axis=1)
            stuck = bool(abs(M[2, 2]) < 0.2 or min(vis[0], vis[1]) < 0.2 or max(abs(Ms[0, 2]), abs(Ms[1, 2]), abs(Ms[2, 0]), abs(Ms[2, 1])) > 0.3)
            if stuck: BROKEN.add(path)
            print(f"  {str(path):24s} X->[{' '.join(f'{v:+.2f}' for v in M[0])}] Y->[{' '.join(f'{v:+.2f}' for v in M[1])}] "
                  f"Z->[{' '.join(f'{v:+.2f}' for v in M[2])}]   |row| {vis[0]:.2f} {vis[1]:.2f} {vis[2]:.2f}  q={np.prod(np.minimum(vis, 1) ** 2) ** (1 / 3):.3f}"
                  + ("   BROKEN (letter-independent parities or X/Y-Z mixing)" if stuck else ""))
res = {}
for pol in sorted(pol_recs):
    recs = pol_recs[pol]; coded = pol.startswith("rep3")
    act = sorted({j for r in recs for j in r["meta"]["active"]})
    # per circuit: letters (C x N), shots expanded to a common length T (resample with replacement if a circuit has fewer)
    # amend (server readout mitigation) returns a varying number of shots per circuit: use the median T, topping up short
    # circuits by resampling their own shots (unbiased for the means and for the cross-circuit pair estimator)
    T = int(np.median([sum(max(int(v), 0) for v in raw[r["key"]]["counts"].values()) for r in recs]))
    X = []; L = []
    for r in recs:
        cnt = {int(k): int(v) for k, v in raw[r["key"]]["counts"].items() if int(v) > 0}
        sh = rng.permutation(np.array([k for k, v in cnt.items() for _ in range(v)]))
        sh = sh[:T] if len(sh) >= T else np.concatenate([sh, rng.choice(sh, T - len(sh))])
        X.append(sh); L.append(r["P"])
    X = np.array(X, dtype=np.int64); L = np.array(L); C = len(X)
    bit = lambda x, q: (x >> (NQ - 1 - q)) & 1
    par = np.zeros((C, T, N, 4)); cls = np.zeros((C, T, N), dtype=np.int64)
    for j in range(N):
        z, xx = bit(X, lay["system"][j]), bit(X, lay["m0"][j])
        par[:, :, j, 1] = 1 - 2 * z; par[:, :, j, 3] = 1 - 2 * xx; par[:, :, j, 2] = par[:, :, j, 1] * par[:, :, j, 3]
        if coded:
            s0, s1 = lay["syndrome3"][j]; cls[:, :, j] = bit(X, s0) | (bit(X, s1) << 1)
    U = np.zeros((N, 4, 4))
    for j in range(N):
        path = tuple(lay["paths"][j]); U[j] = np.eye(4)
        fk = FR.get(pol.split("_")[0], {})
        if path in fk and not pol.endswith("dz100"):
            U[j] = fk[path]
    chi = np.einsum("ctjk,cjk->ctj", par, U[np.arange(N)[None, :], L])      # frame-aligned parity of the drawn letter
    def lam_tables(idx):
        """lambda[j, l, s], q[j, l, s] and unbiased G[j, l] from circuits idx."""
        lam = np.zeros((N, 4, 4)); q = np.zeros((N, 4, 4)); G = np.full((N, 4), np.nan)
        for j in act:
            for l in (1, 2, 3):
                cc = idx[L[idx, j] == l]
                if len(cc) < 4: continue
                g = 0.0; ntot = T * len(cc)
                for s in range(4 if coded else 1):
                    m = cls[cc, :, j] == s; nc = m.sum(1); sc = (chi[cc, :, j] * m).sum(1)
                    Nn = nc.sum()
                    if Nn == 0: continue
                    lam[j, l, s] = sc.sum() / Nn; q[j, l, s] = Nn / ntot
                    den = Nn ** 2 - (nc ** 2).sum()
                    if den > 0: g += (Nn / ntot) * (sc.sum() ** 2 - (sc ** 2).sum()) / den
                G[j, l] = max(g, 1e-6)
        return lam, q, G
    allidx = np.arange(C); lam_all, q_all, G_all = lam_tables(allidx)
    nu = {j: float(np.exp(np.nanmean(np.log(G_all[j, 1:])))) for j in act}
    if a.step == "MSI" or len(act) == 1 or len(act) < N:
        bt = []
        for _ in range(a.B if len(act) <= 2 else 0):        # bootstrap over circuits (few-site runs)
            _, _, Gb = lam_tables(rng.integers(C, size=C))
            bt.append([np.exp(np.nanmean(np.log(Gb[j, 1:]))) for j in act])
        ci = {j: (list(np.percentile(np.array(bt)[:, k], [2.5, 97.5])) if bt else None) for k, j in enumerate(act)}
        let = {j: [float(G_all[j, l]) for l in (1, 2, 3)] for j in act}
        res[pol] = dict(nu_sites=nu, nu_ci=ci, letters=let)
        print(f"{pol}: nu per site " + " ".join(f"{j}:{v:.3f}" + (f" [{ci[j][0]:.3f}, {ci[j][1]:.3f}]" if ci[j] else "")
                                                for j, v in nu.items())
              + "   letters X/Y/Z " + " | ".join("/".join(f"{x:.2f}" for x in let[j]) for j in act))
        continue
    # cross-fitted per-shot weighted values V[c, t, j] = chi * w(l, s) and model means M1[c, j] = sum_s q lambda w
    perm = rng.permutation(C); halves = [perm[: C // 2], perm[C // 2:]]
    V = np.zeros((C, T, N)); M1 = np.zeros((C, N))
    for h in (0, 1):
        tr, te = halves[h], halves[1 - h]
        lam_tr, _, _ = lam_tables(tr); lam_te, q_te, _ = lam_tables(te)
        for j in range(N):
            w = lam_tr[j][L[te, j][:, None], cls[te, :, j]]                 # C_te x T
            V[te, :, j] = chi[te, :, j] * w
            M1[te, j] = (q_te[j, L[te, j]] * lam_te[j, L[te, j]] * lam_tr[j, L[te, j]]).sum(1)
    out = {}
    # bootstrap draws over circuits, shared by all w; marginals nu re-estimated in each draw so that the intervals of
    # log2 A_w include the uncertainty of the single-site factors as well as that of R_w
    BOOT = [rng.integers(C, size=C) for _ in range(a.B)]
    NUB = []
    for ci in BOOT:
        _, _, Gb = lam_tables(ci); NUB.append({j: float(np.exp(np.nanmean(np.log(Gb[j, 1:])))) for j in act})
    print(f"\n{pol}: {C} circuits x {T} shots; per-site nu (marginal): " + " ".join(f"{v:.3f}" for v in nu.values()))
    print(f"   w   subsets   R_w (factorization) [95%]      log2 A_w (joint) [95%]     marginal-only log2 A_w")
    for w in range(1, N + 1):
        subs = list(itertools.combinations(range(N), w))
        if len(subs) > a.subsets: subs = [subs[i] for i in rng.choice(len(subs), a.subsets, replace=False)]
        YS = np.array([np.prod(V[:, :, list(S)], axis=2).mean(1) for S in subs])      # n_sub x C
        MS = np.array([np.prod(M1[:, list(S)], axis=1) for S in subs])
        base = np.array([sum(np.log2(2 * nu[j]) for j in S) for S in subs])
        def stat(ci, nub):
            R = YS[:, ci].sum(1) / MS[:, ci].sum(1)
            bb = np.array([sum(np.log2(2 * nub[j]) for j in S) for S in subs])
            return float(R.mean()), float(np.mean(bb + np.log2(np.maximum(R, 1e-9))))
        R0, A0 = stat(allidx, nu)
        bt = np.array([stat(ci, nub) for ci, nub in zip(BOOT, NUB)])
        Am = float(np.mean([sum(np.log2(2 * nu[j]) for j in S) for S in subs]))
        rl, rh = np.percentile(bt[:, 0], [2.5, 97.5]); al, ah = np.percentile(bt[:, 1], [2.5, 97.5])
        out[w] = dict(R=R0, R_ci=[rl, rh], log2A=A0, log2A_ci=[al, ah], log2A_marg=Am, n_subsets=len(subs))
        print(f"  {w:2d}   {len(subs):5d}    {R0:6.3f} [{rl:6.3f}, {rh:6.3f}]       {A0:+7.3f} [{al:+7.3f}, {ah:+7.3f}]       {Am:+7.3f}")
    res[pol] = dict(nu_sites=nu, by_weight=out)
json.dump(res, open(os.path.join(DATA, f"{a.step}_summary_{a.mode}.json"), "w"), indent=1, default=float)
print(f"\nsaved data/{a.step}_summary_{a.mode}.json")
