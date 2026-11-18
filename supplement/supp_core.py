# -*- coding: utf-8 -*-
"""Shared estimators for the supplementary analysis (same definitions as hardware/analyze.py and hardware/mitigate.py,
vectorised so that bootstrap, leave-one-out and split-half analyses are fast).

Per site j and letter l in {1: X, 2: Y, 3: Z} (Pauli labels in the frame of the injected noise lambda = (a, a, 1)): chi =
frame-aligned parity of the drawn letter; syndrome class s (rep3 only). Soft factor G_{j,l} = sum_s q_s lambda_s^2, with lambda_s^2 estimated without
bias from pairs of shots in different circuits; nu_j = (G_X G_Y G_Z)^(1/3)."""
import os, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
LET = {1: "X", 2: "Y", 3: "Z"}          # letter index -> Pauli in the frame of the injected noise lambda = (a, a, 1)

def a_of(pol):
    return int(pol.split("dz")[1]) / 100

class Run:
    """One experiment folder: manifest, raw counts, jobs (submission order)."""
    def __init__(self, name):
        d = os.path.join(DATA, name); self.name = name
        self.man = json.load(open(os.path.join(d, "manifest.json"))); self.raw = json.load(open(os.path.join(d, "raw.json")))
        jp = os.path.join(d, "jobs.json"); self.jobs = json.load(open(jp)) if os.path.exists(jp) else {}
        self.lay = self.man["layout"]; self.N = len(self.lay.get("system", [])); self.NQ = len(self.lay["physical"])
        order = sorted(self.jobs, key=lambda j: self.jobs[j].get("submitted_unix", 0)); self.job_rank = {j: i for i, j in enumerate(order)}
        self._pa = {}
    def policies(self, kind=None):
        p = sorted({r["meta"]["policy"] for r in self.man["records"] if r["key"] in self.raw and not r["meta"]["policy"].startswith("ro")})
        return [x for x in p if kind is None or x.startswith(kind + "_")]
    def arrays(self, pol):
        """recs, flat (circuit, outcome, count), letters L[c, j], job rank per circuit."""
        if pol in self._pa: return self._pa[pol]
        recs = [r for r in self.man["records"] if r["meta"]["policy"] == pol and r["key"] in self.raw]
        ci, x, v = [], [], []
        for c, r in enumerate(recs):
            for k, n in self.raw[r["key"]]["counts"].items():
                n = int(n)
                if n > 0: ci.append(c); x.append(int(k)); v.append(n)
        L = np.array([r["P"] if len(r["P"]) == self.N else r["P"] + [0] * (self.N - len(r["P"])) for r in recs])
        jr = np.array([self.job_rank.get(self.raw[r["key"]].get("job"), -1) for r in recs])
        out = (recs, np.array(ci), np.array(x, dtype=np.int64), np.array(v, dtype=float), L, jr); self._pa[pol] = out
        return out
    def bit(self, x, q):
        return (x >> (self.NQ - 1 - q)) & 1
    def site_bits(self, j, coded):
        return [self.lay["system"][j], self.lay["m0"][j]] + (list(self.lay["syndrome3"][j]) if coded else [])
    def hist(self, pol, j, conf=None):
        """H[c, y]: counts of site-j outcome y (bits system, m0 [, s0, s1], first = most significant); optional linear
        readout mitigation with per-physical-qubit inverse confusion matrices conf[q] (no clipping)."""
        recs, ci, x, v, L, jr = self.arrays(pol); coded = pol.startswith("rep3"); bits = self.site_bits(j, coded); nb = len(bits)
        y = np.zeros_like(x)
        for q in bits: y = (y << 1) | self.bit(x, q)
        H = np.zeros((len(recs), 2 ** nb)); np.add.at(H, (ci, y), v)
        if conf is not None:
            W = np.ones((1, 1))
            for q in bits: W = np.kron(W, conf[self.lay["physical"][q]])
            H = H @ W.T
        return H, L[:, j], coded, nb

def parities(nb, coded):
    y = np.arange(2 ** nb); zb = (y >> (nb - 1)) & 1; xb = (y >> (nb - 2)) & 1
    par = np.stack([np.zeros_like(y), 1 - 2 * zb, (1 - 2 * zb) * (1 - 2 * xb), 1 - 2 * xb], 1).astype(float)
    cls = (((y >> 1) & 1) | ((y & 1) << 1)) if coded else np.zeros_like(y)
    return par, cls

def frame_U(H0, L0, nb, coded):
    """Letter -> parity directions from noiseless circuits (direction only; off-diagonal kept if > 3 s.e.)."""
    par, _ = parities(nb, coded); U = np.eye(4)
    for l in (1, 2, 3):
        hh = H0[L0 == l]
        if len(hh) == 0: continue
        M = (hh @ par[:, 1:]).sum(0) / hh.sum(); thr = 3 / np.sqrt(max(len(hh), 1)); row = M.copy()
        for k in range(3):
            if k != l - 1 and abs(row[k]) < thr: row[k] = 0.0
        n = np.linalg.norm(row); U[l, 1:] = row / n if n > 0.05 else np.eye(3)[l - 1]
    return U

def class_sums(H, Lt, nb, coded, U):
    """Per letter: sc[c, s] (sum of chi in class s) and nc[c, s] for the circuits of that letter."""
    par, cls = parities(nb, coded); ns = 4 if coded else 1; out = {}
    for l in (1, 2, 3):
        m = Lt == l; chi = par @ U[l]; hh = H[m]
        sc = np.stack([hh[:, cls == s] @ chi[cls == s] for s in range(ns)], 1)
        nc = np.stack([hh[:, cls == s].sum(1) for s in range(ns)], 1)
        out[l] = (np.where(m)[0], sc, nc)
    return out

def G_from(sc, nc, w=None):
    """Unbiased soft factor from circuit sums; w = circuit multiplicities (bootstrap), shape (B, C) or None."""
    if w is None: w = np.ones((1, len(sc)))
    S = w @ sc; S2 = w @ sc ** 2; Nn = w @ nc; N2 = w @ nc ** 2; tot = Nn.sum(1, keepdims=True)
    den = Nn ** 2 - N2
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(den > 0, (Nn / tot) * (S ** 2 - S2) / den, 0.0)
    return np.maximum(g.sum(1), 1e-6)

def boot_weights(C, B, rng):
    return rng.multinomial(C, np.full(C, 1 / C), size=B).astype(float) if B else None

def site_nu(H, Lt, nb, coded, U, B=0, rng=None, idx=None):
    """nu, letters (G_Z, G_Y, G_X), bootstrap draws of nu (over circuits), also the class tables q[l, s], lambda[l, s]."""
    if idx is not None: H, Lt = H[idx], Lt[idx]
    cs = class_sums(H, Lt, nb, coded, U); C = len(H)
    Wb = boot_weights(C, B, rng) if B else None
    G = {}; Gb = {}; tabs = {}
    for l, (ii, sc, nc) in cs.items():
        G[l] = float(G_from(sc, nc)[0])
        if B: Gb[l] = G_from(sc, nc, Wb[:, ii])
        tabs[l] = dict(q=list(nc.sum(0) / nc.sum()), lam=list(np.divide(sc.sum(0), nc.sum(0), out=np.zeros(sc.shape[1]), where=nc.sum(0) > 0)))
    nu = float(np.exp(np.mean(np.log([G[1], G[2], G[3]]))))
    nub = np.exp(np.mean(np.log(np.stack([Gb[1], Gb[2], Gb[3]])), 0)) if B else None
    return dict(nu=nu, G=[G[1], G[2], G[3]], boot=nub, tabs=tabs,
                Gboot=[Gb[1], Gb[2], Gb[3]] if B else None)

def readout_conf(run, pol_prefix="ro"):
    """Inverse per-qubit confusion matrices from ro0 / ro1 records."""
    rm, rr = run.man, run.raw; phys = rm["layout"]["physical"]; nr = len(phys); c01 = np.zeros((nr, 2, 2))
    for r in rm["records"]:
        if r["key"] not in rr or not r["meta"]["policy"].startswith(pol_prefix): continue
        b = int(r["meta"]["policy"][2])
        for k, v in rr[r["key"]]["counts"].items():
            x, v = int(k), int(v)
            if v <= 0: continue
            for i in range(nr): c01[i, b, (x >> (nr - 1 - i)) & 1] += v
    conf, err = {}, {}
    for i, q in enumerate(phys):
        e0 = c01[i, 0, 1] / c01[i, 0].sum(); e1 = c01[i, 1, 0] / c01[i, 1].sum()
        conf[q] = np.linalg.inv(np.array([[1 - e0, e1], [e0, 1 - e1]])); err[q] = (float(e0), float(e1))
    return conf, err

# ---------------------------------------------------------------- joint (multi-site) estimator, as hardware/analyze.py
def per_shot(run, pol, frame=True, frame_pol=None):
    """chi[c, t, j] (frame-aligned), letters L[c, j], job rank; raw counts only (40 shots / circuit)."""
    recs, ci, x, v, L, jr = run.arrays(pol)
    reps = v.astype(int); xs = np.repeat(x, reps); cs = np.repeat(ci, reps)
    C = len(recs); T = np.bincount(cs, minlength=C)
    if T.min() != T.max(): raise ValueError("per_shot needs equal shots per circuit (raw counts)")
    T = int(T[0]); order = np.argsort(cs, kind="stable"); X = xs[order].reshape(C, T)
    N = run.N; chi = np.zeros((C, T, N))
    for j in range(N):
        z = 1 - 2 * run.bit(X, run.lay["system"][j]); xx = 1 - 2 * run.bit(X, run.lay["m0"][j])
        par = np.stack([np.zeros_like(z), z, z * xx, xx], -1).astype(float)
        U = np.eye(4)
        if frame and not pol.endswith("dz100"):
            fp = frame_pol or pol.split("_")[0] + "_dz100"; H0, L0, coded, nb = run.hist(fp, j); U = frame_U(H0, L0, nb, coded)
        chi[:, :, j] = np.einsum("ctk,ck->ct", par, U[L[:, j]])
    return chi, L, jr

def joint(chi, L, sites=None, crossfit=True, n_subsets=30, B=100, seed=5, idx=None):
    """Factorization ratio R_w and joint log2 advantage per weight w (bare memories)."""
    rng = np.random.default_rng(seed)
    if idx is not None: chi, L = chi[idx], L[idx]
    sites = list(range(chi.shape[2])) if sites is None else list(sites)
    chi = chi[:, :, sites]; L = L[:, sites]; C, T, N = chi.shape
    def lam_tab(ii):
        lam = np.zeros((N, 4)); G = np.zeros((N, 4))
        for j in range(N):
            for l in (1, 2, 3):
                cc = ii[L[ii, j] == l]; sc = chi[cc, :, j].sum(1); n = T * len(cc)
                lam[j, l] = sc.sum() / n; G[j, l] = max((sc.sum() ** 2 - (sc ** 2).sum()) / (n ** 2 - len(cc) * T ** 2), 1e-6)
        return lam, G
    allidx = np.arange(C); _, Gall = lam_tab(allidx); nu = np.exp(np.mean(np.log(Gall[:, 1:]), 1))
    SC = chi.sum(1)
    def nu_w(Wt):
        """nu per site for each row of the circuit-weight matrix Wt (B x C)."""
        lg = np.zeros((len(Wt), N))
        for j in range(N):
            for l in (1, 2, 3):
                m = L[:, j] == l; w_ = Wt[:, m]; sc = SC[m, j]; S = w_ @ sc; S2 = w_ @ sc ** 2; sw = w_.sum(1)
                lg[:, j] += np.log(np.maximum((S ** 2 - S2) / ((T * sw) ** 2 - T ** 2 * sw), 1e-6))
        return np.exp(lg / 3)
    perm = rng.permutation(C); halves = [perm[: C // 2], perm[C // 2:]]
    V = np.zeros((C, T, N)); M1 = np.zeros((C, N))
    for h in (0, 1):
        tr, te = halves[h], halves[1 - h]
        lam_tr, _ = lam_tab(tr if crossfit else allidx); lam_te, _ = lam_tab(te)
        for j in range(N):
            w = lam_tr[j][L[te, j]]; V[te, :, j] = chi[te, :, j] * w[:, None]; M1[te, j] = lam_te[j][L[te, j]] * w
    out = {}; Wb = boot_weights(C, B, rng); NUB = nu_w(Wb) if B else None
    for w in range(1, N + 1):
        subs = list(itertools.combinations(range(N), w))
        if len(subs) > n_subsets: subs = [subs[i] for i in rng.choice(len(subs), n_subsets, replace=False)]
        YS = np.array([np.prod(V[:, :, list(S)], axis=2).mean(1) for S in subs]); MS = np.array([np.prod(M1[:, list(S)], axis=1) for S in subs])
        Sm = np.zeros((len(subs), N)); 
        for i, S in enumerate(subs): Sm[i, list(S)] = 1
        base = Sm @ np.log2(2 * nu)
        R0 = (YS.sum(1) / MS.sum(1)); A0 = np.mean(base + np.log2(np.maximum(R0, 1e-9)))
        if B:
            Rb = (YS @ Wb.T) / (MS @ Wb.T)                                     # subsets x B
            bb = Sm @ np.log2(2 * NUB.T)                                       # marginals re-estimated in the same draw
            Rm = Rb.mean(0); Ab = np.mean(bb + np.log2(np.maximum(Rb, 1e-9)), 0)
        else: Rm = Ab = np.zeros(1)
        out[w] = dict(R=float(R0.mean()), R_ci=list(np.percentile(Rm, [2.5, 97.5])), log2A=float(A0),
                      log2A_ci=list(np.percentile(Ab, [2.5, 97.5])), log2A_marg=float(base.mean()))
    return dict(nu=list(nu), by_weight=out)

def slope(byw):
    w = np.array(sorted(byw)); y = np.array([byw[k]["log2A"] for k in w])
    lo = np.array([byw[k]["log2A_ci"][0] for k in w]); hi = np.array([byw[k]["log2A_ci"][1] for k in w])
    return float(np.polyfit(w, y, 1, w=1 / np.maximum(hi - lo, 1e-3))[0])

def zero_crossing(a, s):
    """a where s(a) crosses 0 (linear interpolation between the bracketing points; s increasing in a)."""
    a = np.asarray(a); s = np.asarray(s); o = np.argsort(a); a, s = a[o], s[o]
    for i in range(len(a) - 1):
        if s[i] <= 0 < s[i + 1]: return float(a[i] + (a[i + 1] - a[i]) * (-s[i]) / (s[i + 1] - s[i]))
    return float("nan")

def model_astar(nu1):
    return float((0.5 / np.exp(np.mean(np.log(nu1)))) ** 0.75)
