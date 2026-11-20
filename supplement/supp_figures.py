# -*- coding: utf-8 -*-
"""Supplementary figures from supp_results.json, sim_*.json and the deposited summaries.
Usage: DATA_DIR=<figshare folder> python supp_figures.py"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")); import nstyle  # Nature style
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
R = json.load(open(os.path.join(HERE, "supp_results.json")))
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=170); plt.close(fig)
def sims(t): return sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, f"sim_{t}_[0-9].json")))], [])
LC = {"Z": "#444444", "Y": "#c4452c", "X": "#3b6fb6"}
rows = R["pairs_sites"]; A = sorted({r["a"] for r in rows}); N = 9
# ---------------------------------------------------------------- S-fig: pipeline validation on simulated data
V1, V2, V3 = [x for s in (0, 1) for x in json.load(open(os.path.join(HERE, f"sim_V1_{s}.json")))], sims("V2"), sims("V3")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for k, (a, mk) in enumerate(((1.0, "o"), (0.64, "s"))):
    r = [x for x in V1 if x["a"] == a]; nu = np.array([x["nu"] for x in r]); tr = np.array(r[0]["nu_true"])
    ax[0].errorbar(np.arange(N) + 0.15 * (k - 0.5), 100 * (nu.mean(0) / tr - 1), yerr=196 * nu.std(0) / np.sqrt(len(r)) / tr, fmt=mk, ms=4,
                   capsize=2, label=f"a = {a:.2f} ({len(r)} data sets)")
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("site"); ax[0].set_ylabel(r"bias of $\hat\nu_j$ (%)"); ax[0].set_ylim(-0.4, 0.4)
ax[0].set_title("a  per-site factor: bias (95% of mean)"); ax[0].legend(fontsize=7)
w = np.arange(1, 10); cols = {0.02: "#7aa0d4", 0.05: "#3b6fb6", 0.10: "#1b3f7a"}
for d in (0.02, 0.05, 0.10):
    r = [x for x in V2 if abs(x["delta"] - d) < 1e-9]
    m = np.array([[x["R"][str(k)] for k in w] for x in r]); t = np.array([r[0]["R_true"][str(k)] for k in w])
    ax[1].errorbar(w, m.mean(0), yerr=m.std(0), fmt="o", ms=3, color=cols[d], capsize=1.5, label=fr"simulated drift $\delta$ = {d:.2f}")
    ax[1].plot(w, t, "-", color=cols[d], lw=0.8)
dw = R["pairs_joint"]["bare_dz100"]; Rd = np.array([dw[str(k)]["R"] for k in w]); lo = np.array([dw[str(k)]["R_ci"][0] for k in w]); hi = np.array([dw[str(k)]["R_ci"][1] for k in w])
ax[1].fill_between(w, lo, hi, color="#f0b37e", alpha=0.6, label="experiment, a = 1 (95% CI)"); ax[1].plot(w, Rd, "-", color="#c46a1c", lw=1)
ax[1].set_xlabel("number of sites w"); ax[1].set_ylabel(r"$R_w$"); ax[1].set_title("b  sensitivity of $R_w$ to common-mode drift"); ax[1].legend(fontsize=6.5)
a3 = np.array([x["astar"] for x in V3]); ax[2].hist(a3, bins=15, color="0.7", label=f"simulated, model truth ({len(a3)} sets)")
ax[2].axvline(0.757, color="k", ls="--", lw=1, label="model $a^*$ = 0.757")
var = [R["rob_baseline"]["astar_joint"]] + [v["astar_joint"] for v in R["rob_seeds"].values()]
for i, v in enumerate(var): ax[2].axvline(v, color="#c4452c", lw=0.8, alpha=0.8, label="experiment (7 random seeds)" if i == 0 else None)
ax[2].set_xlabel(r"reversal point $a^*$ (zero of the joint slope)"); ax[2].set_ylabel("count"); ax[2].set_title("c  sampling distribution of $a^*$")
ax[2].legend(fontsize=6.5); save(fig, "sfig_validation")
# ---------------------------------------------------------------- S-fig: per site, per letter
fig, axs = plt.subplots(3, 3, figsize=(10.5, 8.2), sharex=True, sharey=True); aa = np.linspace(0.36, 1.02, 100)
for j in range(N):
    ax = axs.flat[j]; rr = sorted([r for r in rows if r["site"] == j], key=lambda r: r["a"]); g1 = rr[-1]["G"]
    for li, L in enumerate(("X", "Y", "Z")):
        g = np.array([r["G"][li] for r in rr]); ci = np.array([r["G_ci"][li] for r in rr]); x = np.array([r["a"] for r in rr])
        ax.errorbar(x + (li - 1) * 0.006, g, yerr=[g - ci[:, 0], ci[:, 1] - g], fmt="o", ms=2.8, color=LC[L], capsize=1.2, label=f"$G_{L}$")
        ax.plot(aa, g1[li] * (aa ** 2 if L != "Z" else np.ones_like(aa)), "-", color=LC[L], lw=0.7, alpha=0.7)
    nu = np.array([r["nu"] for r in rr]); x = np.array([r["a"] for r in rr])
    ax.plot(x, nu, "D", ms=3.5, mfc="white", mec="k", label=r"$\nu$"); ax.plot(aa, rr[-1]["nu"] * aa ** (4 / 3), "k--", lw=0.8)
    ax.axhline(0.5, color="r", lw=0.6); p = rr[0]["path"]
    ax.set_title(f"site {j}: qubits {p[0]}–{p[1]}", fontsize=8.5)
    if j == 0: ax.legend(fontsize=6.5, ncol=2, loc="lower right")
for ax in axs[-1]: ax.set_xlabel("$a$")
for ax in axs[:, 0]: ax.set_ylabel("soft factor")
save(fig, "sfig_sites")
# ---------------------------------------------------------------- S-fig: residuals and robustness
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
An = [a for a in A if a < 1]; M = np.array([[next(r for r in rows if r["site"] == j and r["a"] == a)["nu"] / next(r for r in rows if r["site"] == j and r["a"] == a)["model"] - 1 for a in An] for j in range(N)])
im = ax[0].imshow(100 * M, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
for j in range(N):
    for k in range(len(An)): ax[0].text(k, j, f"{100 * M[j, k]:+.0f}", ha="center", va="center", fontsize=6)
ax[0].set_xticks(range(len(An))); ax[0].set_xticklabels([f"{a:.2f}" for a in An]); ax[0].set_yticks(range(N)); ax[0].set_xlabel("$a$"); ax[0].set_ylabel("site")
ax[0].set_title(r"a  $\nu_j/[\nu_j(1)a^{4/3}]-1$ (%)"); fig.colorbar(im, ax=ax[0], fraction=0.046)
h = R["rob_halves"]
for a in A:
    x = np.array(h["first"]["nu"][str(a)]); y = np.array(h["second"]["nu"][str(a)]); ax[1].plot(x, y, "o", ms=3, label=f"{a:.2f}")
ax[1].plot([0.15, 0.9], [0.15, 0.9], "k-", lw=0.6); ax[1].set_xlabel(r"$\nu_j$, first half of the run"); ax[1].set_ylabel(r"$\nu_j$, second half")
ax[1].set_title("b  split by submission order"); ax[1].legend(fontsize=6, ncol=2, title="a", title_fontsize=6)
lab, val, mod = [], [], []
def add(l, d): lab.append(l); val.append(d["astar_joint"]); mod.append(d["astar_model"])
add("baseline", R["rob_baseline"]); add("no frame alignment", R["rob_noframe"]); add("no cross-fitting", R["rob_nocrossfit"]); add("all 126 subsets", R["rob_allsubsets"])
add("first half", R["rob_halves"]["first"]); add("second half", R["rob_halves"]["second"])
for k in range(N): add(f"without site {k}", R["rob_loo"][str(k)])
y = np.arange(len(lab))[::-1]; ax[2].plot(val, y, "o", ms=4, color="#c4452c", label="measured (joint slope)"); ax[2].plot(mod, y, "|", ms=8, color="k", label="zero-parameter model")
ax[2].set_yticks(y); ax[2].set_yticklabels(lab, fontsize=6.5); ax[2].set_xlabel("$a^*$"); ax[2].set_title("c  reversal point under variations"); ax[2].legend(fontsize=6.5, loc="lower left")
ax[2].set_xlim(0.72, 0.78); save(fig, "sfig_robust")
# ---------------------------------------------------------------- S-fig: syndrome classes of the coded memory
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
cl = ["s = 00 (no flip)", "s = 01", "s = 10", "s = 11"]; cc = ["#222222", "#3b6fb6", "#c4452c", "#8a6bbf"]
for rn, mk in (("MS_origin_code_raw", "o"), ("MS_origin_code_ro", "s")):
    pts = [p for p in R["code"][rn]["pts"] if p["layer"] == "mit" and p["kind"] == "rep3"]
    pts.sort(key=lambda p: p["a"])
    for li, L in enumerate(("X", "Y", "Z")):
        x = [p["a"] for p in pts]
        for s in range(4):
            axs[0, li].plot(x, [p["tabs"][str(li + 1)]["q"][s] for p in pts], mk + "-", ms=3.5, lw=0.6, color=cc[s], mfc=cc[s] if mk == "o" else "white",
                            label=cl[s] if rn == "MS_origin_code_raw" else None)
            axs[1, li].plot(x, [p["tabs"][str(li + 1)]["lam"][s] for p in pts], mk + "-", ms=3.5, lw=0.6, color=cc[s], mfc=cc[s] if mk == "o" else "white")
        axs[0, li].set_title(f"letter {L}: class probability $q_s$"); axs[1, li].set_title(f"letter {L}: conditional visibility $\\lambda_s$")
        axs[1, li].set_xlabel("$a$"); axs[1, li].axhline(0, color="k", lw=0.5)
axs[0, 0].legend(fontsize=6.5); save(fig, "sfig_syndrome")
# ---------------------------------------------------------------- S-fig: independent six-qubit set (RSP) and site diagnosis
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
S = json.load(open(os.path.join(DATA, "RSP_summary_origin.json"))); col = {"bare": "#3b6fb6", "rep3": "#c4452c", "rep": "#8a6bbf"}
nm = {"bare": "unencoded", "rep3": "rep3", "rep": "rep5"}; aa = np.linspace(0.38, 0.84, 100); rng = np.random.default_rng(3); rsp = {}
for k in ("bare", "rep3", "rep"):
    it = sorted((int(p.split("dz")[1]) / 100, v) for p, v in S.items() if p.startswith(k + "_dz"))
    x = np.array([a for a, _ in it]); v = np.array([d["nu"] for _, d in it]); ci = np.array([d["ci"] for _, d in it]); s = (ci[:, 1] - ci[:, 0]) / 3.92 / v
    p, c = np.polyfit(np.log(x), np.log(v), 1, w=1 / s); xs = np.exp((np.log(0.5) - c) / p)
    bs = [np.exp((np.log(0.5) - cc_[1]) / cc_[0]) for cc_ in (np.polyfit(np.log(x), np.log(v) + rng.normal(0, s), 1, w=1 / s) for _ in range(4000))]
    rsp[k] = dict(astar=float(xs), ci=list(np.percentile(bs, [2.5, 97.5])), p=float(p))
    ax[0].errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o", ms=4, color=col[k], capsize=2, label=f"{nm[k]}: $a^*$ = {xs:.3f}")
    ax[0].plot(aa, np.exp(c) * aa ** p, "-", color=col[k], lw=1)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].set_xlabel("$a$"); ax[0].set_ylabel(r"$\nu$ (platform readout correction)")
ax[0].set_title("a  second qubit set (independent run), 10 shots per circuit"); ax[0].legend(fontsize=7)
import sys; sys.path.insert(0, HERE); from supp_core import Run, parities
def flags(rn):
    """Per chain: X/Y/Z visibilities of the noiseless rep3 circuits and the fault flag (as hardware/analyze.py)."""
    r = Run(rn); out = []
    for j in range(r.N):
        H, L, coded, nb = r.hist("rep3_dz100", j); par, _ = parities(nb, coded); M = np.zeros((3, 3)); n = np.zeros(3)
        for l in (1, 2, 3):
            hh = H[L == l]; M[l - 1] = (hh @ par[:, 1:]).sum(0) / hh.sum(); n[l - 1] = len(hh)
        thr = 3 / np.sqrt(n)[:, None]; Ms = np.where(~np.eye(3, dtype=bool) & (np.abs(M) < thr), 0, M); vis = np.linalg.norm(Ms, axis=1)
        bad = bool(abs(M[2, 2]) < 0.2 or min(vis[0], vis[1]) < 0.2 or max(abs(Ms[0, 2]), abs(Ms[1, 2]), abs(Ms[2, 0]), abs(Ms[2, 1])) > 0.3)
        out.append(dict(path=r.lay["paths"][j], vis=list(vis), flagged=bad))
    return out
runs = [("MS_origin_cal", "MS_summary_origin_cal.json", "12 calibration-ranked chains, simultaneous"),
        ("MS_origin_par5", "MS_summary_origin_par5.json", "5 unflagged chains, simultaneous"),
        ("MSI_origin_iso", "MSI_summary_origin_iso.json", "same 5 chains, one at a time"),
        ("MS_origin_ref", "MS_summary_origin_ref.json", "reference chain")]
calp = [tuple(p) for p in Run("MS_origin_cal").lay["paths"]]; diag = {}
for i, (rn, f, lab) in enumerate(runs):
    S_ = json.load(open(os.path.join(DATA, f))); fl = flags(rn); diag[rn] = []
    for j, d in enumerate(fl):
        x = calp.index(tuple(d["path"])) if tuple(d["path"]) in calp else 12
        v80 = S_["rep3_dz80"]["nu_sites"][str(j)]; v1 = S_["rep3_dz100"]["nu_sites"][str(j)]
        diag[rn].append(dict(d, nu80=v80, nu1=v1))
        ax[1].plot(x + 0.18 * (i - 1.5), v80, "o", ms=4.5, color=f"C{i}", mfc="white" if d["flagged"] else f"C{i}", label=lab if j == 0 else None)
ax[1].plot([], [], "o", mfc="white", mec="0.4", label="flagged as faulty")
ax[1].set_xticks(range(13)); ax[1].set_xticklabels([str(k) for k in range(12)] + ["ref"])
json.dump(diag, open(os.path.join(HERE, "diag_results.json"), "w"), indent=1)
ax[1].axhline(0.5, color="k", ls="--", lw=0.8); ax[1].set_xlabel("chain (index in the calibration-ranked list)"); ax[1].set_ylabel(r"$\nu$ (rep3, a = 0.80)")
ax[1].set_title("b  site diagnosis (rep3 memory, a = 0.80)"); ax[1].legend(fontsize=6.5); ax[1].set_ylim(-0.05, 1.2)
save(fig, "sfig_rsp_diag")
json.dump(rsp, open(os.path.join(HERE, "rsp_fits.json"), "w"), indent=1)
print("figures in", OUT); print("RSP fits", rsp)
