# -*- coding: utf-8 -*-
"""Figure: the experiment on the threshold diagram (Wukong 180, nine Bell pairs, raw counts).
Runs MS_origin_pairs_raw (dephasing (a, a, 1), 18 Nov) and MS_origin_phase (depolarizing (eta, eta, eta) and strong
dephasing, 19 Nov). For every site and setting the letter factors give the effective sum of squared Pauli eigenvalues of
the whole memory channel, Sigma = G_X + G_Y + G_Z (hardware included); the threshold of the theory is Sigma = 1.
(a) joint growth rate (slope of log2 A_w in w, bits per site) against the site-averaged Sigma; (b) per-site factor nu_j
against Sigma_j, with the bound nu <= Sigma/3 of Bell probes; (c) joint advantage log2 A_w in the depolarizing scan."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import nstyle  # Nature-style formatting (figures/nstyle.py)
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "supplement"))
from supp_core import Run, site_nu, frame_U, slope, zero_crossing, DATA
OUT = {}
def collect(run_name):
    run = Run(run_name); S = json.load(open(os.path.join(DATA, f"MS_summary_{run_name[3:]}.json")))
    U = {}
    for j in range(run.N):
        H0, L0, c, nb = run.hist("bare_dz100", j); U[j] = frame_U(H0, L0, nb, c)
    rows = []
    for pol in S:
        code = pol.split("_")[1]; fam, t = code[:2], int(code[2:]) / 100
        sig, nus = [], []
        for j in range(run.N):
            H, L, c, nb = run.hist(pol, j); r = site_nu(H, L, nb, c, np.eye(4) if code == "dz100" else U[j])
            sig.append(sum(r["G"])); nus.append(r["nu"])
        bw = {int(k): v for k, v in S[pol]["by_weight"].items()}
        rows.append(dict(run=run_name, fam=fam if code != "dz100" else "ref", t=t, sigma=sig, nu=nus, slope=slope(bw),
                         slope_marg=float(np.mean(np.log2(2 * np.array(nus)))), byw=bw))
    return rows
rows = collect("MS_origin_pairs_raw") + collect("MS_origin_phase")
st = {"dz": dict(c="#3b6fb6", m="o", lab=r"dephasing $(a,a,1)$"), "dp": dict(c="#c4452c", m="s", lab=r"depolarizing $(\eta,\eta,\eta)$"),
      "ref": dict(c="0.3", m="D", lab="no injected noise")}
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
x = np.linspace(0.5, 3.0, 300)
ax[0].axvspan(0.4, 1.0, color="0.9", zorder=0); ax[1].axvspan(0.4, 1.0, color="0.9", zorder=0)
for k, r in enumerate(rows):
    s = st[r["fam"]]; sg = np.array(r["sigma"]); m = sg.mean()
    ax[0].errorbar(m, r["slope"], xerr=[[m - sg.min()], [sg.max() - m]], fmt=s["m"], color=s["c"], ms=5, capsize=0, elinewidth=0.6,
                   mfc=s["c"] if r["run"].endswith("phase") else "white", label=None)
    ax[1].plot(sg, r["nu"], s["m"], color=s["c"], ms=3.2, alpha=0.75, mfc=s["c"] if r["run"].endswith("phase") else "white")
for fam in ("dz", "dp", "ref"):
    s = st[fam]; ax[0].plot([], [], s["m"], color=s["c"], label=s["lab"])
ax[0].plot([], [], "o", color="0.4", mfc="white", label="open: 18 Nov, filled: 19 Nov")
ax[0].plot(x, np.log2(2 * np.clip(x, 1e-9, None) / 3), "--", color="#c4452c", lw=0.9, label="ideal Bell pairs, depolarizing")
xd = x[x > 1]; ax[0].plot(xd, np.log2(2 * ((xd - 1) / 2) ** (2 / 3)), "--", color="#3b6fb6", lw=0.9, label="ideal Bell pairs, dephasing")
ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(1, color="k", lw=1.2)
ax[0].text(0.54, 0.66, "no protocol\ncan grow", fontsize=6.5, va="top"); ax[0].set_xlim(0.5, 2.7); ax[0].set_ylim(-2.6, 0.75)
ax[0].set_xlabel(r"effective $\Sigma=G_X+G_Y+G_Z$ (site average)"); ax[0].set_ylabel(r"growth rate of $\log_2A_w$ (bits per site)")
ax[0].set_title("a", loc="left", fontweight="bold")
h_, l_ = ax[0].get_legend_handles_labels()
ax[1].plot(x, x / 3, "k-", lw=0.8, label=r"$\nu\leq\Sigma/3$ (Bell probes)"); ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
ax[1].axvline(1, color="k", lw=1.2); ax[1].set_xlim(0.5, 2.7); ax[1].set_ylim(0, 0.95)
ax[1].set_xlabel(r"effective $\Sigma_j$ (per site)"); ax[1].set_ylabel(r"per-site factor $\nu_j$")
ax[1].set_title("b", loc="left", fontweight="bold")
hb, lb = ax[1].get_legend_handles_labels()
fig.legend(h_ + hb, l_ + lb, loc="upper center", bbox_to_anchor=(0.37, 0.0), ncol=4, fontsize=6, frameon=False, columnspacing=1.2)
cm = plt.get_cmap("coolwarm"); dp = sorted([r for r in rows if r["fam"] == "dp"], key=lambda r: r["t"])
for i, r in enumerate(dp):
    w = np.array(sorted(r["byw"])); y = np.array([r["byw"][k]["log2A"] for k in w])
    lo = np.array([r["byw"][k]["log2A_ci"][0] for k in w]); hi = np.array([r["byw"][k]["log2A_ci"][1] for k in w])
    ok = (hi - lo) < 6
    ax[2].errorbar(w[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], fmt="o", ms=3.5, capsize=1.5, color=cm(i / (len(dp) - 1)),
                   label=None)
    ax[2].plot(w, w * r["slope_marg"], "-", lw=0.8, color=cm(i / (len(dp) - 1)), alpha=0.7)
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_ylim(-8, 4.5); ax[2].set_xlabel("number of sites $w$"); ax[2].set_ylabel(r"$\log_2A_w$")
ax[2].set_title("c", loc="left", fontweight="bold")
for i, r in enumerate(dp):               # label each line at its right end (or where it leaves the frame)
    sl = r["slope_marg"]; xe, ye = 9.25, 9 * sl
    if ye < -7.6: xe, ye = -7.6 / sl + 0.15, -7.6
    ax[2].text(xe, ye, f"{r['t']:.2f}", color=tuple(0.75 * np.array(cm(i / (len(dp) - 1))[:3])), fontsize=5.8, va="center")
ax[2].text(9.25, 4.35, r"$\eta$", fontsize=6.5, va="center"); ax[2].set_xlim(0.5, 10.2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_phase.pdf")); fig.savefig(os.path.join(HERE, "fig3_phase.png"), dpi=200)
# numbers
dpz = [r for r in rows if r["fam"] == "dp"]; dzz = [r for r in rows if r["fam"] == "dz" and r["run"].endswith("raw")]
nu1 = np.array(next(r for r in rows if r["fam"] == "ref" and r["run"].endswith("phase"))["nu"])
OUT["dp_astar"] = zero_crossing([r["t"] for r in dpz], [r["slope"] for r in dpz]); OUT["dp_model"] = float((0.5 / np.exp(np.mean(np.log(nu1)))) ** 0.5)
OUT["dp_sigma_at_crossing"] = float(np.interp(OUT["dp_astar"], [r["t"] for r in sorted(dpz, key=lambda r: r["t"])], [np.mean(r["sigma"]) for r in sorted(dpz, key=lambda r: r["t"])]))
OUT["dz_astar"] = zero_crossing([r["t"] for r in dzz], [r["slope"] for r in dzz])
OUT["dz_sigma_at_crossing"] = float(np.interp(OUT["dz_astar"], [r["t"] for r in sorted(dzz, key=lambda r: r["t"])], [np.mean(r["sigma"]) for r in sorted(dzz, key=lambda r: r["t"])]))
OUT["below_threshold"] = [dict(run=r["run"], code=f"{r['fam']}{int(round(100 * r['t']))}", sigma=float(np.mean(r["sigma"])), slope=r["slope"])
                          for r in rows if np.mean(r["sigma"]) < 1.0]
OUT["rows"] = [dict(run=r["run"], fam=r["fam"], t=r["t"], sigma_mean=float(np.mean(r["sigma"])), sigma=r["sigma"], nu=r["nu"],
                    slope=r["slope"], slope_marg=r["slope_marg"]) for r in rows]
json.dump(OUT, open(os.path.join(HERE, "fig3_phase_numbers.json"), "w"), indent=1)
print({k: v for k, v in OUT.items() if k != "rows"})
