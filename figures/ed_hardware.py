# -*- coding: utf-8 -*-
"""Extended Data Figs. 4-8 (hardware). Reads the deposited data (DATA_DIR, default ../data).
  ed4  site screening (86 pairs, 104 chains)
  ed5  per-site, per-letter factors of the nine Bell pairs and noiseless letter-to-parity responses
  ed6  drift across submission batches and cross-site correlations
  ed7  factorization ratio with and without the platform's readout correction
  ed8  compilation control (identity circuits) and readout calibration
Usage: python ed_hardware.py [ed4 ed5 ...]   (default: all)"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nstyle  # Nature-style formatting (figures/nstyle.py)
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
J = lambda *p: json.load(open(os.path.join(DATA, *p)))
save = lambda fig, n: (fig.tight_layout(), fig.savefig(os.path.join(HERE, n + ".pdf")), fig.savefig(os.path.join(HERE, n + ".png"), dpi=150))


def load_run(run):
    man, raw = J(run, "manifest.json"), J(run, "raw.json"); return man, raw


def parities(x, nq, s, m0):
    z = 1 - 2 * ((x >> (nq - 1 - s)) & 1); xx = 1 - 2 * ((x >> (nq - 1 - m0)) & 1)
    return np.array([z, z * xx, xx])                     # X, Y, Z parities


def ed4():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (f, title, qmin) in enumerate((("screen_resultp.json", "a, Bell pairs (86)", 0.7), ("screen_result.json", "b, four-qubit chains (104)", 0.6))):
        R = J(f); ch = R["chains"]; chosen = {"-".join(map(str, p)) for p in R["chosen"]}
        items = sorted(ch.items(), key=lambda kv: -kv[1]["q"]); x = np.arange(len(items))
        q = np.array([v["q"] for _, v in items]); br = np.array([v["broken"] for _, v in items]); sel = np.array([k in chosen for k, _ in items])
        ax[i].bar(x[~br & ~sel], q[~br & ~sel], color="0.6", width=0.8, label="measured")
        ax[i].bar(x[br], q[br], color="#d9a0a0", width=0.8, label="faulty (flagged)")
        ax[i].bar(x[sel], q[sel], color="#3b6fb6", width=0.8, label="selected")
        ax[i].axhline(qmin, color="k", ls="--", lw=0.8); ax[i].set_xlabel("candidate (ranked)"); ax[i].set_ylabel("quality $q$")
        ax[i].set_title(title, fontsize=10, loc="left"); ax[i].legend(fontsize=8)
        print(f, len(items), "flagged", int(br.sum()), "selected", int(sel.sum()))
    save(fig, "ed4_screening")


def ed5():
    man, raw = load_run("MS_origin_pairs_raw"); lay = man["layout"]; N = len(lay["system"]); nq = len(lay["physical"])
    pols = sorted({r["meta"]["policy"] for r in man["records"]})
    acc = {}                                            # (pol, site, letter) -> list of per-circuit (sum chi-vector, n)
    for r in man["records"]:
        if r["key"] not in raw: continue
        pol = r["meta"]["policy"]
        for k, v in raw[r["key"]]["counts"].items():
            x, v = int(k), int(v)
            if v <= 0: continue
            for j in range(N):
                acc.setdefault((pol, j, r["P"][j]), {}).setdefault(r["key"], [np.zeros(3), 0])
                e = acc[(pol, j, r["P"][j])][r["key"]]; e[0] += v * parities(x, nq, lay["system"][j], lay["m0"][j]); e[1] += v
    M = np.zeros((N, 3, 3))
    for j in range(N):
        for l in (1, 2, 3):
            tot = sum(e[0] for e in acc[("bare_dz100", j, l)].values()); n = sum(e[1] for e in acc[("bare_dz100", j, l)].values())
            M[j, l - 1] = tot / n
    U = np.zeros((N, 3, 3))
    for j in range(N):
        for l in range(3):
            row = M[j, l].copy(); nc = len(acc[("bare_dz100", j, l + 1)]); thr = 3 / np.sqrt(nc)
            for k in range(3):
                if k != l and abs(row[k]) < thr: row[k] = 0
            U[j, l] = row / np.linalg.norm(row)
    G = {}
    for pol in pols:
        for j in range(N):
            for l in (1, 2, 3):
                u = U[j, l - 1] if not pol.endswith("dz100") else np.eye(3)[l - 1]
                Sc = np.array([e[0] @ u for e in acc[(pol, j, l)].values()]); nc = np.array([e[1] for e in acc[(pol, j, l)].values()])
                G[(pol, j, l)] = (Sc.sum() ** 2 - (Sc ** 2).sum()) / (nc.sum() ** 2 - (nc ** 2).sum())
    A = sorted({int(p.split("dz")[1]) / 100 for p in pols if not p.endswith("dz100")})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7)); cm = plt.get_cmap("tab10")
    for li, (l, name) in enumerate(((1, "X"), (2, "Y"), (3, "Z"))):
        for j in range(N):
            ax[li].plot(A, [G[(f"bare_dz{int(round(100 * a)):02d}", j, l)] for a in A], "o-", ms=3, lw=0.8, color=cm(j))
        aa = np.linspace(0.38, 0.9, 50); g1 = np.mean([G[("bare_dz100", j, l)] for j in range(N)])
        ax[li].plot(aa, g1 * (aa ** 2 if l < 3 else np.ones_like(aa)), "k--", lw=1.2, label=r"$G_\ell(1)\,a^2$" if l < 3 else r"$G_Z(1)$")
        ax[li].set_xlabel("$a$"); ax[li].set_ylabel(f"letter factor $G_{name}$"); ax[li].set_title(f"{'abc'[li]}, letter {name}", fontsize=10, loc="left"); ax[li].legend(fontsize=8)
    save(fig, "ed5_letters")
    json.dump(dict(M=M.tolist(), G={f"{k[0]}|{k[1]}|{k[2]}": float(v) for k, v in G.items()}), open(os.path.join(HERE, "ed5_letters.json"), "w"), indent=1)
    print("noiseless letter-to-parity matrices (rows X,Y,Z):"); [print(j, np.round(M[j], 2).tolist()) for j in range(N)]


def ed6():
    man, raw = load_run("MS_origin_pairs_raw"); jobs = J("MS_origin_pairs_raw", "jobs.json"); lay = man["layout"]
    N = len(lay["system"]); nq = len(lay["physical"]); order = sorted(jobs, key=lambda j: jobs[j].get("submitted_unix", 0))
    S = {j: np.zeros(N) for j in order}; C = {j: np.zeros(N) for j in order}
    for r in man["records"]:
        if r["meta"]["policy"] != "bare_dz100" or r["key"] not in raw: continue
        jid = raw[r["key"]]["job"]
        for k, v in raw[r["key"]]["counts"].items():
            x, v = int(k), int(v)
            for i in range(N):
                p = parities(x, nq, lay["system"][i], lay["m0"][i]); S[jid][i] += v * p[r["P"][i] - 1]; C[jid][i] += v
    Jb = [j for j in order if C[j].min() > 0]; V = np.array([S[j] / C[j] for j in Jb])
    t = np.arange(len(Jb))          # batches in submission (= queue) order
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(width_ratios=[1.6, 1])); cm = plt.get_cmap("tab10")
    for i in range(N): ax[0].plot(t, V[:, i], "-", lw=0.7, color=cm(i))
    ax[0].plot(t, V.mean(1), "k-", lw=1.6, label="site average"); ax[0].set_xlabel("batch (submission order)"); ax[0].set_ylabel("noiseless visibility")
    ax[0].set_title(f"a, {len(Jb)} submission batches", fontsize=10, loc="left"); ax[0].legend(fontsize=8)
    cc = np.corrcoef(V.T); im = ax[1].imshow(cc, vmin=-1, vmax=1, cmap="RdBu_r"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set_xlabel("site"); ax[1].set_ylabel("site"); ax[1].set_title("b, cross-site correlation", fontsize=10, loc="left")
    save(fig, "ed6_drift"); off = cc[~np.eye(N, dtype=bool)]; print("batches", len(Jb), "mean corr", off.mean())


def ed7():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (f, title) in enumerate((("MS_summary_origin_pairs.json", "a, platform readout correction"), ("MS_summary_origin_pairs_raw.json", "b, raw counts"))):
        S = J(f); cm = plt.get_cmap("coolwarm"); keys = sorted(S, key=lambda k: int(k.split("dz")[1]))
        for c, k in enumerate(keys):
            d = S[k]["by_weight"]; w = np.array(sorted(int(x) for x in d)); R = np.array([d[str(x)]["R"] for x in w])
            lo = np.array([d[str(x)]["R_ci"][0] for x in w]); hi = np.array([d[str(x)]["R_ci"][1] for x in w])
            ax[i].errorbar(w + (c - 3) * 0.05, R, yerr=[R - lo, hi - R], fmt="o", ms=3, capsize=1.5, color=cm(c / (len(keys) - 1)),
                           label=f"a = {int(k.split('dz')[1]) / 100:.2f}")
        ax[i].axhline(1, color="k", lw=0.8); ax[i].set_ylim(0.6, 1.6); ax[i].set_xlabel("number of sites $w$"); ax[i].set_ylabel("$R_w$")
        ax[i].set_title(title, fontsize=10, loc="left")
    ax[0].legend(fontsize=7, ncol=2); save(fig, "ed7_amend")


def ed8():
    d = J("cancel_test_20261118_0200.json"); ncz = [0, 2, 8, 32, 10, 20, 40]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for mode, lab, mk in (("raw_on", "cloud optimization on", "o"), ("raw_on_barrier", "on, barriers after every gate", "s"), ("native_off", "off, native compilation", "^")):
        p0 = [r.get("0" * 6, 0.0) for r in d[mode]["res"]]
        ax[0].plot(ncz[:4], p0[:4], mk + "-", label=lab + " (CZ pairs)"); ax[0].plot(ncz[4:], p0[4:], mk + ":", color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("number of CZ gates (identity circuits)"); ax[0].set_ylabel("P(all zero)"); ax[0].set_ylim(0, 1)
    ax[0].set_title("a, compilation control", fontsize=10, loc="left"); ax[0].legend(fontsize=7)
    for k, (run, lab) in enumerate((("RO_origin_ro", "run 1 (separate)"), ("MS_origin_code_ro", "run 2 (interleaved)"))):
        man, raw = load_run(run); phys = man["layout"]["physical"]; nq = len(phys); c = np.zeros((nq, 2, 2))
        for r in man["records"]:
            if r["key"] not in raw or not r["meta"]["policy"].startswith("ro"): continue
            b = int(r["meta"]["policy"][2])
            for kk, v in raw[r["key"]]["counts"].items():
                x, v = int(kk), int(v)
                for i in range(nq): c[i, b, (x >> (nq - 1 - i)) & 1] += v
        e0 = c[:, 0, 1] / c[:, 0].sum(1); e1 = c[:, 1, 0] / c[:, 1].sum(1); xx = np.arange(nq)
        ax[1].bar(xx + k * 0.4 - 0.3, e0, 0.2, label=f"P(1|0), {lab}"); ax[1].bar(xx + k * 0.4 - 0.1, e1, 0.2, label=f"P(0|1), {lab}")
        ax[1].set_xticks(xx); ax[1].set_xticklabels([str(q) for q in phys])
    ax[1].set_xlabel("qubit"); ax[1].set_ylabel("readout error"); ax[1].set_title("b, readout calibration (chain 111-120-129-138)", fontsize=10, loc="left"); ax[1].legend(fontsize=7)
    save(fig, "ed8_controls")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ed4", "ed5", "ed6", "ed7", "ed8"]):
        globals()[name](); print("done", name, flush=True)
