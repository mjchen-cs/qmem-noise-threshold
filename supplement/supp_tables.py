# -*- coding: utf-8 -*-
"""LaTeX tables for the Supplementary Information (written to supp_tables.tex; one \\newcommand per table).
Usage: DATA_DIR=<figshare folder> python supp_tables.py"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
R = json.load(open(os.path.join(HERE, "supp_results.json"))); SIM = json.load(open(os.path.join(HERE, "sim_summary.json")))
SUM = json.load(open(os.path.join(DATA, "MS_summary_origin_pairs_raw.json"))); SUMA = json.load(open(os.path.join(DATA, "MS_summary_origin_pairs.json")))
RSP = json.load(open(os.path.join(DATA, "RSP_summary_origin.json"))); RF = json.load(open(os.path.join(HERE, "rsp_fits.json")))
DG = json.load(open(os.path.join(HERE, "diag_results.json")))
T = {}
def pm(v, ci, d=3): return f"${v:.{d}f}\\,^{{+{ci[1] - v:.{d}f}}}_{{-{v - ci[0]:.{d}f}}}$"
def br(v, ci, d=3): return f"{v:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
def key(a): return f"bare_dz{int(round(100 * a)):02d}"
rows = R["pairs_sites"]; A = sorted({r["a"] for r in rows}); N = 9
paths = {r["site"]: r["path"] for r in rows}
# ---- validation
s1, s6 = SIM["1.0"], SIM["0.64"]
T["TabValidation"] = r"""\begin{tabular}{lcc}\toprule
 & $a=1.00$ & $a=0.64$\\\midrule
Simulated data sets & %d & %d\\
Largest relative bias of $\hat\nu_j$ (9 sites) & %.2f\%% & %.2f\%%\\
Relative statistical error of $\hat\nu_j$ (mean over sites) & %.1f\%% & %.1f\%%\\
Coverage of 95\%% intervals for $\nu_j$ & %.3f & %.3f\\
Mean of $\hat R_9$ (truth 1) & %.4f & %.4f\\
Standard deviation of $\hat R_9$ & %.4f & %.4f\\
Coverage of 95\%% intervals for $R_9$ & %.3f & %.3f\\
Bias of $\widehat{\log_2A_9}$ & $%+.4f$ & $%+.4f$\\
Standard deviation of $\widehat{\log_2A_9}$ & %.3f & %.3f\\
Coverage of 95\%% intervals for $\log_2A_9$ & %.3f & %.3f\\\bottomrule
\end{tabular}""" % (s1["n"], s6["n"], s1["maxbias"], s6["maxbias"], 100 * s1["relsd"], 100 * s6["relsd"], s1["nucov"], s6["nucov"], s1["R9"], s6["R9"],
                     s1["R9sd"], s6["R9sd"], s1["R9cov"], s6["R9cov"], s1["A9bias"], s6["A9bias"], s1["A9sd"], s6["A9sd"], s1["A9cov"], s6["A9cov"])
# ---- nu_j(a) for all sites
L = [r"\begin{tabular}{cc" + "c" * len(A) + r"}\toprule", "Site & Qubits & " + " & ".join(f"$a={a:.2f}$" for a in A) + r"\\\midrule"]
for j in range(N):
    c = []
    for a in A:
        r = next(x for x in rows if x["site"] == j and x["a"] == a); c.append(f"{r['nu']:.3f}({int(round(1000 * (r['ci'][1] - r['ci'][0]) / 3.92))})")
    L.append(f"{j} & {paths[j][0]}--{paths[j][1]} & " + " & ".join(c) + r"\\")
g = [np.exp(np.mean(np.log([x["nu"] for x in rows if x["a"] == a]))) for a in A]
L.append(r"\midrule\multicolumn{2}{l}{Geometric mean} & " + " & ".join(f"{v:.3f}" for v in g) + r"\\")
gm = [np.exp(np.mean(np.log([x["model"] for x in rows if x["a"] == a]))) for a in A]
L.append(r"\multicolumn{2}{l}{Model $\nu_j(1)a^{4/3}$} & " + " & ".join(f"{v:.3f}" for v in gm) + r"\\\bottomrule\end{tabular}")
T["TabNuSites"] = "\n".join(L)
# ---- letters, long table
L = [r"\begin{longtable}{cccccccr}", r"\caption{\textbf{Letter factors of the nine-pair experiment (raw counts).} For every site $j$ and injected noise $a$: letter factors $G_X,G_Y,G_Z$ with 95\% bootstrap intervals (500 resamplings of circuits), the per-site factor $\nu_j$, the zero-parameter model $\nu_j(1)a^{4/3}$ and the standardized residual $z=(\nu_j-\text{model})/\sigma_j$. Letters are labelled in the frame of the injected noise $\boldsymbol\lambda=(a,a,1)$, so that the model predicts $G_X,G_Y\propto a^2$ and $G_Z$ independent of $a$.}\label{tab:letters}\\\toprule",
     r"Site & $a$ & $G_X$ & $G_Y$ & $G_Z$ & $\nu_j$ & Model & $z$\\\midrule\endfirsthead", r"\toprule Site & $a$ & $G_X$ & $G_Y$ & $G_Z$ & $\nu_j$ & Model & $z$\\\midrule\endhead", r"\bottomrule\endfoot"]
for j in range(N):
    for a in sorted(A, reverse=True):
        r = next(x for x in rows if x["site"] == j and x["a"] == a)
        G = " & ".join(br(r["G"][k], r["G_ci"][k]) for k in range(3))
        L.append(f"{j} & {a:.2f} & {G} & {r['nu']:.3f} & {r['model']:.3f} & " + (f"${r['z']:+.1f}$" if r["z"] is not None else "--") + r"\\")
    if j < N - 1: L.append(r"\midrule")
L.append(r"\end{longtable}")
T["TabLetters"] = "\n".join(L)
# ---- R_w and log2 A_w (as in Fig. 2, hardware/analyze.py, 200 bootstrap resamplings)
def wtab(field, d=3, sumd=SUM):
    A = sorted(int(k.split("dz")[1]) / 100 for k in sumd if k.startswith("bare_dz") and "by_weight" in sumd[k])
    L = [r"\begin{tabular}{c" + "c" * len(A) + r"}\toprule", "$w$ & " + " & ".join(f"$a={a:.2f}$" for a in A) + r"\\\midrule"]
    for w in range(1, 10):
        c = []
        for a in A:
            x = sumd[key(a)]["by_weight"][str(w)]
            c.append(f"{x[field]:.{d}f} [{x[field + '_ci'][0]:.{d}f}, {x[field + '_ci'][1]:.{d}f}]" if field == "R" else
                     f"${x[field]:+.2f}$ [${x[field + '_ci'][0]:+.2f}$, ${x[field + '_ci'][1]:+.2f}$]")
        L.append(f"{w} & " + " & ".join(c) + r"\\")
    L.append(r"\bottomrule\end{tabular}"); return "\n".join(L)
T["TabRw"] = wtab("R"); T["TabAw"] = wtab("log2A"); T["TabRwAmend"] = wtab("R", sumd=SUMA)
# ---- robustness
def rb(lab, d):
    r = d["R_last"]["1.0"]; s = d["slopes"]
    return f"{lab} & {d['astar_joint']:.3f} & {d['astar_model']:.3f} & {s['0.4']:+.3f} & {s['0.88']:+.3f} & {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]\\\\"
L = [r"\begin{tabular}{lccccc}\toprule", r"Analysis variant & $a^*$ (joint slope) & $a^*$ (model) & slope, $a=0.40$ & slope, $a=0.88$ & $R_9$, $a=1$\\\midrule",
     rb("Baseline", R["rob_baseline"]), rb("No frame alignment", R["rob_noframe"]), rb("No cross-fitting (in-sample weights)", R["rob_nocrossfit"]),
     rb("All 126 subsets per $w$ ($w=4,5$)", R["rob_allsubsets"])]
for k, d in R["rob_seeds"].items(): L.append(rb(f"Random seed {k} (halves, subsets, bootstrap)", d))
L.append(r"\midrule"); L.append(rb("First half of the run", R["rob_halves"]["first"])); L.append(rb("Second half of the run", R["rob_halves"]["second"]))
L.append(r"\midrule")
for k in range(N): L.append(rb(f"Without site {k} (8 sites)", R["rob_loo"][str(k)]).replace("R_9", "R_8"))
L.append(r"\bottomrule\end{tabular}"); T["TabRobust"] = "\n".join(L)
# ---- coded memory
L = [r"\begin{tabular}{ccccccc}\toprule", r"Run & $a$ & Circuits & \multicolumn{2}{c}{Raw counts} & \multicolumn{2}{c}{Readout-mitigated}\\\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r" & & per memory & Unencoded & rep3 & Unencoded & rep3\\\midrule"]
for i, rn in enumerate(("MS_origin_code_raw", "MS_origin_code_ro")):
    P = R["code"][rn]["pts"]; aa = sorted({p["a"] for p in P}, reverse=True)
    for a in aa:
        g = {(p["layer"], p["kind"]): p for p in P if p["a"] == a}
        L.append(f"{i + 1} & {a:.2f} & {g[('raw', 'bare')]['n_circ']} & " + " & ".join(br(g[k]["nu"], g[k]["ci"]) for k in (("raw", "bare"), ("raw", "rep3"), ("mit", "bare"), ("mit", "rep3"))) + r"\\")
    if i == 0: L.append(r"\midrule")
L.append(r"\bottomrule\end{tabular}"); T["TabCode"] = "\n".join(L)
L = [r"\begin{tabular}{cccccccc}\toprule", r"Run & $a$ & \multicolumn{3}{c}{Unencoded} & \multicolumn{3}{c}{rep3}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r" & & $G_X$ & $G_Y$ & $G_Z$ & $G_X$ & $G_Y$ & $G_Z$\\\midrule"]
for i, rn in enumerate(("MS_origin_code_raw", "MS_origin_code_ro")):
    P = R["code"][rn]["pts"]; aa = sorted({p["a"] for p in P}, reverse=True)
    for a in aa:
        g = {p["kind"]: p for p in P if p["a"] == a and p["layer"] == "mit"}
        L.append(f"{i + 1} & {a:.2f} & " + " & ".join(f"{g[k]['G'][m]:.3f}" for k in ("bare", "rep3") for m in range(3)) + r"\\")
    if i == 0: L.append(r"\midrule")
L.append(r"\bottomrule\end{tabular}"); T["TabCodeLetters"] = "\n".join(L)
L = [r"\begin{tabular}{ccc" + "c" * 8 + r"}\toprule", r"Run & $a$ & Letter & \multicolumn{4}{c}{$q_s$} & \multicolumn{4}{c}{$\lambda_s$}\\\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
     r" & & & 00 & 01 & 10 & 11 & 00 & 01 & 10 & 11\\\midrule"]
for i, rn in enumerate(("MS_origin_code_raw", "MS_origin_code_ro")):
    P = sorted([p for p in R["code"][rn]["pts"] if p["layer"] == "mit" and p["kind"] == "rep3"], key=lambda p: -p["a"])
    for p in P:
        for m, Lt in enumerate("XYZ"):
            t = p["tabs"][str(m + 1)]
            L.append((f"{i + 1} & {p['a']:.2f}" if m == 0 else " & ") + f" & {Lt} & " + " & ".join(f"{v:.3f}" for v in t["q"]) + " & " + " & ".join(f"${v:+.2f}$" for v in t["lam"]) + r"\\")
    if i == 0: L.append(r"\midrule")
L.append(r"\bottomrule\end{tabular}"); T["TabSyndrome"] = "\n".join(L)
L = [r"\begin{tabular}{cccc}\toprule", r"Qubit & Role & Run 1 (calibration 1 h later) & Run 2 (interleaved)\\", r" & & $P(1|0)$ / $P(0|1)$ & $P(1|0)$ / $P(0|1)$\\\midrule"]
role = {"111": "system", "120": "memory 1", "129": "memory 2 (syndrome)", "138": "memory 3 (syndrome)"}
for q in ("111", "120", "129", "138"):
    e1 = R["code"]["MS_origin_code_raw"]["readout"][q]; e2 = R["code"]["MS_origin_code_ro"]["readout"][q]
    L.append(f"{q} & {role[q]} & {e1[0]:.4f} / {e1[1]:.4f} & {e2[0]:.4f} / {e2[1]:.4f}\\\\")
L.append(r"\bottomrule\end{tabular}"); T["TabReadout"] = "\n".join(L)
# ---- independent qubit set
nm = {"bare": "Unencoded", "rep3": "rep3", "rep": "rep5"}; aa = sorted({int(p.split("dz")[1]) for p in RSP})
L = [r"\begin{tabular}{c" + "c" * 3 + r"}\toprule", r"$a$ & Unencoded & rep3 & rep5\\\midrule"]
for a in aa:
    L.append(f"{a / 100:.2f} & " + " & ".join(br(RSP[f'{k}_dz{a}']['nu'], RSP[f'{k}_dz{a}']['ci']) for k in ("bare", "rep3", "rep")) + r"\\")
L.append(r"\midrule $a^*$ (power-law fit) & " + " & ".join(br(RF[k]["astar"], RF[k]["ci"]) for k in ("bare", "rep3", "rep")) + r"\\")
L.append(r"Exponent $p$ & " + " & ".join(f"{RF[k]['p']:.2f}" for k in ("bare", "rep3", "rep")) + r"\\\bottomrule\end{tabular}")
T["TabRSP"] = "\n".join(L)
# ---- diagnosis
L = [r"\begin{tabular}{clccccc}\toprule", r"Chain & Qubits & Visibilities $X/Y/Z$ & Flag & \multicolumn{3}{c}{$\nu$ (rep3, $a=0.80$)}\\\cmidrule(lr){5-7}",
     r" & & ($a=1$, simultaneous) & & simultaneous (12) & simultaneous (5) & isolated\\\midrule"]
cal = DG["MS_origin_cal"]; p5 = {tuple(d["path"]): d for d in DG["MS_origin_par5"]}; iso = {tuple(d["path"]): d for d in DG["MSI_origin_iso"]}
for i, d in enumerate(cal):
    t = tuple(d["path"]); v = "/".join(f"{x:.2f}" for x in d["vis"])
    L.append(f"{i} & {'--'.join(map(str, d['path']))} & {v} & {'faulty' if d['flagged'] else '--'} & {d['nu80']:.3f} & "
             + (f"{p5[t]['nu80']:.3f}" if t in p5 else "") + " & " + (f"{iso[t]['nu80']:.3f}" if t in iso else "") + r"\\")
rf = DG["MS_origin_ref"][0]
L.append(r"\midrule ref & " + "--".join(map(str, rf["path"])) + " & " + "/".join(f"{x:.2f}" for x in rf["vis"]) + f" & -- & {rf['nu80']:.3f} & & \\\\")
L.append(r"\bottomrule\end{tabular}"); T["TabDiag"] = "\n".join(L)
LT = T.pop("TabLetters")
open(os.path.join(HERE, "supp_table_letters.tex"), "w", encoding="utf-8").write(LT + "\n")
with open(os.path.join(HERE, "supp_tables.tex"), "w", encoding="utf-8") as f:
    for k, v in T.items(): f.write(f"\\newcommand{{\\{k}}}{{%\n{v}}}\n\n")
print("wrote", len(T), "tables")
