# -*- coding: utf-8 -*-
"""Tables of the second-day experiments (Supplementary Section S17): threshold scan of the nine Bell pairs
(MS_origin_phase) and the bias scans of the coding chain (MS_origin_bias, MS_origin_bias2). Needs fig3_phase_numbers.json
and fig4_bias_numbers.json (run ../figures/fig3_phase.py and fig4_bias.py first). Writes supp_day2_tables.tex."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
FIG = os.path.join(HERE, "..", "figures")
P = json.load(open(os.path.join(FIG, "fig3_phase_numbers.json"))); B = json.load(open(os.path.join(FIG, "fig4_bias_numbers.json")))
S = json.load(open(os.path.join(DATA, "MS_summary_origin_phase.json")))
T = {}
rows = sorted([r for r in P["rows"] if r["run"] == "MS_origin_phase"], key=lambda r: ({"ref": 0, "dz": 1, "dp": 2}[r["fam"]], r["t"]))
lab = lambda r: "none" if r["fam"] == "ref" else (f"$a={r['t']:.2f}$" if r["fam"] == "dz" else f"$\\eta={r['t']:.2f}$")
code = lambda r: "bare_dz100" if r["fam"] == "ref" else f"bare_{r['fam']}{int(round(100 * r['t']))}"
L = [r"\begin{tabular}{l" + "c" * 9 + "c}\\toprule", "Noise & " + " & ".join(f"{j}" for j in range(9)) + r" & mean\\\midrule"]
for key, name in (("nu", r"\multicolumn{11}{l}{\emph{Per-site factor $\nu_j$}}"), ("sigma", r"\multicolumn{11}{l}{\emph{Effective $\Sigma_j=G_X+G_Y+G_Z$}}")):
    L.append(name + r"\\")
    for r in rows:
        v = np.array(r[key]); m = np.exp(np.mean(np.log(v))) if key == "nu" else v.mean()
        L.append(lab(r) + " & " + " & ".join(f"{x:.3f}" if key == "nu" else f"{x:.2f}" for x in v) + (f" & {m:.3f}" if key == "nu" else f" & {m:.2f}") + r"\\")
    if key == "nu": L.append(r"\midrule")
L.append(r"\bottomrule\end{tabular}"); T["TabDayTwoSites"] = "\n".join(L)
L = [r"\begin{tabular}{lccccc}\toprule", r"Noise & $\bar\Sigma$ & slope (joint) & slope (marginal) & $R_9$ [95\%] & $\log_2A_9$ [95\%]\\\midrule"]
for r in rows:
    b = S[code(r)]["by_weight"]["9"]
    L.append(f"{lab(r)} & {np.mean(r['sigma']):.2f} & ${r['slope']:+.3f}$ & ${r['slope_marg']:+.3f}$ & {b['R']:.3f} [{b['R_ci'][0]:.2f}, {b['R_ci'][1]:.2f}] & "
             f"${b['log2A']:+.2f}$ [${b['log2A_ci'][0]:+.1f}$, ${b['log2A_ci'][1]:+.1f}$]\\\\")
L.append(r"\bottomrule\end{tabular}"); T["TabDayTwoJoint"] = "\n".join(L)
# coding chain
D = {"MS_origin_bias2": json.load(open(os.path.join(DATA, "MS_origin_bias2_mitigate.json"))), "MS_origin_bias": json.load(open(os.path.join(DATA, "MS_origin_bias_mitigate.json")))}
fam_lz = {"dz": "1.00", "zs": "0.98", "zh": "0.95", "zm": "0.90"}
L = [r"\begin{tabular}{cccccc}\toprule", r"$\lambda_Z$ & $t$ & \multicolumn{2}{c}{Raw counts} & \multicolumn{2}{c}{Readout-mitigated}\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r" & & Unencoded & rep3 & Unencoded & rep3\\\midrule"]
br = lambda x: f"{x['nu']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
for run, fams in (("MS_origin_bias2", ("dz", "zs")), ("MS_origin_bias", ("zh", "zm"))):
    d = D[run]
    L.append(f"-- & 1.00 & {br(d['raw_bare_dz100'][0])} & {br(d['raw_rep3_dz100'][0])} & {br(d['mit_bare_dz100'][0])} & {br(d['mit_rep3_dz100'][0])}\\\\")
    for f in fams:
        ts = sorted({int(k.split(f)[1]) for k in d if f"_bare_{f}" in k and not k.endswith("dz100")})
        for t in ts:
            k = f"{f}{t}"
            L.append(f"{fam_lz[f]} & {t / 100:.2f} & {br(d['raw_bare_' + k][0])} & {br(d['raw_rep3_' + k][0])} & {br(d['mit_bare_' + k][0])} & {br(d['mit_rep3_' + k][0])}\\\\")
    L.append(r"\midrule" if run == "MS_origin_bias2" else "")
L.append(r"\bottomrule\end{tabular}"); T["TabDayTwoCode"] = "\n".join(L)
L = [r"\begin{tabular}{cccc}\toprule", r"$\lambda_Z$ & Unencoded $t^*$ & rep3 $t^*$ & Shift\\\midrule"]
for lz, m in B["measured"].items():
    L.append(f"{float(lz):.2f} & {m['bare']['astar']:.3f} [{m['bare']['ci'][0]:.3f}, {m['bare']['ci'][1]:.3f}] & {m['rep3']['astar']:.3f} [{m['rep3']['ci'][0]:.3f}, {m['rep3']['ci'][1]:.3f}] & "
             f"${m['shift']['median']:+.3f}$ [${m['shift']['ci'][0]:+.3f}$, ${m['shift']['ci'][1]:+.3f}$]\\\\")
o = B["measured_18Nov"]; L.append(f"1.00 (first day) & 0.626 [0.614, 0.638] & 0.536 [0.524, 0.547] & ${o['median']:+.3f}$ [${o['ci'][0]:+.3f}$, ${o['ci'][1]:+.3f}$]\\\\")
L.append(r"\bottomrule\end{tabular}"); T["TabDayTwoShift"] = "\n".join(L)
L = [r"\begin{tabular}{ccccc}\toprule", r"Qubit & Code pilot & Bias 1 & Bias 2 & (first day, runs 1/2)\\\midrule"]
ro = {}
for run in ("MS_origin_codepilot", "MS_origin_bias", "MS_origin_bias2"):
    import sys; sys.path.insert(0, HERE); from supp_core import Run, readout_conf
    _, err = readout_conf(Run(run)); ro[run] = err
first = {111: ("0.0405/0.0200", "0.0378/0.0268"), 120: ("0.0473/0.0435", "0.0527/0.0405"), 129: ("0.0312/0.0255", "0.0272/0.0293"), 138: ("0.0073/0.0192", "0.0103/0.0158")}
for q in (111, 120, 129, 138):
    L.append(f"{q} & " + " & ".join(f"{ro[r][q][0]:.4f}/{ro[r][q][1]:.4f}" for r in ro) + f" & {first[q][0]}, {first[q][1]}\\\\")
L.append(r"\bottomrule\end{tabular}"); T["TabDayTwoRO"] = "\n".join(L)
with open(os.path.join(HERE, "supp_day2_tables.tex"), "w", encoding="utf-8") as f:
    for k, v in T.items(): f.write(f"\\newcommand{{\\{k}}}{{%\n{v}}}\n\n")
print("wrote", list(T))
