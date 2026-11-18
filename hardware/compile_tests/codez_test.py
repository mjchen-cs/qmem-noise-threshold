# -*- coding: utf-8 -*-
"""Does a single injected Z inside the real repetition-code circuit act as theory says? (2026-11-17)
Circuit = rep prep (layout_6q.json) - Z (RZ(pi), exactly as pauli_ops) on ONE memory qubit - inverse prep -
measure all 6. Theory (stim): Z on m0 flips bits {m0, m1}; Z on m_j flips {m_j, m_j+1}; Z on the last flips {m_last};
no injection -> all 0. Also the same with X injected (should flip a different, known pattern) as a control.
Done for rep3 and rep5, with the cloud's single-qubit optimization on  and off.
Cost: about 2 x (4 + 6 + 2) x 2 circuits x 500 shots, 1-2 s of QPU time.  Usage: python codez_test.py"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "..", "data"))
from circuits import to_qprog, pauli_ops, apply_ops, outcome_probs, fuse_1q, native_1q
from pyqpanda3 import qcloud
lay = json.load(open(os.path.join(HERE, "layout_6q.json"))); phys = lay["physical"]; NQ = len(phys)
key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
be = qcloud.QCloudService(api_key=key).backend("WK_C180")
cases = []
for code, pk, mem in (("rep3", "prep_rep3", lay["memory_rep3"]), ("rep5", "prep_rep", lay["memory_rep"])):
    prep = [(g, tuple(q), tuple(p)) for g, q, p in lay[pk]]
    cases.append((code, "none", prep + list(reversed(prep))))
    for j, q in enumerate(mem):
        cases.append((code, f"Z on m{j} (q{phys[q]})", prep + pauli_ops([3], [q]) + list(reversed(prep))))
    cases.append((code, f"X on m0 (q{phys[mem[0]]})", prep + pauli_ops([1], [mem[0]]) + list(reversed(prep))))
# theory from the exact simulator of circuits.py (logical qubit 0 = most significant bit)
def ideal_bits(ops):
    p = outcome_probs(ops, NQ); idx = int(np.argmax(p))
    return [(idx >> (NQ - 1 - k)) & 1 for k in range(NQ)], float(p.max())
# v2: modes = raw_on / raw_off / fused_off / fused_on  (fused = client-side exact 1q fusion, circuits.fuse_1q)
MODES = os.environ.get("CODEZ_MODES", "native_off,raw_on,fused_off").split(",")
out = {}
for mode in MODES:
    opt = mode.endswith("_on"); comp = {"fused": fuse_1q, "native": native_1q, "raw": lambda x: x}[mode.split("_")[0]]
    progs = [to_qprog(comp(ops), NQ, phys) for _, _, ops in cases]
    o = qcloud.QCloudOptions(); o.set_amend(False); o.set_mapping(False); o.set_optimization(opt)
    jid = be.run(progs, 500, o).job_id(); print(f"mode={mode}: job {jid}", flush=True); t0 = time.time()
    while True:
        st = qcloud.QCloudJob(jid).status(); st = getattr(st, "name", str(st))
        if st in ("FINISHED", "FAILED") or time.time() - t0 > 1800:
            break
        time.sleep(15)
    if st != "FINISHED":
        print("  status", st); continue
    r = qcloud.QCloudJob(jid).result(); res = None
    for g in ("get_probs_list", "get_counts_list"):
        try:
            res = getattr(r, g)()
            if res:
                break
        except Exception:
            pass
    out[mode] = res
for opt, res in out.items():
    print(f"\n== mode {opt}: per-qubit P(bit = 1) in logical order "
          f"(system, m0, m1, ...) ; theory pattern in brackets")
    for (code, what, ops), d in zip(cases, res):
        tot = sum(float(v) for v in d.values()); pb = np.zeros(NQ)
        for k, v in d.items():
            s = str(k).strip()[-NQ:].zfill(NQ)
            for c in range(NQ):
                pb[c] += float(v) * int(s[NQ - 1 - c]) / tot      # cbit c = logical qubit c
        th, _ = ideal_bits(ops)
        bad = any(abs(pb[c] - th[c]) > 0.25 for c in range(NQ))
        print(f"  {code} {what:18s} " + " ".join(f"{x:.2f}" for x in pb) + "   [" + " ".join(map(str, th)) + "]" + ("   <== MISMATCH" if bad else ""))
json.dump({str(k): v for k, v in out.items()}, open(os.path.join(DATA, f"codez_test_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1, default=str)
