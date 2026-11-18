# -*- coding: utf-8 -*-
"""Is the injected Z (RZ(pi), as used by pauli_ops) a pure phase flip on every qubit of the rescue path? (2026-11-17)
For each qubit q of the path and each Z implementation (RZ(pi) as in circuits.pauli_ops, the native Z gate, S S), two circuits:
  A  H - Z - H - measure   -> must give 1 (|+> -> |->)
  B  X - Z - measure       -> must give 1 (Z leaves |1> unchanged; a 0 means the 'Z' acted as a bit flip)
and the references H - H (-> 0) and X (-> 1). Run with and without the cloud's single-qubit gate optimization.
Cost: 6 qubits x 8 circuits x 2 options x 500 shots, about 1-2 s of QPU time.
Usage: python zgate_test.py [--qubits 119,128,138,129,139,130]"""
import os, sys, time, json, argparse
import numpy as np
DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
from pyqpanda3.core import QProg, H, X, Z, S, RZ, measure
from pyqpanda3 import qcloud
ap = argparse.ArgumentParser(); ap.add_argument("--qubits", default="119,128,138,129,139,130"); ap.add_argument("--shots", type=int, default=500)
a = ap.parse_args(); qs = [int(x) for x in a.qubits.split(",")]
key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
be = qcloud.QCloudService(api_key=key).backend("WK_C180")
impl = {"RZ(pi)": lambda q: [RZ(q, np.pi)], "Z": lambda q: [Z(q)], "S.S": lambda q: [S(q), S(q)]}
progs, labels = [], []
for q in qs:
    for name, zf in impl.items():
        pA = QProg() << H(q)
        for g in zf(q): pA << g
        pA << H(q) << measure(q, 0); progs.append(pA); labels.append((q, name, "A: H-Z-H -> 1"))
        pB = QProg() << X(q)
        for g in zf(q): pB << g
        pB << measure(q, 0); progs.append(pB); labels.append((q, name, "B: X-Z -> 1"))
    progs.append(QProg() << H(q) << H(q) << measure(q, 0)); labels.append((q, "-", "ref H-H -> 0"))
    progs.append(QProg() << X(q) << measure(q, 0)); labels.append((q, "-", "ref X -> 1"))
res = {}
for opt in (True, False):
    o = qcloud.QCloudOptions(); o.set_amend(False); o.set_mapping(False); o.set_optimization(opt)
    jid = be.run(progs, a.shots, o).job_id(); print(f"optimization={opt}: job {jid}", flush=True); t0 = time.time()
    while True:
        st = qcloud.QCloudJob(jid).status(); st = getattr(st, "name", str(st))
        if st in ("FINISHED", "FAILED") or time.time() - t0 > 1800:
            break
        time.sleep(15)
    if st != "FINISHED":
        print("  status", st); continue
    r = qcloud.QCloudJob(jid).result(); out = None
    for g in ("get_probs_list", "get_counts_list"):
        try:
            out = getattr(r, g)()
            if out:
                break
        except Exception:
            pass
    res[opt] = []
    for lab, d in zip(labels, out):
        tot = sum(float(v) for v in d.values()); p1 = sum(float(v) for k, v in d.items() if str(k).strip()[-1] == "1") / tot
        res[opt].append((lab, p1))
for opt, rows in res.items():
    print(f"\n== single-qubit optimization {opt}: P(1)  (A and B should both be close to 1)")
    for (q, name, what), p1 in rows:
        flag = "  <== NOT a pure Z" if what.startswith("B") and p1 < 0.85 else ("  <== phase not applied" if what.startswith("A") and p1 < 0.85 else "")
        print(f"  q{q:3d}  {name:7s} {what:18s} {p1:.3f}{flag}")
json.dump({str(k): v for k, v in res.items()}, open(os.path.join(DATA, f"zgate_test_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1, default=str)
