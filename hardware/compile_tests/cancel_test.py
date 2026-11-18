# -*- coding: utf-8 -*-
"""Does the cloud optimizer delete (cancel) most of our circuits? (2026-11-18)
All circuits are the identity in theory (ideal output 000000 on the path 119-128-138-129-139-130):
  A_k : H H . (CZ 119-128) x 2k . H H              k = 0, 1, 4, 16   (0 .. 32 CZ)
  B_k : (rep5 prep . rep5 un-prep) x k             k = 1, 2, 4       (10 .. 40 CZ)
If the hardware really executes them, the error must grow with the number of CZ. If it stays flat near the
readout floor with optimization ON, the optimizer is cancelling the circuit before it reaches the chip.
Modes: raw_on  / raw_on_barrier (BARRIER after every gate, optimization ON) /
native_off (our RPhi + virtual-Z compile, optimization OFF).  Cost: 7 x 3 circuits x 1000 shots, ~1 s.
Usage: python cancel_test.py"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "..", "data"))
from circuits import to_qprog, native_1q
from pyqpanda3 import qcloud
from pyqpanda3.core import QProg, H, RZ, RY, RX, X, CZ, BARRIER, RPHI, measure
lay = json.load(open(os.path.join(HERE, "layout_6q.json"))); phys = lay["physical"]; NQ = len(phys)
prep = [(g, tuple(q), tuple(p)) for g, q, p in lay["prep_rep"]]
cases = []
for k in (0, 1, 4, 16):
    cases.append((f"A  {2 * k:2d} CZ (pairs on 119-128)", [("H", (0,), ()), ("H", (1,), ())] + [("CZ", (0, 1), ())] * (2 * k)
                  + [("H", (0,), ()), ("H", (1,), ())]))
for k in (1, 2, 4):
    cases.append((f"B  {10 * k:2d} CZ (rep5 prep+unprep x{k})", (prep + list(reversed(prep))) * k))


def qprog_barrier(ops):
    prog = QProg()
    for name, qs, ps in ops:
        q = [phys[x] for x in qs]
        g = {"H": lambda: H(q[0]), "X": lambda: X(q[0]), "CZ": lambda: CZ(q[0], q[1]),
             "RZ": lambda: RZ(q[0], ps[0]), "RY": lambda: RY(q[0], ps[0]), "RX": lambda: RX(q[0], ps[0])}[name]()
        prog << g << BARRIER(phys)
    prog << measure(phys, list(range(NQ)))
    return prog


key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
be = qcloud.QCloudService(api_key=key).backend("WK_C180")
modes = {"raw_on": ([to_qprog(o, NQ, phys) for _, o in cases], True),
         "raw_on_barrier": ([qprog_barrier(o) for _, o in cases], True),
         "native_off": ([to_qprog(native_1q(o), NQ, phys) for _, o in cases], False)}
out = {}
for mode, (progs, opt) in modes.items():
    o = qcloud.QCloudOptions(); o.set_amend(False); o.set_mapping(False); o.set_optimization(opt)
    try:
        jid = be.run(progs, 1000, o).job_id()
    except Exception as e:
        print(f"mode={mode}: submission failed: {e}"); continue
    print(f"mode={mode}: job {jid}", flush=True); t0 = time.time()
    while True:
        st = qcloud.QCloudJob(jid).status(); st = getattr(st, "name", str(st))
        if st in ("FINISHED", "FAILED") or time.time() - t0 > 1800:
            break
        time.sleep(10)
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
    out[mode] = dict(job=jid, res=res)
print(f"\n{'circuit':34s} " + "  ".join(f"{m:>22s}" for m in out) + "\n" + " " * 35 + "  ".join(f"{'P(000000)  mean bit err':>22s}" for _ in out))
for i, (name, _) in enumerate(cases):
    row = []
    for m, v in out.items():
        d = v["res"][i]; tot = sum(float(x) for x in d.values()); p0 = 0.0; be_ = 0.0
        for k_, x in d.items():
            s = str(k_).strip()[-NQ:].zfill(NQ); p0 += float(x) * (s == "0" * NQ) / tot; be_ += float(x) * s.count("1") / NQ / tot
        row.append(f"{p0:10.3f} {be_:11.3f}")
    print(f"{name:34s} " + "  ".join(f"{r_:>22s}" for r_ in row))
json.dump(out, open(os.path.join(DATA, f"cancel_test_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1, default=str)
