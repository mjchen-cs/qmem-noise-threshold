# -*- coding: utf-8 -*-
"""Multi-site experiments on Wukong 180 (or offline dry runs).

Every site is a chain s-m1[-m2-m3] of physical qubits (system qubit s; memory m1, or the repetition code m1 m2 m3).
In every circuit each active site independently draws a letter X/Y/Z, a Pauli twirl on s from the commutant of the
letter (this realizes the hidden channel of the learning task on s), and a Pauli error on each memory qubit from the
dephasing channel lambda = (a, a, 1). Encoder, injected errors and decoder are applied per site; all qubits are
measured. Settings (noise levels, memories) are shuffled together so that they share the processor's drift.

Steps
  MS   all sites at once (main experiments: 9 Bell pairs; the coding comparison on one chain; screening runs)
  MSI  each site alone, the others idle (crosstalk control)
  RO   readout calibration: all qubits in |0..0> and |1..1>
Runs are resumable: submitted jobs are recorded in <data>/<STEP>_<mode><tag>/jobs.json and collected on the next call.

Examples
  python run.py --fetch_chip
  python run.py --step MS --paths 94-104,92-102 --kinds bare --a 1.00,0.40,0.80 --circ 4000 --shots 40 --tag _pairs_raw
  python run.py --step MS --paths 111-120-129-138 --kinds bare,rep3 --a 1.00,0.56,0.60 --ro 300 --tag _code_ro
  (add --mode dryrun to rehearse offline)"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from circuits import ENCODERS, sample_commutant, pauli_ops, counts_to_logical
from cloud import BACKEND, JobFailed, DryRunner, OriginRunner, fetch_chip
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))


def pauli_probs(lam):
    lx, ly, lz = lam
    q = np.clip(np.array([(1 + lx + ly + lz) / 4, (1 + lx - ly - lz) / 4, (1 - lx + ly - lz) / 4, (1 - lx - ly + lz) / 4]), 0, None)
    return q / q.sum()


def build(a, rng):
    paths = [[int(q) for q in t.split("-")] for t in a.paths.split(",")]
    N = len(paths); Ls = [len(p) for p in paths]; off = [int(sum(Ls[:j])) for j in range(N)]; nq = sum(Ls)
    kinds = a.kinds.split(",")
    if "rep3" in kinds and min(Ls) < 4:
        raise SystemExit("rep3 needs 4-qubit chains")
    lay = dict(paths=paths, physical=[q for p in paths for q in p], system=[off[j] for j in range(N)],
               m0=[off[j] + 1 for j in range(N)],
               syndrome3=[[off[j] + 2, off[j] + 3] if Ls[j] == 4 else [] for j in range(N)],
               memory_bare=[[off[j] + 1] for j in range(N)],
               memory_rep3=[[off[j] + 1, off[j] + 2, off[j] + 3] if Ls[j] == 4 else [] for j in range(N)])
    blocks = [[off[j] + t for t in range(Ls[j])] for j in range(N)]
    shift = lambda ops, o: [(g, tuple(q + o for q in qs), tuple(p)) for g, qs, p in ops]
    recs = []
    if a.step == "RO":
        for r in range(a.ro):
            for b in (0, 1):
                recs.append(dict(key=f"RO_{b}_{r}", nq=nq, kind="readout", P=[b] * nq, shots=a.shots,
                                 ops=[("X", (q,), ()) for q in range(nq)] if b else [], meta=dict(rep=r, policy=f"ro{b}", active=[])))
        return lay, recs
    avals = [float(v) for v in a.a.split(",")]
    active = [list(range(N))] if a.step == "MS" else [[j] for j in range(N)]
    for kind in kinds:
        for a_ in avals:
            pol = f"{kind}_dz{int(round(100 * a_)):02d}"; pp = pauli_probs((a_, a_, 1.0))
            for act in active:
                for r in range(a.circ):
                    P = [0] * N; ops = []
                    for j in act:
                        P[j] = int(rng.integers(1, 4))
                        ops += shift(ENCODERS[kind], off[j])
                        ops += pauli_ops(sample_commutant([P[j]], rng), [off[j]])
                        mem = lay["memory_" + kind][j]
                        ops += pauli_ops([int(rng.choice(4, p=pp)) for _ in mem], mem)
                    for j in act:
                        ops += list(reversed(shift(ENCODERS[kind], off[j])))
                    tag = "" if a.step == "MS" else f"i{act[0]}_"
                    recs.append(dict(key=f"{a.step}_{pol}_{tag}{r}", nq=nq, kind="quantum", P=P, ops=ops, shots=a.shots,
                                     meta=dict(rep=r, policy=pol, blocks=blocks, active=act)))
    for r in range(a.ro):                                # interleaved readout calibration
        for b in (0, 1):
            recs.append(dict(key=f"{a.step}_ro{b}_{r}", nq=nq, kind="readout", P=[0] * N, shots=a.shots,
                             ops=[("X", (q,), ()) for q in range(nq)] if b else [],
                             meta=dict(rep=r, policy=f"ro{b}", blocks=blocks, active=[])))
    return lay, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["MS", "MSI", "RO"]); ap.add_argument("--mode", choices=["dryrun", "origin"], default="dryrun")
    ap.add_argument("--paths", help="sites as s-m1[-m2-m3], comma separated (physical qubits)")
    ap.add_argument("--kinds", default="bare", help="bare and/or rep3"); ap.add_argument("--a", default="1.00,0.80", help="dephasing parameters a")
    ap.add_argument("--circ", type=int, default=4000); ap.add_argument("--shots", type=int, default=40)
    ap.add_argument("--ro", type=int, default=0, help="interleaved readout-calibration circuits per state (RO: number per state)")
    ap.add_argument("--tag", default=""); ap.add_argument("--batch", type=int, default=150); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--amend", action="store_true", help="server-side readout correction (not recommended)")
    ap.add_argument("--chip", default=os.path.join(DATA, "chip.json")); ap.add_argument("--fetch_chip", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.fetch_chip:
        fetch_chip(BACKEND, a.chip); return
    if not a.step or not a.paths:
        ap.error("--step and --paths are required")
    if a.step == "RO" and a.ro == 0:
        a.ro = 20
    rng = np.random.default_rng(a.seed)
    lay, recs = build(a, rng)
    np.random.default_rng(a.seed + 7).shuffle(recs)
    out_dir = os.path.join(DATA, f"{a.step}_{a.mode}{a.tag}"); os.makedirs(out_dir, exist_ok=True)
    man_f, raw_f, jobs_f = (os.path.join(out_dir, f) for f in ("manifest.json", "raw.json", "jobs.json"))
    if os.path.exists(man_f) and json.load(open(man_f))["layout"]["physical"] != lay["physical"]:
        raise SystemExit("this run folder belongs to other qubits: use a new --tag")
    raw = json.load(open(raw_f)) if os.path.exists(raw_f) else {}
    jobs = json.load(open(jobs_f)) if os.path.exists(jobs_f) else {}
    chip_fetched = json.load(open(a.chip)).get("fetched") if os.path.exists(a.chip) else None
    json.dump(dict(step=a.step, mode=a.mode, layout=lay, args=vars(a), chip_fetched=chip_fetched, amend=a.amend,
                   n_circuits=len(recs), records=[{k: v for k, v in r.items() if k != "ops"} for r in recs]),
              open(man_f, "w"), indent=1)
    print(f"{a.step} [{a.mode}]: {len(recs)} circuits, {sum(r['shots'] for r in recs)} shots on qubits {lay['physical']}", flush=True)
    if a.mode == "origin":
        runner = OriginRunner(lay["physical"], amend=a.amend)
    else:
        flips = [0.0] * len(lay["physical"])
        if os.path.exists(a.chip):
            ro = {int(s["qubit"]): s["readout_fidelity"] for s in json.load(open(a.chip))["single_qubits"]}
            flips = [1 - ro.get(q, 1.0) for q in lay["physical"]]
        runner = DryRunner(flips)
    if a.status:
        for jid, j in jobs.items():
            done = all(k in raw for k in j["keys"])
            print(f"  {jid}: {len(j['keys'])} circuits, submitted {j['submitted']}: "
                  f"{'COLLECTED' if done else (runner.status(jid) if a.mode == 'origin' else 'dryrun')}")
        return
    pending = {k for j in jobs.values() for k in j["keys"]}
    todo = [r for r in recs if r["key"] not in raw and r["key"] not in pending]
    queue = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    while queue:
        part = queue.pop(0)
        try:
            jid = runner.submit(part, a.shots)
        except Exception as e:
            if len(part) == 1:
                raise
            print(f"  submission of {len(part)} circuits failed ({e}); splitting", flush=True)
            queue[:0] = [part[:len(part) // 2], part[len(part) // 2:]]; continue
        jobs[jid] = dict(keys=[r["key"] for r in part], shots=a.shots, submitted=time.strftime("%Y-%m-%d %H:%M:%S"),
                         submitted_unix=time.time())
        json.dump(jobs, open(jobs_f, "w"), indent=1)
        print(f"  submitted {jid}: {len(part)} circuits", flush=True)
    bykey = {r["key"]: r for r in recs}; failed = {}
    for jid, j in list(jobs.items()):
        if all(k in raw for k in j["keys"]):
            continue
        if hasattr(runner, "n_expected"):
            runner.n_expected[jid], runner.shots_of[jid] = len(j["keys"]), j["shots"]
        try:
            counts, info = runner.fetch(jid)
        except JobFailed as e:
            print(f"  {e} -> re-submitted on the next call", flush=True); failed[jid] = dict(j, error=str(e)); continue
        for k, c in zip(j["keys"], counts):
            raw[k] = dict(counts=counts_to_logical(c, bykey[k]["nq"]), job=jid)
        j.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), finished_unix=time.time(), info=info)
        json.dump(raw, open(raw_f, "w")); json.dump(jobs, open(jobs_f, "w"), indent=1, default=str)
        print(f"  collected {jid} ({len(j['keys'])} circuits)", flush=True)
    for jid in failed:
        jobs.pop(jid, None)
    json.dump(jobs, open(jobs_f, "w"), indent=1, default=str)
    print(f"done: {len(raw)}/{len(recs)} circuits stored in {out_dir}")


if __name__ == "__main__":
    main()
