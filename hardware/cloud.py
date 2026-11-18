# -*- coding: utf-8 -*-
"""Execution back ends: Origin Quantum cloud (pyqpanda3) and an offline dry run with exact simulation plus
symmetric readout flips. The API key is read from the environment variable ORIGIN_API_KEY and never written."""
import os, json, time
import numpy as np
from circuits import outcome_probs, to_qprog, native_1q

BACKEND = "WK_C180"


class JobFailed(RuntimeError):
    """The cloud reported FAILED (not billed); the circuits are re-submitted on the next run."""


def fetch_chip(backend, out_f):
    from pyqpanda3 import qcloud
    key = os.environ.get("ORIGIN_API_KEY")
    if not key:
        raise SystemExit("set ORIGIN_API_KEY first")
    be = qcloud.QCloudService(api_key=key).backend(backend)
    ci = be.chip_info()

    def call(o, name):
        try:
            return getattr(o, name)()
        except Exception:
            return None
    two = []
    for d in (call(ci, "double_qubits_info") or []):
        qs = call(d, "get_qubits")
        if qs is not None:
            two.append(dict(qubits=[int(qs[0]), int(qs[1])], fidelity=call(d, "get_fidelity")))
    one = []
    for s in (call(ci, "single_qubit_info") or []):
        one.append(dict(qubit=call(s, "get_qubit_id"), gate_fidelity=call(s, "get_single_gate_fidelity"),
                        readout_fidelity=call(s, "get_readout_fidelity"), t1=call(s, "get_t1"), t2=call(s, "get_t2")))
    snap = dict(backend=backend, available_qubits=[int(q) for q in ci.available_qubits()],
                double_qubits=two, single_qubits=one, fetched=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(snap, open(out_f, "w"), indent=1)
    print(f"snapshot: {len(snap['available_qubits'])} qubits, {len(two)} couplers -> {out_f}")



class DryRunner:
    """Offline rehearsal: exact outcome probabilities (independent blocks simulated separately) plus symmetric
    readout flips (one flip probability per logical qubit)."""
    def __init__(self, flips, seed=0):
        self.flips, self.rng, self.jobs = flips, np.random.default_rng(seed), {}

    def submit(self, recs, shots):
        jid = f"dry{len(self.jobs)}"
        out = []
        for r in recs:
            nq = r["nq"]
            if r.get("meta", {}).get("blocks"):     # independent blocks (MS): simulate each block on its own
                bits = np.zeros((shots, nq), dtype=np.int64)
                for blk in r["meta"]["blocks"]:
                    pos = {q: t for t, q in enumerate(blk)}
                    sub = [(g, tuple(pos[q] for q in qs), p_) for g, qs, p_ in r["ops"] if g != "BARRIER" and all(q in pos for q in qs)]
                    pb = outcome_probs(sub, len(blk)); ib = self.rng.choice(len(pb), size=shots, p=pb / pb.sum())
                    bits[:, blk] = (ib[:, None] >> np.arange(len(blk) - 1, -1, -1)) & 1
            else:
                p = outcome_probs(r["ops"], nq)
                idx = self.rng.choice(len(p), size=shots, p=p / p.sum())
                bits = (idx[:, None] >> np.arange(nq - 1, -1, -1)) & 1
            bits ^= (self.rng.random(bits.shape) < np.array(self.flips)[None, :])
            cnt = {}
            for row in bits:
                k = "".join(str(b) for b in row[::-1])              # cbit 0 rightmost, as pyqpanda3
                cnt[k] = cnt.get(k, 0) + 1
            out.append(cnt)
        self.jobs[jid] = (out, time.time())
        return jid

    def fetch(self, jid):
        out, t = self.jobs[jid]
        return out, dict(submitted=t, finished=time.time())


class OriginRunner:
    def __init__(self, phys, backend=BACKEND, amend=False, cloud_opt=False):
        """phys: physical qubit of every logical qubit. amend: server-side readout correction (off: raw counts;
        it rounds and truncates quasi-probabilities, see the Supplementary Information). cloud_opt: the cloud
        circuit optimization (off: it cancels gates across CZ and miscompiled circuits with injected Paulis);
        circuits are always compiled locally to native gates (circuits.native_1q)."""
        from pyqpanda3 import qcloud
        key = os.environ.get("ORIGIN_API_KEY")
        if not key:
            raise SystemExit("set ORIGIN_API_KEY first (the key is never written to disk)")
        self.qc = qcloud
        self.backend = qcloud.QCloudService(api_key=key).backend(backend)
        o = qcloud.QCloudOptions()
        o.set_amend(bool(amend)); o.set_mapping(False); o.set_optimization(bool(cloud_opt))
        print(f"  [OriginRunner] amend={amend}  cloud optimization={cloud_opt}  native compilation", flush=True)
        self.options, self.phys = o, list(phys)
        self.n_expected, self.shots_of = {}, {}

    def submit(self, recs, shots):
        progs = [to_qprog(native_1q(r["ops"]), r["nq"], self.phys) for r in recs]
        for attempt in range(3):
            try:
                jid = self.backend.run(progs, shots, self.options).job_id()
                self.n_expected[jid], self.shots_of[jid] = len(recs), shots
                return jid
            except Exception as e:
                print(f"  submit retry {attempt + 1}: {e}", flush=True); time.sleep(15 * (attempt + 1))
        raise RuntimeError("submission failed repeatedly")

    def status(self, jid):
        try:
            st = self.qc.QCloudJob(jid).status()
            return getattr(st, "name", str(st))
        except Exception as e:
            return f"status query failed ({e})"

    def error(self, jid):
        try:
            return self.qc.QCloudJob(jid).query().error_message()
        except Exception:
            return "?"

    def fetch(self, jid):
        job = self.qc.QCloudJob(jid)
        t0 = time.time(); last = None; polls = 0
        while True:
            name = self.status(jid); polls += 1
            if name == "FINISHED":
                break
            if polls % 6 == 0 and name not in ("FAILED",):
                # the status endpoint sometimes keeps reporting COMPUTING for a job the console shows as finished:
                # every 2 minutes try to read the result directly and accept it if it is complete
                try:
                    r_ = job.result(); c_ = []
                    for g_ in ("get_counts_list", "get_probs_list"):
                        try:
                            c_ = getattr(r_, g_)() or []
                        except Exception:
                            c_ = []
                        if c_:
                            break
                    if len(c_) == self.n_expected.get(jid, 1):
                        print(f"    {jid}: status says {name}, but the complete result is available -> collecting",
                              flush=True)
                        break
                except Exception:
                    pass
            if name == "FAILED":
                raise JobFailed(f"job {jid} failed: {self.error(jid)}")
            if name != last or time.time() - t0 > 60 and int(time.time() - t0) % 300 < 20:
                print(f"    {jid}: {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                last = name
            time.sleep(20)
        res = job.result()
        n_exp = self.n_expected.get(jid, 1)
        counts = []
        try:
            counts = res.get_counts_list() or []
        except Exception:
            pass
        if len(counts) != n_exp:                 # with readout mitigation the cloud may return probabilities only
            try:
                probs = res.get_probs_list() or []
                if len(probs) == n_exp:
                    counts = [{k: int(round(float(v) * self.shots_of[jid])) for k, v in p.items()} for p in probs]
            except Exception:
                pass
        if len(counts) != n_exp and n_exp == 1:
            for getter in ("get_counts", "get_probs"):
                try:
                    c = getattr(res, getter)()
                    if c:
                        counts = [c if getter == "get_counts" else
                                  {k: int(round(float(v) * self.shots_of[jid])) for k, v in c.items()}]
                        break
                except Exception:
                    pass
        if len(counts) != n_exp:
            raise RuntimeError(f"job {jid}: expected {n_exp} results, got {len(counts)}")
        info = dict(polled_s=time.time() - t0)
        info["result_methods"] = [x for x in dir(res) if not x.startswith("_")]
        for attr in ("mapping_qprog", "get_mapping_qprog", "mapping_qubits", "get_mapping", "origin_ir", "compiled_prog"):
            try:
                v = getattr(res, attr)
                info[attr] = str(v() if callable(v) else v)[:200000]       # compiled program / physical qubits
            except Exception:
                pass
        try:
            q = job.query()
            info["query_methods"] = [x for x in dir(q) if not x.startswith("_")]
            for attr in ("mapping_qubits", "get_mapping", "compiled_prog", "qubits", "get_qubits"):
                try:
                    v = getattr(q, attr); info["query_" + attr] = str(v() if callable(v) else v)[:200000]
                except Exception:
                    pass
        except Exception:
            pass
        for attr in ("execution_time", "running_time", "time", "get_running_time"):  # noqa   # record whatever is offered
            try:
                v = getattr(res, attr); info[attr] = v() if callable(v) else v
            except Exception:
                pass
        return counts, info
