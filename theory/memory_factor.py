# -*- coding: utf-8 -*-
"""Exact per-site factor nu of the hardware protocols (Bell pair 'bare' and phase-flip repetition code 'rep3' with the chain
encoder of hardware/circuits.py) for an ideal processor and independent Pauli noise lambda = (lx, ly, lz) on every memory
qubit. All 4^m memory errors are enumerated; each maps the decoded outcome deterministically, so the class tables q_s,
lambda_{l,s} and the soft factors G_l = sum_s q_s lambda_{l,s}^2 are exact. nu = (G_X G_Y G_Z)^(1/3).
Usage: python memory_factor.py   (prints the ideal crossings nu = 1/2 for the channel families used in the experiments)"""
import itertools
import numpy as np
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
P1 = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
ENC = {"bare": [("H", (0,)), ("H", (1,)), ("CZ", (0, 1)), ("H", (1,))],
       "rep3": [("H", (0,)), ("H", (1,)), ("CZ", (0, 1)), ("H", (1,)), ("H", (2,)), ("CZ", (1, 2)), ("H", (2,)),
                ("H", (3,)), ("CZ", (2, 3)), ("H", (3,)), ("H", (1,)), ("H", (2,)), ("H", (3,))]}
MEM = {"bare": [1], "rep3": [1, 2, 3]}
FAMILIES = {"dz": lambda t: (t, t, 1.0), "dp": lambda t: (t, t, t), "zs": lambda t: (t, t, 0.98),
            "zh": lambda t: (t, t, 0.95), "zm": lambda t: (t, t, 0.90)}


def _apply(psi, ops, nq):
    psi = psi.reshape([2] * nq)
    for g, qs in ops:
        if g == "CZ":
            c, t = qs; psi = np.moveaxis(psi, (c, t), (0, 1)).copy(); psi[1, 1] = -psi[1, 1]; psi = np.moveaxis(psi, (0, 1), (c, t))
        else:
            M = H if g == "H" else P1[g]
            psi = np.moveaxis(np.tensordot(M, np.moveaxis(psi, qs[0], 0), axes=1), 0, qs[0])
    return psi.reshape(-1)


def pauli_probs(lam):
    lx, ly, lz = lam
    q = np.array([(1 + lx + ly + lz) / 4, (1 + lx - ly - lz) / 4, (1 - lx + ly - lz) / 4, (1 - lx - ly + lz) / 4])
    return np.clip(q, 0, None) / np.clip(q, 0, None).sum()


_CACHE = {}
def _outcomes(kind):
    """decoded outcome (s bit, m1 bit, syndrome class) for every memory error string."""
    if kind in _CACHE: return _CACHE[kind]
    nq = 2 + (2 if kind == "rep3" else 0); out = {}
    for err in itertools.product(range(4), repeat=len(MEM[kind])):
        ops = list(ENC[kind]) + [("XYZ"[e - 1], (q,)) for e, q in zip(err, MEM[kind]) if e] + list(reversed(ENC[kind]))
        psi = np.zeros(2 ** nq, complex); psi[0] = 1; y = int(np.argmax(np.abs(_apply(psi, ops, nq)) ** 2))
        b = [(y >> (nq - 1 - i)) & 1 for i in range(nq)]
        out[err] = (b[0], b[1], (b[2] | (b[3] << 1)) if nq == 4 else 0)
    _CACHE[kind] = out; return out


def letters(kind, lam):
    """Exact soft factors (G_X, G_Y, G_Z) (letters in the frame of lambda, as in the hardware analysis)."""
    q = pauli_probs(lam); acc = {}
    for err, (bs, bm, s) in _outcomes(kind).items():
        w = float(np.prod([q[e] for e in err])); z, x = 1 - 2 * bs, 1 - 2 * bm
        for l, chi in enumerate((z, z * x, x)):
            a = acc.setdefault((l, s), [0.0, 0.0]); a[0] += w; a[1] += w * chi
    return np.array([sum(v[1] ** 2 / v[0] for (ll, s), v in acc.items() if ll == l and v[0] > 1e-15) for l in range(3)])


def nu(kind, lam):
    return float(np.exp(np.mean(np.log(np.maximum(letters(kind, lam), 1e-12)))))


def crossing(kind, fam, factor=1.0, lo=0.02, hi=1.0):
    """t* with factor * nu(kind, FAMILIES[fam](t)) = 1/2 (nan if none in [lo, hi])."""
    from scipy.optimize import brentq
    f = lambda t: factor * nu(kind, FAMILIES[fam](t)) - 0.5
    return float(brentq(f, lo, hi)) if f(lo) * f(hi) < 0 else float("nan")


if __name__ == "__main__":
    for fam in FAMILIES:
        b, r = crossing("bare", fam), crossing("rep3", fam)
        print(f"{fam}: ideal crossing bare {b:.3f}, rep3 {r:.3f}, shift {b - r:+.3f}")
