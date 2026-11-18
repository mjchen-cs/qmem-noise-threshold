# -*- coding: utf-8 -*-
"""Maximize K(W) over all m-qubit memory marginals W (all protocols with m memory qubits, any system, any encoding).
Usage: python maximize_K.py m lx,ly,lz restarts [seed]
e.g.   python maximize_K.py 2 0.577,0.577,0.577 10     (on the sphere: expect 1)
       python maximize_K.py 2 0.6,0.6,0.6 10           (outside: expect ((1+sum lam^2)/2)^m)
Uses the operator form K(W) = tr[(R_m (W x N(W)^{-1/2}))^2] with R_m = R_1^{x m}."""
import sys, time, itertools
import numpy as np
from scipy.optimize import minimize
from kfunction import P1, kron, pauli_probs


def setup(lam, m):
    p = pauli_probs(lam)
    T1 = sum(p[a] * np.kron(P1[a], P1[a]) for a in range(4))
    F1 = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            F1[i * 2 + j, j * 2 + i] = 1
    R1t = (T1 @ F1).reshape(2, 2, 2, 2)
    d = 2 ** m; idx = list(itertools.product(range(2), repeat=m)); R = np.zeros((d * d, d * d), complex)
    for ia, a in enumerate(idx):
        for ib, b in enumerate(idx):
            for ja, a2 in enumerate(idx):
                for jb, b2 in enumerate(idx):
                    v = 1.0 + 0j
                    for k in range(m):
                        v *= R1t[a[k], b[k], a2[k], b2[k]]
                        if v == 0: break
                    R[ia * d + ib, ja * d + jb] = v
    Ps = [np.prod([p[i] for i in mu]) for mu in itertools.product(range(4), repeat=m)]
    Ss = [kron(*[P1[i] for i in mu]) for mu in itertools.product(range(4), repeat=m)]
    return R, Ps, Ss


def K_of_W(W, R, Ps, Ss):
    V = sum(pp * S @ W @ S for pp, S in zip(Ps, Ss))
    w, U = np.linalg.eigh((V + V.conj().T) / 2); Zm = (U * np.clip(w, 1e-14, None) ** -0.5) @ U.conj().T
    M = R @ np.kron(W, Zm); return float(np.real(np.trace(M @ M)))


if __name__ == "__main__":
    m = int(sys.argv[1]); lam = tuple(float(x) for x in sys.argv[2].split(",")); restarts = int(sys.argv[3])
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    R, Ps, Ss = setup(lam, m); d = 2 ** m

    def f(v):
        A = (v[:d * d] + 1j * v[d * d:]).reshape(d, d); W = A @ A.conj().T; W /= np.trace(W).real
        return -K_of_W(W, R, Ps, Ss)
    rnd = max(-f(rng.normal(size=2 * d * d)) for _ in range(200))
    best, t0 = -1, time.time()
    for r in range(restarts):
        res = minimize(f, rng.normal(size=2 * d * d) * (0.3 if r % 2 else 1), method="L-BFGS-B", options=dict(maxiter=400))
        best = max(best, -res.fun)
    kap = (1 + sum(x * x for x in lam)) / 2
    print(f"m={m} lam={lam} sum lam^2={2 * kap - 1:.3f}: max over 200 random W {rnd:.4f}; optimized {best:.4f}; "
          f"max(1,kappa)^m = {max(1, kap) ** m:.4f}  ({time.time() - t0:.0f} s)")
