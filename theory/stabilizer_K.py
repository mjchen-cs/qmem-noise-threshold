# -*- coding: utf-8 -*-
"""K for stabilizer-code memories, W = Pi_C / 2^k, as a classical quantity:
K = 2^k sum_e sum_L P(e,L)^2 / q_e = 2^k 2^{-H2(L|E)}   (e: syndrome of the Pauli error, L: logical class).
Run as a script to cross-check against the general formula (kfunction.K_reduced) on random codes."""
import itertools
import numpy as np
import stim
from kfunction import pauli_probs as pvec


def symp(m):
    """all 4^m Paulis as (x, z) bit vectors; index = base-4 digits (0=I 1=X 2=Y 3=Z), qubit 0 most significant"""
    digs = np.array(list(itertools.product(range(4), repeat=m)))
    x = ((digs == 1) | (digs == 2)).astype(np.uint8); z = ((digs == 2) | (digs == 3)).astype(np.uint8)
    return digs, x, z


def code_from_tableau(T, m, k):
    """stabilizers = images of Z_k..Z_{m-1}; logicals = images of X_0..X_{k-1}, Z_0..Z_{k-1}"""
    gens = [T.z_output(j) for j in range(k, m)]; logs = [T.x_output(j) for j in range(k)] + [T.z_output(j) for j in range(k)]

    def xz(p):
        a = np.array([int(p[i]) for i in range(m)]); return ((a == 1) | (a == 2)).astype(np.uint8), ((a == 2) | (a == 3)).astype(np.uint8)
    return [xz(g) for g in gens], [xz(l) for l in logs]


def joint_table(lam, m, k, stabs, logs, digs, X, Zb):
    """P[e, L]: joint distribution of syndrome and logical class of the Pauli error"""
    p = pvec(lam); pm = np.prod(p[digs], axis=1)
    comm = lambda ax, az: ((X @ az + Zb @ ax) % 2).astype(np.int64)
    syn = np.zeros(len(pm), np.int64)
    for j, (ax, az) in enumerate(stabs): syn |= comm(ax, az) << j
    lab = np.zeros(len(pm), np.int64)
    for j, (ax, az) in enumerate(logs): lab |= comm(ax, az) << j
    return np.bincount(syn * (1 << (2 * k)) + lab, weights=pm, minlength=(1 << (m - k)) * (1 << (2 * k))).reshape(1 << (m - k), 1 << (2 * k))


def K_code(lam, m, k, stabs, logs, digs, X, Zb):
    P = joint_table(lam, m, k, stabs, logs, digs, X, Zb); q = P.sum(1); mask = q > 0
    return float(2 ** k * np.sum((P[mask] ** 2).sum(1) / q[mask]))


if __name__ == "__main__":
    from kfunction import K_reduced
    rng = np.random.default_rng(0); worst = 0
    for m, k in ((2, 1), (3, 1), (3, 2), (4, 2)):
        digs, X, Zb = symp(m)
        for t in range(3):
            T = stim.Tableau.random(m); stabs, logs = code_from_tableau(T, m, k)
            lam = tuple(rng.uniform(-0.9, 0.9, 3))
            if pvec(lam).min() < 0: continue
            Pi = np.eye(2 ** m, dtype=complex)
            for j in range(k, m):
                S = T.z_output(j).to_unitary_matrix(endian="big"); Pi = Pi @ (np.eye(2 ** m) + S) / 2
            worst = max(worst, abs(K_code(lam, m, k, stabs, logs, digs, X, Zb) - K_reduced(Pi / np.trace(Pi).real, lam, m)))
    print(f"classical formula vs general formula on random codes (m <= 4): max deviation {worst:.2e}")
