# -*- coding: utf-8 -*-
"""Random stabilizer codes against the conjecture: K <= 1 for channels on or inside the sphere sum lambda^2 = 1.
Usage: python stabilizer_scan.py m [seed] [codes]     (the paper: m = 2..7, 300 codes each)"""
import sys
import numpy as np
import stim
from stabilizer_K import pvec, symp, code_from_tableau, K_code

M = int(sys.argv[1]); rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
ncodes = int(sys.argv[3]) if len(sys.argv) > 3 else 300


def rand_lam(r):
    while True:
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * r
        if pvec(v).min() >= -1e-12: return tuple(v)


digs, X, Zb = symp(M); worst_in = 0; worst_out = 0
for t in range(ncodes):
    k = int(rng.integers(1, M + 1)); T = stim.Tableau.random(M); stabs, logs = code_from_tableau(T, M, k)
    lam = rand_lam(1.0 if t % 2 == 0 else rng.uniform(0.6, 1.0))
    worst_in = max(worst_in, K_code(lam, M, k, stabs, logs, digs, X, Zb))
    lam2 = rand_lam(rng.uniform(1.0, 1.6)); kap = (1 + sum(l * l for l in lam2)) / 2
    worst_out = max(worst_out, np.log(K_code(lam2, M, k, stabs, logs, digs, X, Zb)) / np.log(kap) / k)
print(f"m={M}: {ncodes} random codes; max K on/inside the sphere: {worst_in:.6f}; above it, max log K / (k log kappa) = {worst_out:.4f}")
