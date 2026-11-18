# -*- coding: utf-8 -*-
"""Depolarizing noise at eta = 1/sqrt(3): K (averaged over syndromes) stays <= 1 for random codes, while the
syndrome-conditioned quantity 2^k sum_L P(L|e)^2 exceeds 1 for some syndromes, so no syndrome-by-syndrome bound holds.
Usage: python stabilizer_depolarizing.py [codes per m]   (the paper: 150)"""
import sys
import numpy as np
import stim
from stabilizer_K import symp, code_from_tableau, joint_table

rng = np.random.default_rng(11); eta = 1 / np.sqrt(3); ncodes = int(sys.argv[1]) if len(sys.argv) > 1 else 150
for m in range(2, 8):
    digs, X, Zb = symp(m); wK = 0; wE = 0
    for t in range(ncodes):
        k = int(rng.integers(1, m)); T = stim.Tableau.random(m); stabs, logs = code_from_tableau(T, m, k)
        P = joint_table((eta, eta, eta), m, k, stabs, logs, digs, X, Zb); q = P.sum(1); ok = q > 0
        col = (P[ok] ** 2).sum(1) / q[ok] ** 2
        wK = max(wK, 2 ** k * np.sum(q[ok] * col)); wE = max(wE, 2 ** k * col.max())
    print(f"m={m}: max K {wK:.6f};  max over codes and syndromes of 2^k sum_L P(L|e)^2 = {wE:.4f}")
