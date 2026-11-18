# -*- coding: utf-8 -*-
"""The quantity K = 2^{-H~down_2(S|A)} of a system-memory state after Pauli memory noise.

K_direct(W, lam, m):   from the definition, using a purification of the memory marginal W
K_reduced(W, lam, m):  the reduction theorem, sum_{mu,nu} p_mu p_nu |tr(s_mu W s_nu N(W)^{-1/2})|^2
K_m1_closed(r, n, lam): closed form for one memory qubit, W = (I + r n.sigma)/2
Run as a script to cross-check the three expressions on random inputs."""
import itertools
import numpy as np

I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)
P1 = [I2, X, Y, Z]


def pauli_probs(lam):
    lx, ly, lz = lam
    return np.array([(1 + lx + ly + lz) / 4, (1 + lx - ly - lz) / 4, (1 - lx + ly - lz) / 4, (1 - lx - ly + lz) / 4])


def kron(*a):
    o = np.array([[1.]], complex)
    for x in a:
        o = np.kron(o, x)
    return o


def mpow(A, p, eps=1e-15):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * np.clip(w, eps, None) ** p) @ V.conj().T


def paulis(m):
    return [kron(*[P1[i] for i in mu]) for mu in itertools.product(range(4), repeat=m)]


def channel(W, lam, m):
    p = pauli_probs(lam)
    return sum(np.prod(p[list(mu)]) * S @ W @ S for mu, S in zip(itertools.product(range(4), repeat=m), paulis(m)))


def K_reduced(W, lam, m):
    p = pauli_probs(lam); Zm = mpow(channel(W, lam, m), -0.5)
    Ps = [(np.prod(p[list(mu)]), S) for mu, S in zip(itertools.product(range(4), repeat=m), paulis(m))]
    return float(sum(pa * pb * abs(np.trace(Sa @ W @ Sb @ Zm)) ** 2 for pa, Sa in Ps for pb, Sb in Ps).real)


def K_direct(W, lam, m):
    d = 2 ** m; w, V = np.linalg.eigh(W)
    psi = sum(np.sqrt(max(w[i], 0)) * np.kron(np.eye(d)[i], V[:, i]) for i in range(d))      # system (dim d) x memory
    rho = np.outer(psi, psi.conj()); p = pauli_probs(lam)
    out = sum(np.prod(p[list(mu)]) * np.kron(np.eye(d), S) @ rho @ np.kron(np.eye(d), S)
              for mu, S in zip(itertools.product(range(4), repeat=m), paulis(m)))
    rA = np.trace(out.reshape(d, d, d, d), axis1=0, axis2=2); Yq = np.kron(np.eye(d), mpow(rA, -0.25)); t = Yq @ out @ Yq
    return float(np.trace(t @ t).real)


def K_m1_closed(r, n, lam):
    lam = np.asarray(lam, float); n = np.asarray(n, float) / np.linalg.norm(n)
    L = float(np.sum(lam ** 2 * n ** 2)); q2 = r * r * L; D = 1 - q2; L2 = float(np.sum(lam ** 2))
    M = float(np.sum(lam ** 4 * n ** 2) / L) if L > 0 else 0.0
    return 0.5 * ((1 + r * r - 2 * q2) / D + (1 - r * r) * (L2 / np.sqrt(D) + M * (1 / D - 1 / np.sqrt(D))))


if __name__ == "__main__":
    rng = np.random.default_rng(0); err1 = err2 = 0.0
    for t in range(30):
        while True:
            lam = rng.uniform(-1, 1, 3)
            if pauli_probs(lam).min() >= 0: break
        m = 1 + t % 2; d = 2 ** m
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); W = A @ A.conj().T; W /= np.trace(W).real
        err1 = max(err1, abs(K_reduced(W, lam, m) - K_direct(W, lam, m)))
        if m == 1:
            v = np.real([np.trace(W @ s) for s in (X, Y, Z)]); r = np.linalg.norm(v)
            err2 = max(err2, abs(K_m1_closed(r, v, lam) - K_reduced(W, lam, 1)))
    print(f"max |K_reduced - K_direct| = {err1:.2e};  max |closed form (m=1) - K_reduced| = {err2:.2e}")
    # the Bell (maximally mixed marginal) value
    lam = (0.6, 0.6, 0.6); print(f"Bell input, lam={lam}, m=2: K = {K_reduced(np.eye(4) / 4, lam, 2):.6f} = kappa^2 = {((1 + 3 * 0.36) / 2) ** 2:.6f}")
