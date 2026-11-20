"""Classical duality inequality behind CSS codes with independent X/Z noise:
F(A,B;b) = 2^k sum_{(s,l)} P(s,l)^2 / P(s), where e ~ Bernoulli((1-b)/2)^n, s = e.A (basis of A), (s,l) = e.B (basis of B), A < B, k = dim B - dim A.
Claim: F(A,B;b) * F(B^perp, A^perp; a) <= 2^k whenever (1+a^2)(1+b^2) <= 2."""
import numpy as np, itertools
rng = np.random.default_rng(2)
def rref_basis(M):
    M = M.copy() % 2; r = 0; rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return M[:r]
def perp(B, n):
    # basis of {x: B x = 0}
    if len(B) == 0: return np.eye(n, dtype=np.int64)
    R = rref_basis(B); piv = [int(np.argmax(r)) for r in R]; free = [c for c in range(n) if c not in piv]; out = []
    for f in free:
        x = np.zeros(n, dtype=np.int64); x[f] = 1
        for r, pc in zip(R, piv): x[pc] = r[f]
        out.append(x)
    return np.array(out, dtype=np.int64).reshape(-1, n)
def F(Abasis, Bbasis, b, n):
    """Bbasis rows: first rows span A (given by Abasis), extended to B."""
    q = (1 - b) / 2; E = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    w = E.sum(1); prob = q ** w * (1 - q) ** (n - w)
    ext = extend(Abasis, Bbasis)
    rA = len(Abasis); k = len(ext)
    s = (E @ Abasis.T % 2) @ (1 << np.arange(rA)) if rA else np.zeros(len(E), dtype=np.int64)
    l = (E @ ext.T % 2) @ (1 << np.arange(k)) if k else np.zeros(len(E), dtype=np.int64)
    J = np.bincount(s * (1 << k) + l, weights=prob, minlength=(1 << rA) * (1 << k)).reshape(1 << rA, 1 << k)
    Ps = J.sum(1); ok = Ps > 0
    return 2 ** k * np.sum((J[ok] ** 2).sum(1) / Ps[ok])
def extend(A, B):
    """rows of B completing A to a basis of span(A,B)."""
    cur = rref_basis(A) if len(A) else np.zeros((0, B.shape[1]), dtype=np.int64); ext = []
    for r in B:
        M = np.vstack([cur, r[None]]) if len(cur) else r[None]
        if len(rref_basis(M)) > len(cur): cur = rref_basis(M); ext.append(r)
    return np.array(ext, dtype=np.int64).reshape(-1, B.shape[1])
