# -*- coding: utf-8 -*-
"""Circuits as plain gate lists [(name, qubits, params)] on logical qubits, an exact state-vector simulator,
native compilation for the Wukong 180 processor (RPhi pulses with virtual-Z frames carried through CZ) and the
translation to pyqpanda3. Logical qubit 0 is the most significant bit of every outcome index."""
import numpy as np


def sample_commutant(P, rng):
    """Uniformly random Pauli string R (indices 0=I 1=X 2=Y 3=Z) that commutes with P."""
    P = np.asarray(P)
    while True:
        R = rng.integers(4, size=len(P))
        anti = sum(1 for p, r in zip(P, R) if p and r and p != r)
        if anti % 2 == 0:
            return [int(x) for x in R]


PAULI_GATES = {0: [], 1: [("X",)], 2: [("RZ", np.pi), ("X",)], 3: [("RZ", np.pi)]}   # up to global phases


def pauli_ops(R, qubits):
    ops = []
    for r, q in zip(R, qubits):
        for g in PAULI_GATES[int(r)]:
            ops.append((g[0], (q,), (g[1],) if len(g) > 1 else ()))
    return ops


# ------------------------------------------------------------------ numpy reference simulator (verification)
_G1 = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "H": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "SDG": np.diag([1, -1j]),
}


def _rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def _ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], complex)
def _rx(t): return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]], complex)


def apply_ops(psi, ops, nq):
    psi = psi.reshape([2] * nq)
    for name, qs, ps in ops:
        if name in _G1 or name in ("RZ", "RY", "RX", "RPHI"):
            g = _G1[name] if name in _G1 else (_rphi(*ps) if name == "RPHI" else {"RZ": _rz, "RY": _ry, "RX": _rx}[name](ps[0]))
            psi = np.moveaxis(np.tensordot(g, np.moveaxis(psi, qs[0], 0), axes=1), 0, qs[0])
        elif name == "CRX":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1] = np.tensordot(_rx(ps[0]), psi[1], axes=1)
            psi = np.moveaxis(psi, (0, 1), (c, t))
        elif name == "CNOT":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1] = psi[1][::-1].copy()
            psi = np.moveaxis(psi, (0, 1), (c, t))
        elif name == "CZ":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1, 1] = -psi[1, 1]
            psi = np.moveaxis(psi, (0, 1), (c, t))
        elif name == "BARRIER":
            pass
        else:
            raise ValueError(name)
    return psi.reshape(-1)


def outcome_probs(ops, nq):
    psi = np.zeros(2 ** nq, complex); psi[0] = 1
    return np.abs(apply_ops(psi, ops, nq)) ** 2           # index: qubit 0 = most significant bit


# ------------------------------------------------------------------ pyqpanda3 translation
def to_qprog(ops, nq, phys=None):
    """Translate to a pyqpanda3 QProg. phys maps logical -> physical qubit (no server mapping);
    classical bit k always holds logical qubit k."""
    from pyqpanda3.core import QProg, H, S, X, RZ, RY, RX, CRX, CNOT, CZ, measure, RZ as _RZ
    phys = list(range(nq)) if phys is None else list(phys)
    prog = QProg()
    for name, qs, ps in ops:
        q = [phys[x] for x in qs]
        if name == "H": prog << H(q[0])
        elif name == "S": prog << S(q[0])
        elif name == "SDG": prog << _RZ(q[0], -np.pi / 2)            # S^dag up to a global phase
        elif name == "X": prog << X(q[0])
        elif name == "RZ": prog << RZ(q[0], float(ps[0]))
        elif name == "RY": prog << RY(q[0], float(ps[0]))
        elif name == "RX": prog << RX(q[0], float(ps[0]))
        elif name == "RPHI":
            from pyqpanda3.core import RPHI
            prog << RPHI(q[0], float(ps[0]), float(ps[1]))
        elif name == "CRX": prog << CRX(q[0], q[1], float(ps[0]))
        elif name == "CNOT": prog << CNOT(q[0], q[1])
        elif name == "CZ": prog << CZ(q[0], q[1])
        elif name == "BARRIER":
            from pyqpanda3.core import BARRIER
            prog << BARRIER(list(q))
        else: raise ValueError(name)
    prog << measure([phys[k] for k in range(nq)], list(range(nq)))
    return prog


def counts_to_logical(counts, nq):
    """pyqpanda3 count keys are bit strings with cbit 0 as the rightmost character; return {index: count} with
    logical qubit 0 as the most significant bit (the convention of outcome_probs)."""
    out = {}
    for k, v in counts.items():
        k = str(k).strip()
        if k.lower().startswith("0x"):
            k = format(int(k, 16), "b")
        elif not set(k) <= {"0", "1"}:
            k = format(int(k), "b")                        # decimal outcome index
        k = k[-nq:].zfill(nq)
        bits = [int(ch) for ch in reversed(k)]          # bits[c] = value of cbit c = logical qubit c
        idx = 0
        for c in range(nq):
            idx = (idx << 1) | bits[c]
        out[idx] = out.get(idx, 0) + int(v)
    return out


_1Q_NAMES = ("X", "H", "S", "SDG", "RZ", "RY", "RX")


def _mat1(name, ps):
    if name in _G1:
        return _G1[name]
    return {"RZ": _rz, "RY": _ry, "RX": _rx}[name](ps[0])


# Exact single-qubit fusion to RZ RY RZ (used only by the compilation control tests)
def _zyz(U, tol=1e-9):
    U = U / np.sqrt(np.linalg.det(U))          # SU(2)
    b = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
    if abs(U[1, 0]) < tol:                      # diagonal
        a, c = 2 * np.angle(U[1, 1]), 0.0
    elif abs(U[0, 0]) < tol:                    # anti-diagonal
        a, c = 2 * np.angle(U[1, 0]), 0.0
    else:
        s, d = 2 * np.angle(U[1, 1]), 2 * np.angle(U[1, 0])
        a, c = (s + d) / 2, (s - d) / 2
    out = []
    for name, t in (("RZ", c), ("RY", b), ("RZ", a)):
        t = (t + np.pi) % (2 * np.pi) - np.pi
        if abs(t) > tol and abs(abs(t) - 2 * np.pi) > tol:
            out.append((name, t))
    return out


def fuse_1q(ops):
    pend, out = {}, []

    def flush(q):
        U = pend.pop(q, None)
        if U is not None:
            out.extend((n, (q,), (t,)) for n, t in _zyz(U))

    for name, qs, ps in ops:
        if name in _1Q_NAMES:
            q = qs[0]
            pend[q] = _mat1(name, ps) @ pend.get(q, np.eye(2, dtype=complex))
        else:
            for q in qs:
                flush(q)
            out.append((name, qs, ps))
    for q in sorted(pend):
        flush(q)
    return out



# ---------------------------------------------------------------------------
# Native compilation with virtual Z (2026-11-18). Wukong's native 1q gate is RPhi(theta, phi) =
# exp(-i theta/2 (cos phi X + sin phi Y))  (checked against pyqpanda3.core.RPHI.matrix()).
# Each run of 1q gates U = RZ(a) RY(b) RZ(c) = RZ(a+c) RPHI(b, pi/2 - c): one physical pulse, and the trailing
# RZ(a+c) is carried forward as a frame (it commutes with CZ) and folded into the next run on that qubit.
# The frame left at the end is dropped: it is diagonal, so Z-basis outcome probabilities are unchanged.
# Only CZ is crossed by the frame; before any other 2q gate the frame is emitted as an explicit RZ (exact).
def _rphi(theta, phi):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s * np.exp(-1j * phi)], [-1j * s * np.exp(1j * phi), c]], complex)


def native_1q(ops, tol=1e-9):
    pend, out = {}, []

    def emit(q, keep_frame):
        U = pend.pop(q, None)
        if U is None:
            return
        U = U / np.sqrt(np.linalg.det(U))
        b = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
        if abs(U[1, 0]) < tol:
            a, c = 2 * np.angle(U[1, 1]), 0.0
        elif abs(U[0, 0]) < tol:
            a, c = 2 * np.angle(U[1, 0]), 0.0
        else:
            s_, d_ = 2 * np.angle(U[1, 1]), 2 * np.angle(U[1, 0])
            a, c = (s_ + d_) / 2, (s_ - d_) / 2
        if b > tol:
            out.append(("RPHI", (q,), (float(b), float((np.pi / 2 - c + np.pi) % (2 * np.pi) - np.pi))))
        else:
            a, c = a + c, 0.0      # pure Z rotation
        f = a + c
        if keep_frame:
            pend[q] = _rz(f)
        else:
            f = (f + np.pi) % (2 * np.pi) - np.pi
            if abs(f) > tol:
                out.append(("RZ", (q,), (float(f),)))

    for name, qs, ps in ops:
        if name in _1Q_NAMES or name == "RPHI":
            q = qs[0]
            pend[q] = _mat1(name, ps) @ pend.get(q, np.eye(2, dtype=complex)) if name != "RPHI" \
                else _rphi(*ps) @ pend.get(q, np.eye(2, dtype=complex))
        elif name in ("CZ", "BARRIER"):
            for q in qs:
                emit(q, keep_frame=True)
            out.append((name, qs, ps))
        else:
            for q in qs:
                emit(q, keep_frame=False)
            out.append((name, qs, ps))
    for q in sorted(pend):
        emit(q, keep_frame=True)   # final frame dropped
    pend.clear()
    return out


def check_native(ops, nq):
    """total-variation distance between the Z-basis outcome distributions of ops and native_1q(ops)."""
    return 0.5 * float(np.abs(outcome_probs(ops, nq) - outcome_probs(native_1q(ops), nq)).sum())


# ------------------------------------------------------------------ memories (logical qubits of one site)
# Site = chain s - m1 - m2 - m3 (s: system qubit). Gate lists on logical qubits 0..3.
# bare: Bell pair on (s, m1). rep3: s entangled with the phase-flip repetition code on (m1, m2, m3), chain encoder.
# After the injected noise the encoder is applied in reverse (decoder); then s and m1 hold the Bell-measurement
# outcome of the (logical) pair and m2, m3 hold the syndrome.
ENCODERS = {
    "bare": [('H', (0,), ()), ('H', (1,), ()), ('CZ', (0, 1), ()), ('H', (1,), ())],
    "rep3": [('H', (0,), ()), ('H', (1,), ()), ('CZ', (0, 1), ()), ('H', (1,), ()), ('H', (2,), ()), ('CZ', (1, 2), ()), ('H', (2,), ()), ('H', (3,), ()), ('CZ', (2, 3), ()), ('H', (3,), ()), ('H', (1,), ()), ('H', (2,), ()), ('H', (3,), ())],
}
