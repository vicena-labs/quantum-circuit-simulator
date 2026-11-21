"""Hamiltonian simulation: product formulas, qDRIFT, Trotter error bounds and multi-product formulas.

Every circuit approximates exp(-i H t) for a `hamiltonians.Hamiltonian` and is a
Program, so it runs through every solver and, where noted, keeps time as a
shared parameter. Identity terms only add a global phase and are dropped.

    from quantum_circuit_simulator import dynamics, hamiltonians as hm
    h = hm.transverse_field_ising(hm.chain(8), j=1.0, h=0.7)
    circuit = dynamics.suzuki(h, "t", steps=4, order=4)          # time as a parameter
    dynamics.trotter_error_bound(h, 1.0, steps=4, order=2)      # rigorous operator-norm bound
    dynamics.mpf(h, "Z0", 1.0, ks=(2, 3, 4))                     # Richardson-style extrapolation

Conventions: a product formula of order q with r steps has error O(t^(q+1) / r^q)
in operator norm; terms are applied in the Hamiltonian's term order, each as
`hamiltonians.pauli_rotation`.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .hamiltonians import Hamiltonian, load, pauli_rotation, trotter
from .interop import Angle, Program


# ---------------------------------------------------------------- product formulas
def suzuki(hamiltonian, time, steps=1, order=2, initial=None):
    """A Program for exp(-i H time) from `steps` steps of Suzuki's even-order product formula.

    order 2 is the symmetric Strang splitting; order 2k is built from five
    copies of order 2k - 2 (Suzuki, J. Math. Phys. 32, 400 (1991)), so a step
    holds 5^(k-1) Strang steps. time is a number or a parameter name, and
    initial follows `hamiltonians.trotter`, which this calls.
    """
    if order < 2 or order % 2:
        raise ValueError("Suzuki formulas have even order 2, 4, 6, ...")
    return trotter(hamiltonian, time, steps, order, initial)


def qdrift(hamiltonian, time, samples, seed=None, initial=None):
    """A Program for one random qDRIFT circuit approximating exp(-i H time) (Campbell, arXiv:1811.08017).

    With H = sum_j h_j P_j and lambda = sum_j |h_j|, each of the `samples`
    rotations is exp(-i sign(h_j) (lambda time / samples) P_j) with term j
    drawn with probability |h_j| / lambda. The channel averaged over draws
    has diamond-norm error at most 2 lambda^2 time^2 / samples. time is a
    number or a parameter name; seed makes the draw reproducible.
    """
    terms = [(c, p) for c, p in hamiltonian.terms if p]
    weights = np.array([abs(c) for c, _ in terms])
    lam = weights.sum()
    picks = np.random.default_rng(seed).choice(len(terms), size=samples, p=weights / lam)
    tau = Angle(time, lam / samples) if isinstance(time, str) else float(time) * lam / samples
    prep = ([("h", [q], []) for q in range(hamiltonian.n_qubits)] if initial == "plus"
            else list(initial or []))
    body = [g for j in picks for g in pauli_rotation(terms[j][1], tau * (2 * math.copysign(1.0, terms[j][0])))]
    return Program(hamiltonian.n_qubits, prep + body, [time] if isinstance(time, str) else [])


# ---------------------------------------------------------------- Trotter error bounds
def trotter_error_bound(hamiltonian, time, steps=1, order=1):
    """An upper bound on ||S(time/steps)^steps - exp(-i H time)|| for the product formulas of `trotter`.

    Commutator bounds of Childs, Su, Tran, Wiebe and Zhu, "Theory of Trotter
    error with commutator scaling", arXiv:1912.08854, with H = sum_j H_j and
    r = steps:

    - order 1 (their first-order bound): (t^2 / (2r)) sum_{j<k} ||[H_j, H_k]||.
    - order 2 (their second-order bound, for the symmetric formula whose
      middle term is H_1): (t^3 / (12 r^2)) sum_j ||[B_j, [B_j, H_j]]||
      + (t^3 / (24 r^2)) sum_j ||[H_j, [H_j, B_j]]||, with B_j = sum_{k>j} H_k.
      `evolution_step` puts the last term in the middle, so H_1 is the last term.

    Commutators are computed exactly in the Pauli algebra (two anticommuting
    Paulis give [aP, bQ] = 2ab PQ, commuting ones zero); the norm of each
    resulting Pauli sum is bounded by the l1 norm of its merged coefficients.
    """
    t, r = float(time), int(steps)
    terms = [_sum([(c, p)]) for c, p in hamiltonian.terms if p]
    if order == 1:
        total = math.fsum(_norm(_commutator(a, b)) for a, b in combinations(terms, 2))
        return t ** 2 / (2 * r) * total
    if order == 2:
        terms = terms[::-1]   # the middle term first, as in Childs et al.
        outer = inner = 0.0
        for j, h_j in enumerate(terms):
            b_j = _add(terms[j + 1:])
            c = _commutator(b_j, h_j)
            outer += _norm(_commutator(b_j, c))
            inner += _norm(_commutator(h_j, _commutator(h_j, b_j)))
        return t ** 3 / (12 * r ** 2) * outer + t ** 3 / (24 * r ** 2) * inner
    raise ValueError("bounds are implemented for order 1 and 2")


# A Pauli sum is a dict {word: complex coefficient}, a word a sorted tuple of (wire, axis).
_PRODUCT = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
            ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}


def _sum(terms):
    return {tuple(sorted(p.items())): complex(c) for c, p in terms}


def _add(sums):
    out = {}
    for s in sums:
        for word, c in s.items():
            out[word] = out.get(word, 0) + c
    return out


def _commutator(a, b):
    """[A, B] for Pauli sums: only anticommuting pairs contribute, 2 a b P Q each."""
    out = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            phase, word = _multiply(pa, pb)
            if phase.imag != 0:   # P Q = +-i R exactly when P and Q anticommute
                out[word] = out.get(word, 0) + 2 * ca * cb * phase
    return {w: c for w, c in out.items() if abs(c) > 1e-15}


def _multiply(pa, pb):
    """(phase, word) with P_a P_b = phase * word."""
    out, phase = dict(pa), 1
    for q, axis in pb:
        mine = out.get(q)
        if mine is None:
            out[q] = axis
        elif mine == axis:
            del out[q]
        else:
            factor, out[q] = _PRODUCT[(mine, axis)]
            phase *= factor
    return phase, tuple(sorted(out.items()))


def _norm(pauli_sum):
    return math.fsum(abs(c) for c in pauli_sum.values())


# ---------------------------------------------------------------- multi-product formulas
def mpf(hamiltonian, observable, time, ks, order=2, initial=None, **options):
    """<O(time)> from a static multi-product formula over product formulas with k_i steps.

    The estimate is sum_i c_i <O>_{k_i}, where <O>_k is the exact expectation
    on `trotter(hamiltonian, time, k, order)` and c = `mpf_coefficients(ks, order)`
    cancels the leading error orders (Carrera Vazquez, Egger, Ochsner and
    Woerner, arXiv:2207.07272; qiskit-addon-mpf's exact static coefficients).
    Each <O>_k comes from `Hamiltonian.expectation` (exact Pauli cones or the
    state vector); options pass through to it. Returns the estimate, the
    coefficients and every component.
    """
    obs = observable if isinstance(observable, Hamiltonian) else load(observable, hamiltonian.n_qubits)
    coefficients = mpf_coefficients(ks, order)
    components = []
    for k in ks:
        result = obs.expectation(trotter(hamiltonian, time, k, order, initial), **options)
        components.append({"steps": int(k), "value": float(result["value"]), "method": result["method"]})
    value = math.fsum(c * row["value"] for c, row in zip(coefficients, components))
    return {"value": value, "coefficients": coefficients, "components": components, "order": order}


def mpf_coefficients(ks, order=2):
    """Exact static MPF coefficients c for step counts ks, as floats.

    Solves the Vandermonde system sum_i c_i = 1 and sum_i c_i k_i^-(order + 2j) = 0
    for j = 0 .. len(ks) - 2: a symmetric formula of even order has an error
    expansion in k^-order, k^-(order+2), ..., so the combination cancels the
    first len(ks) - 1 of them (Carrera Vazquez et al., arXiv:2207.07272, Eq. 4).
    Solved in exact rational arithmetic, since the system is ill-conditioned.
    """
    from fractions import Fraction
    ks = [int(k) for k in ks]
    if len(set(ks)) != len(ks) or min(ks) < 1:
        raise ValueError("ks must be distinct positive step counts")
    rows = [[Fraction(1)] * len(ks)] + [[Fraction(1, k ** (order + 2 * j)) for k in ks] for j in range(len(ks) - 1)]
    rhs = [Fraction(1)] + [Fraction(0)] * (len(ks) - 1)
    return [float(c) for c in _solve(rows, rhs)]


def _solve(rows, rhs):
    """Gauss-Jordan elimination over Fractions."""
    n = len(rhs)
    a = [row[:] + [b] for row, b in zip(rows, rhs)]
    for i in range(n):
        pivot = next(k for k in range(i, n) if a[k][i] != 0)
        a[i], a[pivot] = a[pivot], a[i]
        a[i] = [v / a[i][i] for v in a[i]]
        for k in range(n):
            if k != i and a[k][i] != 0:
                a[k] = [vk - a[k][i] * vi for vk, vi in zip(a[k], a[i])]
    return [row[-1] for row in a]
