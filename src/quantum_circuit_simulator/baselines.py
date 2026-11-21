"""QAOA research practice: fixed-angle and transferred initial angles, and classical baselines.

    from quantum_circuit_simulator import algorithms as al, baselines, hamiltonians as hm
    cost = hm.maxcut(hm.random_regular(12, 3))
    result = al.qaoa(cost, depth=1, init="fixed-angle")          # starts at the analytic p = 1 optimum
    deeper = al.qaoa(cost, depth=2, init=result)                 # parameter transfer (INTERP)
    best = baselines.brute_force(cost)
    baselines.approximation_ratio(result.value, best["value"])

Angle convention: the package's QAOA (`hamiltonians.qaoa`) applies
exp(-i gamma H_cost) then exp(-i beta sum X), with parameters ordered gamma_1,
beta_1, ..., gamma_p, beta_p. For Max-Cut, `hamiltonians.maxcut` is
H = sum_e (Z Z - 1) / 2 = -C, the negative of the usual cut operator
C = sum_e (1 - Z Z) / 2, so the package's gamma is minus the gamma of the
literature (Farhi et al., Wang et al., Wurtz and Lykov), which applies
exp(-i gamma C). Every angle here is returned in the package's convention.

Cost Hamiltonians are diagonal Pauli sums; energies are <H>, which the
baselines minimise (for Max-Cut, minus the cut weight).
"""
from __future__ import annotations

import math

import numpy as np

from .algorithms import cost_diagonal
from .hamiltonians import load

# Fixed angles for Max-Cut on 3-regular graphs in the literature convention (gamma_1..p, beta_1..p),
# from the tables of Wurtz and Lykov, arXiv:2107.00677. Their expected cut fractions, exact on graphs
# of girth above 2p + 1: 0.6925 (p = 1), 0.7559 (p = 2), 0.7924 (p = 3).
WURTZ_LYKOV_3_REGULAR = {
    2: ((0.4882, 0.8977), (0.5550, 0.2921)),
    3: ((0.4222, 0.7983, 0.9370), (0.6089, 0.4594, 0.2354)),
}


# ---------------------------------------------------------------- initial angles
def initial_angles(cost, depth, init):
    """Starting angles for `algorithms.qaoa(..., init=...)` in the package's parameter order.

    init is "fixed-angle" (`fixed_angles`), or a previous result (anything
    with `.parameters`) or a sequence of angles gamma_1, beta_1, ...: used as
    they are at the same depth, and extended from a shallower depth by the
    INTERP rule of Zhou, Wang, Choi, Pichler and Lukin, arXiv:1812.01041.
    """
    if isinstance(init, str):
        if init != "fixed-angle":
            raise ValueError("init must be 'fixed-angle', a previous result or a sequence of angles")
        return fixed_angles(cost, depth)
    angles = [float(a) for a in getattr(init, "parameters", init)]
    if len(angles) % 2 or not 0 < len(angles) <= 2 * depth:
        raise ValueError(f"expected an even number of angles, at most {2 * depth}")
    while len(angles) < 2 * depth:
        angles = interpolate(angles)
    return angles


def fixed_angles(cost, depth=1):
    """Fixed QAOA angles for unweighted Max-Cut on a regular graph, in the package's convention.

    p = 1 uses the closed-form optimum for triangle-free d-regular graphs
    (Wang, Hadfield, Jiang and Rieffel, arXiv:1706.02998): the expected cut of
    each edge is 1/2 + 1/2 sin(4 beta) sin(gamma) cos(gamma)^(d-1), maximal at
    beta = pi/8, gamma = arctan(1/sqrt(d-1)). p = 2 and 3 on 3-regular graphs
    use Wurtz and Lykov's table (arXiv:2107.00677). The cost must be uniform
    positive Z Z terms (plus a constant) on a regular graph, as `hamiltonians.maxcut` builds.
    """
    degree, coefficient = _regular_maxcut(load(cost))
    if depth == 1:
        gammas, betas = (math.atan(1 / math.sqrt(degree - 1)),), (math.pi / 8,)
    elif degree == 3 and depth in WURTZ_LYKOV_3_REGULAR:
        gammas, betas = WURTZ_LYKOV_3_REGULAR[depth]
    else:
        raise ValueError("fixed angles exist here for p = 1 on any regular graph and p = 2, 3 on 3-regular graphs")
    # exp(-i gamma_pkg c Z Z) = exp(+i gamma (1/2) Z Z) up to phase, so gamma_pkg = -gamma / (2c).
    return [a for g, b in zip(gammas, betas) for a in (-g / (2 * coefficient), b)]


def _regular_maxcut(cost):
    """(degree, coefficient) of a cost with equal positive Z Z terms on a regular graph."""
    edges = [tuple(p) for c, p in cost.terms if p]
    coefficients = {round(c, 12) for c, p in cost.terms if p}
    if any(len(p) != 2 or set(p.values()) != {"Z"} for _, p in cost.terms if p) or len(coefficients) != 1:
        raise ValueError("fixed angles need an unweighted Max-Cut cost: equal Z Z terms only")
    (coefficient,) = coefficients
    degrees = {q: 0 for q in range(cost.n_qubits)}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    if coefficient <= 0 or len(set(degrees.values())) != 1 or degrees[0] < 2:
        raise ValueError("fixed angles need a regular graph of degree at least 2 and positive Z Z weights")
    return degrees[0], coefficient


def interpolate(angles):
    """Depth p angles to depth p + 1 by INTERP (Zhou et al., arXiv:1812.01041):
    x_i' = (i-1)/p x_(i-1) + (p-i+1)/p x_i for i = 1..p+1, with x_0 = x_(p+1) = 0, for gammas and betas."""
    gammas, betas = angles[0::2], angles[1::2]
    p = len(gammas)

    def grow(x):
        x = [0.0, *x, 0.0]
        return [(i - 1) / p * x[i - 1] + (p - i + 1) / p * x[i] for i in range(1, p + 2)]
    return [a for g, b in zip(grow(gammas), grow(betas)) for a in (g, b)]


# ---------------------------------------------------------------- classical baselines
def brute_force(cost):
    """The exact minimum of a diagonal cost by enumerating every bitstring (practical to about 24 qubits).

    Returns value (minimum energy), bitstring (wire 0 rightmost, as everywhere
    in the package), maximum and degeneracy (bitstrings at the minimum).
    """
    h = load(cost)
    diagonal = cost_diagonal(h)
    index = int(np.argmin(diagonal))
    low = diagonal[index]
    return {"value": float(low), "bitstring": format(index, f"0{h.n_qubits}b"), "maximum": float(diagonal.max()),
            "degeneracy": int(np.count_nonzero(np.abs(diagonal - low) < 1e-9))}


def simulated_annealing(cost, *, sweeps=1000, restarts=16, seed=0, temperatures=None):
    """Seeded Metropolis simulated annealing on an Ising cost (Z and Z Z terms), vectorised over restarts.

    Each sweep proposes a flip of every spin in turn, in all restarts at once.
    temperatures is (hot, cold) for the geometric schedule; by default hot is
    twice the largest local field bound and cold 1% of the smallest
    coefficient. Returns the best value and bitstring found and every restart's best.
    """
    h = load(cost)
    n = h.n_qubits
    offset, fields, couplings = _ising(h)
    scale = np.abs(fields) + np.abs(couplings).sum(axis=1)
    small = min(abs(c) for c, p in h.terms if p)
    hot, cold = temperatures or (2 * scale.max(), 0.01 * small)
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1.0, 1.0], size=(restarts, n))

    def energy(s):
        return offset + s @ fields + 0.5 * np.einsum("ri,ij,rj->r", s, couplings, s)
    best, best_spins = energy(spins), spins.copy()
    for temperature in np.geomspace(hot, cold, sweeps):
        for i in range(n):
            delta = -2 * spins[:, i] * (fields[i] + spins @ couplings[:, i])
            accept = (delta <= 0) | (rng.random(restarts) < np.exp(-np.maximum(delta, 0) / temperature))
            spins[accept, i] *= -1
        current = energy(spins)
        better = current < best
        best[better], best_spins[better] = current[better], spins[better]
    winner = int(np.argmin(best))
    return {"value": float(best[winner]), "bitstring": _bitstring(best_spins[winner]),
            "restarts": [float(v) for v in best], "sweeps": sweeps}


def _ising(h):
    """(offset, fields h_i, symmetric couplings W with zero diagonal) with E(s) = offset + h.s + s.W.s / 2."""
    n = h.n_qubits
    offset, fields, couplings = 0.0, np.zeros(n), np.zeros((n, n))
    for c, p in h.terms:
        if set(p.values()) - {"Z"} or len(p) > 2:
            raise ValueError("this baseline takes Ising costs: Z and Z Z terms only")
        wires = sorted(p)
        if not wires:
            offset += c
        elif len(wires) == 1:
            fields[wires[0]] += c
        else:
            a, b = wires
            couplings[a, b] += c
            couplings[b, a] += c
    return offset, fields, couplings


def _bitstring(spins):
    """Spin +1 is bit 0 (Z eigenvalue +1); wire 0 rightmost."""
    return "".join("0" if s > 0 else "1" for s in spins[::-1])


def goemans_williamson(cost, *, rounds=100, seed=0):
    """The Goemans-Williamson SDP relaxation and random-hyperplane rounding for a Z Z cost (needs cvxpy).

    Minimises sum c_ij X_ij over positive semidefinite X with unit diagonal,
    a lower bound on the minimum energy, then rounds `rounds` seeded random
    hyperplanes and keeps the best. For Max-Cut (c_ij > 0) the expected
    rounded cut is at least 0.878 of the optimum (Goemans and Williamson,
    J. ACM 42, 1115 (1995)). Returns value, bitstring and sdp_bound.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("the Goemans-Williamson baseline needs cvxpy: pip install cvxpy") from exc
    h = load(cost)
    offset, fields, couplings = _ising(h)
    if np.any(fields):
        raise ValueError("Goemans-Williamson takes Z Z terms only (no fields)")
    n = h.n_qubits
    x = cp.Variable((n, n), PSD=True)
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum(cp.multiply(couplings, x))), [cp.diag(x) == 1])
    problem.solve()
    values, vectors = np.linalg.eigh(x.value)
    embedding = vectors * np.sqrt(np.clip(values, 0, None))   # rows are the unit vectors v_i
    spins = np.sign(embedding @ np.random.default_rng(seed).normal(size=(n, rounds)))
    spins[spins == 0] = 1
    energies = offset + 0.5 * np.einsum("ir,ij,jr->r", spins, couplings, spins)
    winner = int(np.argmin(energies))
    return {"value": float(energies[winner]), "bitstring": _bitstring(spins[:, winner]),
            "sdp_bound": float(offset + problem.value)}


def approximation_ratio(value, optimum):
    """value / optimum: the fraction of the optimum reached, for a maximised positive quantity
    (a cut weight) or a minimised negative energy (`hamiltonians.maxcut`'s <H> = -cut).

    `algorithms.qaoa` reports a different, shift-invariant ratio, (E - max) / (min - max).
    """
    if optimum == 0:
        raise ValueError("the optimum is zero; the ratio is undefined")
    return value / optimum
