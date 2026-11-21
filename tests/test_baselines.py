"""QAOA fixed angles and transfer, and classical baselines for cost Hamiltonians."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantum_circuit_simulator import algorithms as al, baselines, hamiltonians as hm

# The Petersen graph: 3-regular, girth 5, so triangle-free and the p = 1 closed form is exact.
PETERSEN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
            (5, 7), (7, 9), (6, 9), (6, 8), (5, 8)]
# The Heawood graph: 3-regular, girth 6, so p = 2 sees a tree around every edge.
HEAWOOD = sorted({tuple(sorted((i, (i + s) % 14))) for i in range(14)
                  for s in ((1, -1, 5) if i % 2 == 0 else (1, -1, -5))})
P1_FRACTION = 0.5 + 0.5 / math.sqrt(3) * (2 / 3)   # 0.69245, the p = 1 optimum at d = 3


def cut_fraction(edges, angles):
    cost = hm.maxcut(edges)
    program = hm.qaoa(cost, len(angles) // 2)
    return -cost.expectation(program, values=dict(zip(program.parameters, angles)))["value"] / len(edges)


def test_p1_matches_the_closed_form_per_edge():
    # Literature convention: each edge cuts with 1/2 + 1/2 sin(4 beta) sin(gamma) cos(gamma)^2; the package's gamma is -gamma.
    for gamma, beta in ((0.3, 0.2), (-0.7, 0.5), (1.1, -0.35)):
        closed = 0.5 + 0.5 * math.sin(4 * beta) * math.sin(gamma) * math.cos(gamma) ** 2
        assert cut_fraction(PETERSEN, [-gamma, beta]) == pytest.approx(closed, abs=1e-10)
    angles = baselines.fixed_angles(hm.maxcut(PETERSEN), 1)
    assert angles == pytest.approx([-math.atan(1 / math.sqrt(2)), math.pi / 8])
    assert cut_fraction(PETERSEN, angles) == pytest.approx(P1_FRACTION, abs=1e-10)
    assert round(P1_FRACTION, 4) == 0.6925


def test_p2_table_angles_reach_the_tree_value_on_girth_six():
    assert len(HEAWOOD) == 21
    assert cut_fraction(HEAWOOD, baselines.fixed_angles(hm.maxcut(HEAWOOD), 2)) == pytest.approx(0.7559, abs=1e-4)


def test_fixed_angle_init_starts_at_the_analytic_value():
    cost = hm.maxcut(PETERSEN)
    result = al.qaoa(cost, depth=1, init="fixed-angle", steps=5, shots=0)
    assert result.trace[0]["value"] == pytest.approx(-P1_FRACTION * len(PETERSEN), abs=1e-6)
    # Weighted uniformly by 2: the angles rescale so the start is unchanged up to the weight.
    doubled = al.qaoa(hm.maxcut(PETERSEN, [2.0] * 15), depth=1, init="fixed-angle", steps=1, shots=0)
    assert doubled.trace[0]["value"] == pytest.approx(-2 * P1_FRACTION * len(PETERSEN), abs=1e-6)
    with pytest.raises(ValueError):
        baselines.fixed_angles(hm.maxcut([(0, 1), (1, 2)]), 1)   # not regular


def test_parameter_transfer_interpolates_to_deeper_circuits():
    assert baselines.interpolate([0.3, 0.1]) == pytest.approx([0.3, 0.1, 0.3, 0.1])
    assert baselines.interpolate([1.0, 0.5, 2.0, 0.25]) == pytest.approx([1.0, 0.5, 1.5, 0.375, 2.0, 0.25])
    cost = hm.maxcut(PETERSEN)
    shallow = al.qaoa(cost, depth=1, init="fixed-angle", steps=20, shots=0)
    deep = al.qaoa(cost, depth=2, init=shallow, steps=20, shots=0)
    assert deep.trace[0]["parameters"] == pytest.approx(baselines.interpolate(shallow.parameters))
    assert deep.value <= shallow.value + 1e-9
    same = al.qaoa(cost, depth=1, init=[0.2, 0.3], steps=1, shots=0)
    assert same.trace[0]["parameters"] == pytest.approx([0.2, 0.3])


@pytest.mark.parametrize("cost", [
    hm.maxcut(PETERSEN),
    hm.maxcut(hm.random_regular(12, 3, seed=4)),
    hm.Hamiltonian(10, [(float(c), {a: "Z", b: "Z"}) for (a, b), c in
                        zip(hm.random_regular(10, 4, seed=1), np.random.default_rng(2).normal(size=20))]
                   + [(float(c), {q: "Z"}) for q, c in enumerate(np.random.default_rng(3).normal(size=10))]),
], ids=["petersen", "regular-12", "spin-glass-10"])
def test_brute_force_equals_annealing(cost):
    exact = baselines.brute_force(cost)
    annealed = baselines.simulated_annealing(cost, sweeps=300, seed=0)
    assert annealed["value"] == pytest.approx(exact["value"])
    diagonal = al.cost_diagonal(cost)
    assert diagonal[int(annealed["bitstring"], 2)] == pytest.approx(exact["value"])
    assert diagonal[int(exact["bitstring"], 2)] == pytest.approx(exact["value"])


def test_petersen_optimum_and_ratio():
    best = baselines.brute_force(hm.maxcut(PETERSEN))
    assert best["value"] == -12   # the Petersen graph's maximum cut is 12 of 15 edges
    assert baselines.approximation_ratio(-P1_FRACTION * 15, best["value"]) == pytest.approx(P1_FRACTION * 15 / 12)


def test_goemans_williamson():
    pytest.importorskip("cvxpy")
    cost = hm.maxcut(PETERSEN)
    out = baselines.goemans_williamson(cost, seed=0)
    assert out["sdp_bound"] <= -12 + 1e-6
    assert out["value"] <= -0.878 * 12
