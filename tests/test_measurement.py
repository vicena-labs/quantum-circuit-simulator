"""Commuting groups, diagonalizing Cliffords, exact shot budgets and classical shadows."""
import math

import numpy as np
import pytest

from quantum_circuit_simulator import hamiltonians as hm, measurement as ms
from quantum_circuit_simulator.api import Circuit
from quantum_circuit_simulator.interop import parse_observable
from quantum_circuit_simulator.statevector import sample


def state(n=4, noise=0.0):
    c = Circuit(n)
    for q in range(n):
        c.gate("RY", [q], theta=0.3 + 0.25 * q)
    for q in range(n - 1):
        c.gate("CZ", [q, q + 1])
        if noise:
            c.depolarize([q, q + 1], noise)
    for q in range(n):
        c.gate("RX", [q], theta=0.6 - 0.15 * q)
    return c


def exact(circuit, spec):
    return circuit.prepare(parse_observable(spec, circuit.n_qubits)).evaluate(gradients=False).value


def test_group_counts_and_pairwise_commutation():
    xyz = hm.load("XX + YY + ZZ")
    assert len(ms.group(xyz, "qwc")) == 3
    assert len(ms.group(xyz, "commuting")) == 1
    h = hm.heisenberg(hm.chain(6), j=(1.0, 0.7, 0.4), h=0.3) + hm.transverse_field_ising(hm.chain(6), 0.5, 0.2)
    for method, commute in (("qwc", ms.qubitwise_commute), ("commuting", ms.commute_fully)):
        groups = ms.group(h, method)
        assert sum(len(g) for g in groups) == len([t for t in h.terms if t[1]])
        assert all(commute(p, q) for g in groups for _, p in g for _, q in g)
    assert len(ms.group(h, "commuting")) <= len(ms.group(h, "qwc"))


def test_clifford_diagonalises_every_group():
    h = hm.load("XX + YY + ZZ") + hm.load("0.5*XXII + 0.5*IIYY - 0.2*ZZZZ + 0.3*XYYX")
    for g in ms.group(h, "commuting"):
        d = ms.diagonalizing_clifford(g, 4)
        assert all(set(m["wires"]) for m in d["measure"])
        assert d["qasm"].startswith("OPENQASM 2.0;")


def test_analytic_group_variance_matches_sampling():
    """Measure XX + YY + ZZ in one Bell-type basis and compare the single-shot variance with the exact one."""
    circuit = state(2)
    (members,) = ms.group(hm.load("XX + 0.5*YY - 0.8*ZZ"), "commuting")
    clifford = ms.diagonalizing_clifford(members, 2)
    shots = 20000
    counts = sample(ms.append_clifford(circuit, clifford["gates"]), shots, seed=5)["counts"]
    values = []
    for bits, count in counts.items():
        shot = sum(m["coefficient"] * m["sign"] * (-1) ** sum(int(bits[-1 - q]) for q in m["wires"])
                   for m in clifford["measure"])
        values += [shot] * count
    values = np.array(values)
    energy = sum(c * exact(circuit, [(1.0, p)]) for c, p in members)
    variance = ms.group_variance(members, lambda w: exact(circuit, [(1.0, dict(w))]) if w else 1.0)
    assert values.var(ddof=1) == pytest.approx(variance, rel=0.1)
    assert abs(values.mean() - energy) < 4 * math.sqrt(variance / shots)


def test_shot_budget_is_optimal_and_beats_qwc():
    circuit = state(2)
    h = hm.load("XX + 0.5*YY - 0.8*ZZ + 0.3*ZI")
    budget = ms.shot_budget(circuit, h, precision=0.01)
    assert budget["groups"] < 3 and budget["qwc_shots"] > budget["shots"]
    variances = [g["variance"] for g in budget["per_group"]]
    assert budget["optimal_shots"] == pytest.approx(sum(math.sqrt(v) for v in variances) ** 2 / 1e-4)
    # The allotted shots reach the requested standard error.
    error = math.sqrt(sum(g["variance"] / g["shots"] for g in budget["per_group"] if g["shots"]))
    assert error <= 0.01 * (1 + 1e-9)


def test_shadows_estimate_pauli_observables():
    observables = ["Z0*Z1", "X2", "Y1*Z3", "X0*X1*X2", "0.5*Z2 - X3"]
    for circuit in (state(4), state(4, noise=0.02)):
        values = [exact(circuit, o) for o in observables]
        small = ms.shadows(circuit, observables, 500, seed=7)["results"]
        large = ms.shadows(circuit, observables, 4000, seed=8)["results"]
        for value, s, l in zip(values, small, large):
            assert abs(l["estimate"] - value) < 4 * l["standard_error"]
            assert abs(s["estimate"] - value) < 4 * s["standard_error"]
            assert l["standard_error"] < 0.5 * s["standard_error"]
    medians = ms.shadows(state(4), observables, 2000, median_of_means=5, seed=9)["results"]
    for value, m in zip([exact(state(4), o) for o in observables], medians):
        assert abs(m["estimate"] - value) < 5 * m["standard_error"]
