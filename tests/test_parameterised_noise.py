"""Depolarising rates as shared parameters: the same answers as rebuilding, with gradients."""
import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, Truncation
from quantum_circuit_simulator import mitigation
from quantum_circuit_simulator.statevector import StateVectorPlan

RATES = np.array([0.02, 0.05, 0.0, 0.11])
OBS = [(0.6, {0: "Z"}), (0.4, {1: "Z", 2: "X"}), (-0.3, {3: "Y"}), (0.2, {0: "Z", 3: "Z"})]


def circuit(rates, parameterised, scale=1.0):
    c = Circuit(4, 5 if parameterised else 1)
    for q in range(4):
        c.gate("RX", [q], 0.4 + 0.3 * q, param=0)
        c.pauli_error(q, "XYZX"[q], 2 * rates[0], *((1, 2 * scale) if parameterised else ()))
        c.depolarize([q], rates[0], *((1, scale) if parameterised else ()))
    for layer, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (0, 1), (1, 2))):
        c.gate("RZZ", [a, b], -np.pi / 2)
        k = 1 + layer % 3
        c.depolarize([a, b], rates[k], *((k + 1, scale) if parameterised else ()))
        c.gate("RY", [b], 0.7)
    return c


@pytest.mark.parametrize("backend", ["native", "graph", "dense"])
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_binding_matches_rebuilding_and_gradients_match_differences(backend, scale):
    plan = circuit(RATES, True, scale).prepare(OBS, backend=backend)
    for new in (RATES, np.array([0.3, 0.0, 0.2, 0.01]), np.array([0.4, 15 / 16, 0.5, 0.4])):
        shift = np.r_[0.1, (new - RATES) / scale]
        result = plan.evaluate(shift)
        rebuilt = circuit(new, False).prepare(OBS).evaluate([0.1])
        assert result.value == pytest.approx(rebuilt.value, abs=1e-12)
        for j in range(5):
            e = np.zeros(5)
            e[j] = 1e-6
            f = lambda s: plan.evaluate_value(list(s))
            if j and new[j - 1] == 0.0:  # only one side of zero is a probability
                slope = (-3 * f(shift) + 4 * f(shift + e) - f(shift + 2 * e)) / 2e-6
            else:
                slope = (f(shift + e) - f(shift - e)) / 2e-6
            assert result.gradient[j] == pytest.approx(slope, abs=2e-8)


def test_truncated_gradient_bound_holds():
    exact = circuit(RATES, True).prepare(OBS).evaluate([0.0] * 5)
    for backend in ("native", "graph"):
        plan = circuit(RATES, True).prepare(OBS, backend=backend, truncation=Truncation(min_coefficient=0.02))
        result = plan.evaluate([0.0] * 5)
        assert abs(result.value - exact.value) <= result.error_bound + 1e-15
        assert np.all(np.abs(np.array(result.gradient) - exact.gradient) <= np.array(result.gradient_error_bound) + 1e-12)


def test_density_matrix_agrees_at_a_shift():
    c, shift = circuit(RATES, True), [0.1, 0.03, 0.02, 0.1, -0.05]
    value = StateVectorPlan(c, OBS).evaluate(shift, gradients=False)[0]
    assert value == pytest.approx(c.prepare(OBS).evaluate_value(shift), abs=1e-12)


@pytest.mark.parametrize("backend", ["native", "graph", "dense"])
def test_invalid_probability_is_refused_and_leaves_the_plan_unchanged(backend):
    plan = circuit(RATES, True).prepare(OBS, backend=backend)
    before = plan.evaluate_value([0.0] * 5)
    for bad in (-0.03, 2.0):
        with pytest.raises(ValueError):
            plan.evaluate_value([0.0, bad, 0.0, 0.0, 0.0])
    assert plan.evaluate_value([0.0] * 5) == before


def test_mitigation_refuses_parameterised_noise():
    with pytest.raises(ValueError, match="parameterised noise"):
        mitigation.pna(circuit(RATES, True), OBS)
