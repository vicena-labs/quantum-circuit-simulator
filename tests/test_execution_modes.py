"""Cross-backend integration, representation fallback and value-only contracts."""
import math
import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, ConeOverflow


@pytest.mark.parametrize('backend', ['native', 'python', 'graph', 'dense', 'auto'])
def test_rebinding_value_only_and_duplicate_observables(backend):
    c = Circuit(4, 2).gate('RY', [3], .2, 0).gate('RX', [1], -.3, 1)
    c.fsim([3, 1], .19, .31, 0).gate('ECR', [1, 0])
    c.depolarize([3, 1], .02).relax(3, .12, 1.3, 1.6).readout(3, .03, .01)
    obs = [(1e16, {3: 'Z'}), (1., {3: 'Z', 0: 'I'}), (-1e16, {3: 'Z'}),
           (.25, {1: 'X'}), (-.25, {1: 'X'}), (.2, {})]
    p = c.prepare(obs, backend=backend)
    oracle = c.prepare([(1., {3: 'Z'}), (.2, {})], backend='dense')
    vectors = [[0., 0.], [.25, -.3], [-.2, .3]]
    values = p.evaluate_values(vectors)
    for vector, value in zip(vectors, values):
        result = p.evaluate(vector)
        expected = oracle.evaluate(vector)
        assert result.value == pytest.approx(expected.value, abs=3e-12)
        assert result.gradient == pytest.approx(expected.gradient, abs=3e-11)
        assert value == pytest.approx(result.value, abs=3e-12)
        assert p.evaluate(vector, gradients=False).gradient == ()
    assert p.diagnostics['input_observable_terms'] == 6
    assert p.diagnostics['unique_observable_terms'] == 2


@pytest.mark.parametrize('backend', ['native', 'python', 'graph', 'dense'])
def test_zero_observable_and_zero_angle_derivative(backend):
    c = Circuit(1, 1, max_terms=1).gate('RY', [0], 0., 0)
    zero = c.prepare([(1., {0: 'X'}), (-1., {0: 'X'})], backend=backend)
    assert zero.evaluate([.7]).value == 0.
    assert zero.evaluate([.7]).gradient == (0.,)
    plan = c.prepare([(1., {0: 'X'})], backend=backend)
    assert plan.evaluate().value == pytest.approx(0., abs=1e-14)
    assert plan.evaluate().gradient == pytest.approx((1.,), abs=1e-14)
    assert plan.evaluate_value([.3]) == pytest.approx(math.sin(.3), abs=1e-14)


def test_dense_local_cone_not_full_register_and_reverse_shared_gradient():
    c = Circuit(256, 1).gate('RY', [255], .3, 0).gate('RY', [255], .2, 0)
    c.relax(255, .1, 1., 1.)
    p = c.prepare([(1., {255: 'Z'})], backend='dense', dense_max_qubits=1)
    result = p.evaluate([.1])
    a = math.exp(-.1)
    assert result.value == pytest.approx(a*math.cos(.7) + 1-a, abs=2e-12)
    assert result.gradient == pytest.approx((-2*a*math.sin(.7),), abs=2e-12)
    assert p.diagnostics['max_cone_qubits'] == 1


def test_explicit_auto_fallback_and_graph_budget_fallback():
    def brickwork(max_terms):
        c = Circuit(4, 1, max_terms=max_terms)
        for layer in range(3):
            for q in range(4): c.gate('RY', [q], .3 + .1*q + .05*layer, 0 if q == 0 else -1)
            for q in range(layer % 2, 3, 2): c.gate('CZ', [q, q + 1])
        return c
    obs = [(1., {1: 'X'})]
    with pytest.raises(ConeOverflow):
        brickwork(2).prepare(obs).evaluate()
    reference = brickwork(1_000).prepare(obs).evaluate()
    automatic = brickwork(2).prepare(obs, backend='auto')
    result = automatic.evaluate()
    assert result.backend == 'dense'
    assert result.value == pytest.approx(reference.value, abs=2e-12)
    assert result.gradient == pytest.approx(reference.gradient, abs=2e-12)
    assert automatic.diagnostics['fallback_reason']
    obs = [(1., {0: 'X'})]
    # Raising the sparse budget permits fallback from an intentionally tiny graph.
    c = Circuit(1, 1).gate('RY', [0], .3, 0)
    graph = c.prepare(obs, backend='graph', graph_max_nodes=1, graph_max_edges=1)
    assert graph.evaluate().backend == 'native'
    assert graph.diagnostics['requested_backend'] == 'graph'
    assert graph.diagnostics['fallback_reason']


def test_dense_budgets_fail_before_large_allocation():
    c = Circuit(3).gate('H', [0]).gate('CZ', [0, 1]).gate('CZ', [1, 2])
    with pytest.raises(ConeOverflow, match='qubits'):
        c.prepare([(1., {2: 'Z'})], backend='dense', dense_max_qubits=2)
    with pytest.raises(ConeOverflow, match='bytes'):
        c.prepare([(1., {2: 'Z'})], backend='dense', dense_max_bytes=1024)


def test_auto_handles_preparation_readout_overflow():
    c = Circuit(3, max_terms=1)
    for wire in range(3):
        c.gate('RY', [wire], .2).readout(wire, .02, .01)
    obs = [(1., {q: 'Z' for q in range(3)})]
    with pytest.raises(ConeOverflow):
        c.prepare(obs)
    plan = c.prepare(obs, backend='auto')
    assert plan.backend == 'dense'
    assert plan.evaluate_value() == pytest.approx((-.01+.97*math.cos(.2))**3, abs=2e-12)
    assert plan.diagnostics['fallback_reason']


@pytest.mark.parametrize('bad', [0, 1, None, 'false'])
def test_gradient_mode_is_strict_boolean(bad):
    plan = Circuit(1).prepare([(1., {0: 'Z'})])
    with pytest.raises(ValueError, match='boolean'):
        plan.evaluate(gradients=bad)


def test_dense_shared_stage_cones_and_gradient_finite_difference():
    c = Circuit(3, 2).gate('RY', [0], .4, 0).gate('RX', [2], -.2, 1)
    c.fsim([2, 0], .2, .6, 1).gate('CZ', [0, 2])
    c.depolarize([0, 2], .03).relax(0, .2, 1., 1.5)
    p = c.prepare([(.7, {0: 'Z'}), (.3, {0: 'X'})], backend='dense')
    shifts = [.1, -.15]
    result = p.evaluate(shifts)
    for i in range(2):
        low, high = list(shifts), list(shifts)
        low[i] -= 1e-5
        high[i] += 1e-5
        fd = (p.evaluate_value(high)-p.evaluate_value(low))/2e-5
        assert result.gradient[i] == pytest.approx(fd, abs=2e-9)
    assert p.diagnostics['density_groups'] == 1


@pytest.mark.parametrize('backend', ['native', 'python', 'graph', 'dense'])
def test_idle_readout_contracts_before_exponential_initialization(backend):
    c = Circuit(32, max_terms=1)
    for wire in range(32):
        c.readout(wire, .02, .03)
    p = c.prepare([(1., {q: 'Z' for q in range(32)})], backend=backend)
    assert p.evaluate_value() == pytest.approx(.96**32, abs=2e-14)


@pytest.mark.parametrize('name', ['RX', 'RY', 'RZ'])
def test_closed_form_transfers_against_matrix_definition(name):
    from quantum_circuit_simulator._core._reference import cached_gate_ptm, rotation, transfer
    for angle in (0., math.pi/2, math.pi, -.49, 1e3):
        u, du = rotation(name[1], angle)
        expected = transfer(u, du)
        actual = cached_gate_ptm(name, angle, True)
        for a, b in zip(actual, expected):
            assert np.allclose(a, b, rtol=0., atol=3e-15)
