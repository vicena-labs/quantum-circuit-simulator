"""Truncated mode, joint cone groups and threaded batches."""
import math

import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, Truncation


def brickwork(n, layers, nparams, noise, seed=0, angle=1.):
    rng = np.random.default_rng(seed)
    c = Circuit(n, nparams, max_terms=200_000)
    for layer in range(layers):
        for q in range(n):
            c.gate('RY', [q], theta=angle * float(rng.uniform(-1, 1)), param=(2 * layer) % nparams)
            c.gate('RZ', [q], theta=angle * float(rng.uniform(-1, 1)), param=(2 * layer + 1) % nparams)
        for q in range(layer % 2, n - 1, 2):
            c.gate('CZ', [q, q + 1])
            if noise:
                c.depolarize([q, q + 1], noise)
    return c


OBSERVABLES = [(1., {0: 'Z'}), (-.5, {2: 'X', 3: 'X'}), (.25, {4: 'Y'}), (.7, {5: 'Z', 6: 'Z'})]


@pytest.mark.parametrize('backend', ['native', 'graph'])
@pytest.mark.parametrize('noise', [0., .01])
@pytest.mark.parametrize('truncation', [Truncation(1e-2), Truncation(1e-3), Truncation(max_weight=2),
                                        Truncation(1e-3, 3)])
def test_truncation_error_is_within_reported_bound(noise, truncation, backend):
    c = brickwork(8, 5, 4, noise)
    shifts = [.1, -.2, .3, .05]
    exact = c.prepare(OBSERVABLES).evaluate(shifts)
    for gradients in (True, False):
        approx = c.prepare(OBSERVABLES, backend=backend, truncation=truncation).evaluate(
            shifts, gradients=gradients)
        assert approx.backend == backend
        assert approx.error_bound is not None
        assert abs(approx.value - exact.value) <= approx.error_bound * (1 + 1e-12) + 1e-14
        if gradients:
            for got, want, bound in zip(approx.gradient, exact.gradient, approx.gradient_error_bound):
                assert abs(got - want) <= bound * (1 + 1e-12) + 1e-14
        else:
            assert approx.gradient_error_bound is None


@pytest.mark.parametrize('backend', ['native', 'graph'])
def test_inactive_truncation_is_bitwise_exact(backend):
    c = brickwork(8, 4, 4, .01)
    exact = c.prepare(OBSERVABLES, backend=backend).evaluate([.1, .2, .3, .4])
    same = c.prepare(OBSERVABLES, backend=backend, truncation=Truncation()).evaluate([.1, .2, .3, .4])
    assert (same.value, same.gradient) == (exact.value, exact.gradient)
    assert same.error_bound == 0. and same.gradient_error_bound == (0.,) * 4
    assert exact.error_bound is None and 'error_bound' not in exact.as_dict()


@pytest.mark.parametrize('backend', ['native', 'graph'])
def test_weight_limit_drops_the_entangled_term_and_bounds_it(backend):
    # Heisenberg: the last CZ maps X0 to X0 Z1, weight 2 while wires 0 and 1
    # both stay open (each has an earlier CZ); were CZ(0, 1) either wire's
    # first two-wire gate, early contraction would leave weight 1.
    c = Circuit(3, 1).gate('RY', [0], .2).gate('RY', [1], .6).gate('RY', [2], .5)
    c.gate('CZ', [0, 2]).gate('CZ', [1, 2]).gate('RY', [0], theta=.3, param=0).gate('CZ', [0, 1])
    exact = c.prepare([(1., {0: 'X'})]).evaluate([.2])
    assert exact.value != 0.
    approx = c.prepare([(1., {0: 'X'})], backend=backend, truncation=Truncation(max_weight=1)).evaluate([.2])
    assert approx.value == 0. and approx.gradient == (0.,)
    # Dropped coefficient 1 with no tangent, after one unit-generator rotation.
    assert approx.error_bound == pytest.approx(1.)
    assert approx.gradient_error_bound == pytest.approx((1.,))


def test_truncation_threshold_is_relative_to_the_observable():
    c = brickwork(6, 4, 2, .01)
    obs = [(1., {1: 'Z'}), (.5, {3: 'X'})]
    base = c.prepare(obs, truncation=Truncation(1e-3)).evaluate([.1, .2])
    for scale in (1e-20, 1e15):
        scaled = c.prepare([(scale * w, p) for w, p in obs], truncation=Truncation(1e-3)).evaluate([.1, .2])
        assert scaled.value / scale == pytest.approx(base.value, rel=1e-12)
        assert scaled.error_bound / scale == pytest.approx(base.error_bound, rel=1e-12)


@pytest.mark.parametrize('backend', ['python', 'dense'])
def test_truncation_is_refused_by_exact_only_backends(backend):
    with pytest.raises(ValueError, match='truncation'):
        Circuit(2).gate('H', [0]).prepare([(1., {0: 'X'})], backend=backend, truncation=Truncation(1e-3))


def test_joint_group_over_the_cap_splits_into_single_cones():
    # X0 and X3 on a two-layer brickwork need 4 terms together and at most 3
    # apart, so a cap of 3 splits the joint group into single cones.
    def brickwork(max_terms):
        c = Circuit(4, 2, max_terms=max_terms)
        for layer in range(2):
            for q in range(4):
                c.gate('RY', [q], .3 + .1 * q + .05 * layer, 0 if q == 0 else 1 if q == 3 else -1)
            for q in range(layer % 2, 3, 2):
                c.gate('CZ', [q, q + 1])
        return c
    obs = [(1., {0: 'X'}), (1., {3: 'X'})]
    reference = brickwork(1_000).prepare(obs).evaluate([0., 0.])
    plan = brickwork(3).prepare(obs)
    result = plan.evaluate([0., 0.])
    assert result.value == pytest.approx(reference.value, abs=1e-14)
    assert result.gradient == pytest.approx(reference.gradient, abs=1e-14)
    assert plan.diagnostics['cone_groups'] == 2


@pytest.mark.parametrize('backend', ['native', 'graph'])
@pytest.mark.parametrize('truncation', [None, Truncation(1e-4)])
def test_thread_count_does_not_change_results(truncation, backend):
    c = Circuit(600, 2)
    for q in range(600):
        c.gate('RY', [q], theta=.01 * q, param=q % 2)
    for q in range(0, 599, 2):
        c.gate('CZ', [q, q + 1]).depolarize([q, q + 1], .01)
    for q in range(600):
        c.gate('RX', [q], theta=-.02 * q, param=(q + 1) % 2)
    obs = [(1. / 600, {q: 'Z'}) for q in range(600)] + [(.1, {q: 'X', q + 1: 'X'}) for q in range(0, 599, 7)]
    results = [c.prepare(obs, backend=backend, truncation=truncation, threads=t).evaluate([.1, -.1])
               for t in (1, 4)]
    assert results[0] == results[1]
    assert results[0].backend == 'wide-' + backend


@pytest.mark.parametrize('angle', [0., 1.])
def test_truncated_graph_bound_holds_far_from_its_preparation_angles(angle):
    # The graph chooses discarded terms at the prepared angles (all zero, or
    # random); the bound is recomputed at each binding and must still hold.
    c = brickwork(8, 5, 4, .01, angle=angle)
    plan = c.prepare(OBSERVABLES, backend='graph', truncation=Truncation(1e-3))
    exact = c.prepare(OBSERVABLES)
    for shifts in ([0.] * 4, [.1, -.2, .3, .05], [1.3, -2.1, .7, 2.9], [3., 3., -3., 3.]):
        want, got = exact.evaluate(shifts), plan.evaluate(shifts)
        assert abs(got.value - want.value) <= got.error_bound * (1 + 1e-12) + 1e-14
        for g, w, b in zip(got.gradient, want.gradient, got.gradient_error_bound):
            assert abs(g - w) <= b * (1 + 1e-12) + 1e-14


FAR = [1.3, -2.1, .7, 2.9]


def test_truncated_graph_reprepares_when_its_bound_grows():
    # Prepared at zero angles, the graph discards terms that matter at FAR.
    c = brickwork(8, 5, 4, 0., angle=0.)
    exact = c.prepare(OBSERVABLES).evaluate(FAR)
    fixed = c.prepare(OBSERVABLES, backend='graph', truncation=Truncation(1e-3, reprepare_factor=None))
    auto = c.prepare(OBSERVABLES, backend='graph', truncation=Truncation(1e-3))
    stale, fresh = fixed.evaluate(FAR), auto.evaluate(FAR)
    assert fixed.diagnostics['coefficient_pruning'] and 'graph_reprepared' not in fixed.diagnostics
    assert auto.diagnostics['graph_reprepared'] == 1
    assert fresh.error_bound < stale.error_bound
    assert abs(fresh.value - exact.value) <= fresh.error_bound * (1 + 1e-12) + 1e-14
    assert abs(fresh.value - exact.value) < abs(stale.value - exact.value)
    # Back at the angles it was rebuilt for, the graph is not rebuilt again.
    auto.evaluate(FAR)
    assert auto.diagnostics['graph_reprepared'] == 1


def test_weight_only_truncation_never_reprepares():
    c = brickwork(8, 5, 4, 0., angle=0.)
    plan = c.prepare(OBSERVABLES, backend='graph', truncation=Truncation(max_weight=2))
    plan.evaluate(FAR)
    assert 'graph_reprepared' not in plan.diagnostics


def test_reprepare_over_budget_keeps_the_graph_and_stops_trying():
    c = brickwork(8, 5, 4, 0., angle=0.)
    plan = c.prepare(OBSERVABLES, backend='graph', truncation=Truncation(1e-3))
    plan._plan.max_edges = 1  # the next compilation cannot fit
    exact = c.prepare(OBSERVABLES).evaluate(FAR)
    result = plan.evaluate(FAR)
    assert result.backend == 'graph'
    assert abs(result.value - exact.value) <= result.error_bound * (1 + 1e-12) + 1e-14
    assert 'max_edges' in plan.diagnostics['reprepare_failed']
    assert plan.diagnostics['graph_reprepared'] == 0


@pytest.mark.parametrize('factor', [1., .5, float('nan'), float('inf')])
def test_invalid_reprepare_factor(factor):
    with pytest.raises(ValueError):
        Truncation(1e-3, reprepare_factor=factor)
