"""Product formulas, qDRIFT, Trotter error bounds and multi-product formulas, checked against expm."""
from __future__ import annotations

import numpy as np
import pytest

scipy_linalg = pytest.importorskip("scipy.linalg")

from quantum_circuit_simulator import dynamics, hamiltonians as hm
from quantum_circuit_simulator.interop import Program
from quantum_circuit_simulator.statevector import statevector

TFIM = hm.transverse_field_ising(hm.chain(4), j=1.0, h=0.7)
HEISENBERG = hm.heisenberg(hm.chain(4), j=(1.0, 0.8, 0.6), h=0.3)


def unitary(program):
    """The circuit's unitary, column by column from each basis state (wire 0 least significant)."""
    n = program.n_qubits
    columns = []
    for x in range(2 ** n):
        prep = [("x", [q], []) for q in range(n) if x >> q & 1]
        columns.append(statevector(Program(n, prep + program.instructions)))
    return np.array(columns).T


def exact(h, t):
    return scipy_linalg.expm(-1j * t * h.matrix())


def error(h, t, steps, order):
    return np.linalg.norm(unitary(hm.trotter(h, t, steps, order)) - exact(h, t), 2)


def slope(steps, errors):
    return np.polyfit(np.log(steps), np.log(errors), 1)[0]


@pytest.mark.parametrize("h", [TFIM, hm.heisenberg(hm.chain(3), j=(1.0, 0.8, 0.6), h=0.3)],
                         ids=["tfim", "heisenberg"])
def test_suzuki_error_scales_as_steps_to_minus_order(h):
    order2 = [np.linalg.norm(unitary(dynamics.suzuki(h, 1.0, r, 2)) - exact(h, 1.0), 2) for r in (4, 8, 16)]
    order4 = [np.linalg.norm(unitary(dynamics.suzuki(h, 1.0, r, 4)) - exact(h, 1.0), 2) for r in (2, 4, 8)]
    assert slope((4, 8, 16), order2) == pytest.approx(-2, abs=0.15)
    assert slope((2, 4, 8), order4) == pytest.approx(-4, abs=0.3)
    assert order4[-1] < order2[-1]


def test_suzuki_time_is_a_shared_parameter():
    numeric = HEISENBERG.expectation(dynamics.suzuki(TFIM, 0.6, 2, 4, initial="plus"))["value"]
    shared = HEISENBERG.expectation(dynamics.suzuki(TFIM, "t", 2, 4, initial="plus"), values={"t": 0.6})["value"]
    assert shared == pytest.approx(numeric, abs=1e-10)
    with pytest.raises(ValueError):
        dynamics.suzuki(TFIM, 1.0, 1, 3)


def test_qdrift_averaged_error_decreases_with_samples():
    t, target = 0.5, exact(TFIM, 0.5)[:, 0]
    means = []
    for samples in (20, 80, 320):
        infidelities = [1 - abs(np.vdot(target, statevector(dynamics.qdrift(TFIM, t, samples, seed=s)))) ** 2
                        for s in range(24)]
        means.append(np.mean(infidelities))
    assert means[0] > means[1] > means[2]
    assert slope((20, 80, 320), means) == pytest.approx(-1, abs=0.3)   # infidelity ~ lambda^2 t^2 / N
    shared = dynamics.qdrift(TFIM, "t", 10, seed=3)
    assert shared.parameters == ["t"]


def test_first_order_bound_matches_hand_calculation():
    h = hm.Hamiltonian(1, [(0.3, {0: "X"}), (-0.5, {0: "Z"})])
    # [aX, bZ] = 2ab XZ, norm 2|ab| = 0.3; bound = t^2 / (2r) * 0.3.
    assert dynamics.trotter_error_bound(h, 2.0, steps=4, order=1) == pytest.approx(4 / 8 * 0.3)
    commuting = hm.Hamiltonian(2, [(1.0, {0: "Z"}), (2.0, {0: "Z", 1: "Z"})])
    assert dynamics.trotter_error_bound(commuting, 1.0, 1, 1) == 0.0
    assert dynamics.trotter_error_bound(commuting, 1.0, 1, 2) == 0.0


@pytest.mark.parametrize("h", [TFIM, HEISENBERG, hm.heisenberg(hm.chain(3, periodic=True), j=(1.0, 0.5, -0.7))],
                         ids=["tfim", "heisenberg", "xyz-ring"])
@pytest.mark.parametrize("order", [1, 2])
def test_commutator_bound_holds(h, order):
    for t, steps in ((0.5, 1), (1.0, 3), (2.0, 8)):
        bound = dynamics.trotter_error_bound(h, t, steps, order)
        assert error(h, t, steps, order) <= bound
    # Second order is the tighter bound at many steps.
    assert dynamics.trotter_error_bound(h, 1.0, 20, 2) < dynamics.trotter_error_bound(h, 1.0, 20, 1)


def test_mpf_coefficients():
    assert dynamics.mpf_coefficients((1, 2)) == pytest.approx([-1 / 3, 4 / 3])
    for ks in ((2, 3, 4), (3, 5, 7, 9), (2, 4)):
        assert sum(dynamics.mpf_coefficients(ks)) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        dynamics.mpf_coefficients((2, 2))


def test_mpf_beats_its_best_component():
    h = hm.transverse_field_ising(hm.chain(6), j=1.0, h=0.8)
    observable = hm.Hamiltonian(6, [(1 / 6, {q: "Z"}) for q in range(6)])   # mean magnetisation
    t = 1.0
    psi = exact(h, t)[:, 0]
    truth = float(np.real(np.vdot(psi, observable.matrix() @ psi)))
    result = dynamics.mpf(h, observable, t, ks=(2, 3, 4))
    best = min(abs(row["value"] - truth) for row in result["components"])
    assert abs(result["value"] - truth) < best / 5
    assert sum(result["coefficients"]) == pytest.approx(1.0)
