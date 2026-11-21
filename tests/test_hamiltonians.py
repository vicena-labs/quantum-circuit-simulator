"""Hamiltonian models and loaders, and the Trotter and QAOA circuits built from them."""
import json
from functools import reduce

import numpy as np
import pytest

from quantum_circuit_simulator import hamiltonians as hm
from quantum_circuit_simulator.statevector import statevector


def test_models_have_the_expected_terms_and_round_trip(tmp_path):
    tfim = hm.transverse_field_ising(hm.grid(3, 3), j=1.0, h=0.5)
    assert tfim.n_qubits == 9 and len(tfim.terms) == 12 + 9
    xxz = hm.heisenberg(hm.chain(6, periodic=True), j=(1.0, 1.0, 0.5), h=0.2)
    assert len(xxz.terms) == 6 * 3 + 6
    cut = hm.maxcut([(0, 1), (1, 2)], weights=[2.0, 1.0])
    assert cut.offset == -1.5
    path = tmp_path / "h.json"
    xxz.to_json(path)
    again = hm.load(path)
    assert again.terms == xxz.terms and json.loads(path.read_text(encoding="utf-8"))["schema"] == "quantum-circuit-simulator-hamiltonian/1"
    assert hm.load("0.5*Z0*Z1 - X2").n_qubits == 3
    assert hm.load(xxz.to_string(), n_qubits=6).terms == xxz.terms
    assert hm.load(cut.to_string(), n_qubits=3).offset == -1.5
    assert len(hm.random_regular(12, 3, seed=1)) == 18
    assert hm.model("heisenberg", "grid", (2, 3)).n_qubits == 6


def _fermion_ops(n_orbitals):
    """Jordan-Wigner annihilation operators built from matrices, wire 0 least significant."""
    lower = np.array([[0, 1], [0, 0]], complex)    # |0><1|: occupied |1> -> empty |0>
    z, eye = np.diag([1.0, -1.0]).astype(complex), np.eye(2, dtype=complex)
    ops = []
    for j in range(n_orbitals):
        factors = [z if q < j else lower if q == j else eye for q in range(n_orbitals)]
        ops.append(reduce(np.kron, list(reversed(factors))))
    return ops


def test_fermi_hubbard_matches_fermion_operators():
    edges, t, u = [(0, 1), (1, 2), (2, 3), (0, 3)], 1.0, 4.0
    model = hm.fermi_hubbard(edges, t=t, u=u)
    n = 4
    c = _fermion_ops(2 * n)
    number = [op.conj().T @ op for op in c]
    want = np.zeros((2 ** (2 * n),) * 2, complex)
    for spin in (0, n):
        for a, b in edges:
            hop = c[a + spin].conj().T @ c[b + spin]
            want += -t * (hop + hop.conj().T)
    for i in range(n):
        want += u * number[i] @ number[i + n]
    assert np.allclose(model.matrix(), want, atol=1e-12)


def test_trotter_converges_to_exact_evolution_and_differentiates_in_time():
    expm = pytest.importorskip("scipy.linalg").expm
    h = hm.heisenberg(hm.chain(4), j=(1.0, 0.8, 0.6), h=0.3)
    exact = expm(-1j * 0.7 * h.matrix())[:, 0]
    second = statevector(hm.trotter(h, 0.7, steps=40, order=2))
    assert abs(abs(np.vdot(exact, second)) - 1) < 1e-6
    first = statevector(hm.trotter(h, 0.7, steps=40, order=1))
    assert abs(abs(np.vdot(exact, first)) - 1) < 1e-3
    observable = hm.Hamiltonian(4, [(1.0, {1: "Z"})])
    program = hm.trotter(h, "t", steps=6, initial=[("x", [0], [])])
    result = observable.expectation(program, values={"t": 0.5}, gradients=True)
    up = observable.expectation(program, values={"t": 0.5 + 1e-6})["value"]
    down = observable.expectation(program, values={"t": 0.5 - 1e-6})["value"]
    assert result["gradient"][0] == pytest.approx((up - down) / 2e-6, abs=1e-6)


def test_ground_state_energy_dense_and_sparse():
    h = hm.transverse_field_ising(hm.chain(8), j=1.0, h=1.0)
    dense = h.ground_state_energy()
    assert dense == pytest.approx(np.linalg.eigvalsh(h.matrix())[0])
    pytest.importorskip("scipy")
    big = hm.transverse_field_ising(hm.chain(14), j=1.0, h=0.5)
    assert big.ground_state_energy() < -13.0


def test_qaoa_program_shares_gamma_across_weighted_edges():
    cost = hm.maxcut([(0, 1), (1, 2), (2, 0)], weights=[1.0, 2.0, 3.0])
    program = hm.qaoa(cost, depth=2)
    assert program.parameters == ["gamma_1", "beta_1", "gamma_2", "beta_2"]
    from quantum_circuit_simulator.interop import convert
    assert convert(program).circuit.n_parameters == 4
    value = cost.expectation(program, values=[0.4, 0.3, 0.2, 0.1])["value"]
    from quantum_circuit_simulator.algorithms import cost_diagonal
    psi = statevector(program, values=[0.4, 0.3, 0.2, 0.1])
    assert value == pytest.approx(float(np.abs(psi) ** 2 @ cost_diagonal(cost)), abs=1e-12)
    assert "OPENQASM 2.0" in program.to_qasm([0.4, 0.3, 0.2, 0.1])
