"""Fermion-to-qubit mappings, molecular Hamiltonians from PySCF, Z2 tapering and the UCCSD ansatz."""
import numpy as np
import pytest

from quantum_circuit_simulator import chemistry as ch


def _spectrum(hamiltonian):
    return np.linalg.eigvalsh(hamiltonian.matrix())


def test_mappings_agree_on_number_and_hopping_operators():
    number = {((m, 1), (m, 0)): 1.0 for m in range(5)}
    hopping = {((0, 1), (3, 0)): -0.7, ((3, 1), (0, 0)): -0.7, ((1, 1), (4, 0)): 0.3j, ((4, 1), (1, 0)): -0.3j,
               ((2, 1), (2, 0), (4, 1), (4, 0)): 1.1}
    for op in (number, hopping):
        reference = _spectrum(ch.to_qubits(op, 5, "jordan-wigner"))
        for mapping in ("bravyi-kitaev", "parity"):
            assert np.allclose(_spectrum(ch.to_qubits(op, 5, mapping)), reference, atol=1e-12)
    # The number operator counts electrons in every encoding: n = sum (1 - Z_j) / 2 under Jordan-Wigner.
    assert ch.to_qubits(number, 5).offset == pytest.approx(2.5)
    # Seeley-Richard-Love on 4 modes: a+_2 + a_2 = X2 X3 Z1 (update set {3}, parity set {1}).
    majorana = ch.to_qubits({((2, 1),): 1.0, ((2, 0),): 1.0}, 4, "bravyi-kitaev")
    assert majorana.terms == [(1.0, {1: "Z", 2: "X", 3: "X"})]


def test_normal_ordering_applies_the_anticommutation_relations():
    assert ch.normal_ordered({((0, 0), (0, 1)): 1.0}) == {(): 1.0, ((0, 1), (0, 0)): -1.0}
    assert ch.normal_ordered({((1, 1), (1, 1)): 1.0}) == {}
    assert ch.normal_ordered({((0, 1), (2, 1)): 1.0}) == {((2, 1), (0, 1)): -1.0}


def test_h2_matches_fci_in_every_mapping_with_identical_spectra():
    pytest.importorskip("pyscf")
    spectra = []
    for mapping in ch.MAPPINGS:
        h2 = ch.molecule("H2", mapping=mapping)
        assert h2.n_qubits == 4 and h2.n_electrons == 2 and h2.hf_occupation == (1, 1, 0, 0)
        assert h2.exact_energy() == pytest.approx(h2.energies["fci"], abs=1e-8)
        spectra.append(_spectrum(h2.hamiltonian))
        # The Hartree-Fock Program, evaluated by the package's own solver, gives the PySCF HF energy.
        assert h2.hamiltonian.expectation(h2.hf_program())["value"] == pytest.approx(h2.energies["hf"], abs=1e-8)
    assert h2.provenance["basis"] == "sto-3g" and h2.provenance["pyscf"]
    assert np.allclose(spectra[0], spectra[1], atol=1e-10) and np.allclose(spectra[0], spectra[2], atol=1e-10)


def test_lih_active_space_reproduces_casci():
    pytest.importorskip("pyscf")
    lih = ch.molecule("LiH", active=(2, 3), mapping="bravyi-kitaev")
    assert lih.n_qubits == 6 and lih.provenance["frozen_core_orbitals"] == 1
    assert lih.exact_energy() == pytest.approx(lih.energies["casci"], abs=1e-8)
    assert lih.hamiltonian.expectation(lih.hf_program())["value"] == pytest.approx(lih.energies["hf"], abs=1e-8)


def test_tapering_keeps_the_ground_energy_of_the_hartree_fock_sector():
    pytest.importorskip("pyscf")
    for mapping in ch.MAPPINGS:
        h2 = ch.molecule("H2", mapping=mapping, taper=True)
        # Particle-number parity of each spin and one point-group symmetry: 4 qubits -> 1.
        assert h2.n_qubits == 1 and len(h2.tapering.removed) == 3
        assert h2.exact_energy() == pytest.approx(h2.energies["fci"], abs=1e-8)
        assert h2.hamiltonian.expectation(h2.hf_program())["value"] == pytest.approx(h2.energies["hf"], abs=1e-8)
    lih = ch.molecule("LiH", active=(2, 3), mapping="parity", taper=True)
    assert lih.n_qubits < 6
    assert lih.exact_energy() == pytest.approx(lih.energies["casci"], abs=1e-8)


def test_uccsd_double_excitation_reaches_fci_on_h2():
    pytest.importorskip("pyscf")
    from scipy.optimize import minimize_scalar
    h2 = ch.molecule("H2")
    ansatz = ch.uccsd(h2.n_qubits, h2.n_electrons)
    assert ansatz.parameters == ["s_0_2", "s_1_3", "d_0_1_2_3"]

    def energy(theta):
        return h2.hamiltonian.expectation(ansatz, values={"s_0_2": 0.0, "s_1_3": 0.0, "d_0_1_2_3": theta})["value"]

    assert energy(0.0) == pytest.approx(h2.energies["hf"], abs=1e-8)
    best = minimize_scalar(energy, bounds=(-1.0, 1.0), method="bounded", options={"xatol": 1e-8})
    assert best.fun == pytest.approx(h2.energies["fci"], abs=1e-6)
