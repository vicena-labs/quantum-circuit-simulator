"""Noise inversion (PNA), PEC and readout mitigation against exact noise-free values."""
import math

import pytest

from quantum_circuit_simulator import devices, mitigation as mt
from quantum_circuit_simulator.api import Circuit
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.statevector import sample

QASM = """OPENQASM 2.0; include "qelib1.inc"; qreg q[4];
h q[0]; cx q[0],q[1]; ry(0.4) q[2]; cx q[1],q[2]; rx(0.7) q[3]; cx q[2],q[3]; h q[1];"""
OBSERVABLE = "Z0*Z2 + 0.5*X1 - 0.3*Z3*Z2"


@pytest.fixture(scope="module")
def chip():
    """A tiny circuit on ibm_fez with amplified noise (depolarizing, relaxation and readout), and its ideal value."""
    device = devices.get("ibm_fez")
    noisy = compile_circuit(QASM, device, noise_scale=3.0)
    ideal = compile_circuit(QASM, device, noise_scale=0.0).expectation(OBSERVABLE).value
    return noisy, ideal


def test_pna_cancels_the_known_noise_exactly(chip):
    compiled, ideal = chip
    exact = mt.pna(compiled, OBSERVABLE)
    assert abs(exact["unmitigated"] - ideal) > 1e-2
    assert exact["value"] == pytest.approx(ideal, abs=1e-10)
    # A noise model learned at half strength under-corrects: biased, between noisy and ideal.
    half = mt.pna(compiled, OBSERVABLE, scale=0.5)
    assert min(exact["unmitigated"], ideal) < half["value"] < max(exact["unmitigated"], ideal)
    assert abs(half["value"] - ideal) > 1e-3


def test_inverted_circuit_follows_each_noise_stage_with_its_inverse(chip):
    compiled, _ = chip
    inverted = mt.invert_noise(compiled)
    names = [s.name for s in inverted._impl.stages]
    noise = [i for i, s in enumerate(compiled.circuit._impl.stages) if s.name in ("DEPOLARIZE", "T1_T2")]
    assert sum(n.startswith("INVERSE_") for n in names) == len(noise)
    assert all(names[i + 1] == "INVERSE_" + names[i] for i, n in enumerate(names) if n in ("DEPOLARIZE", "T1_T2"))


def test_pec_overhead_matches_the_depolarizing_closed_form():
    for k, p in ((1, 0.05), (2, 0.03)):
        circuit = Circuit(2).gate("H", [0]).depolarize(list(range(k)), p)
        d = 4 ** k
        f = 1 - d * p / (d - 1)
        gamma = (2 * (d - 1) / f - (d - 2)) / d   # k = 1: (3/f - 1) / 2
        overhead = mt.pec_overhead(circuit)
        assert overhead["gamma"] == pytest.approx(gamma, rel=1e-12)
        assert overhead["shot_multiplier"] == pytest.approx(gamma ** 2, rel=1e-12)


def test_pec_estimate_is_unbiased_and_converges(chip):
    _, ideal = chip
    compiled = compile_circuit(QASM, devices.get("ibm_fez"), noise_scale=10.0)
    few = mt.pec(compiled, OBSERVABLE, 100, seed=3)
    many = mt.pec(compiled, OBSERVABLE, 400, seed=4)
    assert few["twirled_stages"] > 0 and few["gamma"] > 1
    assert abs(few["estimate"] - ideal) < 4 * few["standard_error"]
    assert abs(many["estimate"] - ideal) < 4 * many["standard_error"]
    assert many["standard_error"] < 0.7 * few["standard_error"]
    assert abs(many["estimate"] - ideal) < abs(many["unmitigated"] - ideal)


@pytest.fixture(scope="module")
def readout_only():
    """Two qubits with asymmetric readout error and no other noise."""
    def build(readout):
        c = Circuit(2).gate("RY", [0], theta=0.5).gate("RY", [1], theta=1.1).gate("CZ", [0, 1]).gate("RY", [1], theta=0.3)
        if readout:
            c.readout(0, 0.03, 0.08).readout(1, 0.05, 0.02)
        return c
    ideal = build(False).prepare([(1.0, {0: "Z", 1: "Z"})]).evaluate(gradients=False).value
    return build(True), ideal


def test_readout_correct_recovers_the_ideal_product_exactly(readout_only):
    circuit, ideal = readout_only
    measured = {w: circuit.prepare([(1.0, {q: "Z" for q in w})]).evaluate(gradients=False).value
                for w in ((0,), (1,), (0, 1))}
    assert abs(measured[(0, 1)] - ideal) > 1e-2
    assert mt.readout_correct(circuit, "Z0*Z1", measured)["value"] == pytest.approx(ideal, abs=1e-12)
    # Symmetrised readout (TREX): the measured product is the ideal one times prod (1 - 2p).
    symmetric = {0: (0.04, 0.04), 1: (0.07, 0.07)}
    assert mt.readout_correct(symmetric, "Z0*Z1", ideal * 0.92 * 0.86)["value"] == pytest.approx(ideal, abs=1e-12)


def test_readout_mitigation_from_counts(readout_only):
    circuit, ideal = readout_only
    counts = sample(circuit, 40000, seed=11)["counts"]
    corrected = mt.readout_correct(circuit, "Z0*Z1", counts)
    assert abs(corrected["value"] - ideal) < 4 * corrected["standard_error"]
    mitigated = mt.mitigated_counts(counts, circuit)
    parity = lambda bits: (-1) ** (int(bits[-1]) + int(bits[-2]))
    quasi = sum(parity(b) * v for b, v in mitigated["quasi_probabilities"].items())
    assert quasi == pytest.approx(corrected["value"], abs=1e-12)   # the same linear inversion
    clipped = sum(parity(b) * v for b, v in mitigated["probabilities"].items())
    assert sum(mitigated["probabilities"].values()) == pytest.approx(1.0)
    assert abs(clipped - ideal) < 4 * corrected["standard_error"] + mitigated["negativity"]
    raw = sum(parity(b) * c for b, c in counts.items()) / 40000
    assert abs(raw - ideal) > 4 * corrected["standard_error"]
