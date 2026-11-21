"""The direct RZZ stage, uninformative truncation bounds and the convergence ladder."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, Truncation
from quantum_circuit_simulator.circuits import kicked_ising
from quantum_circuit_simulator.experiments import ladder, refine
from quantum_circuit_simulator.interop import from_qasm

OBSERVABLE = [(1.0, {1: "Z"}), (0.5, {0: "X", 2: "Y"}), (-0.3, {1: "X", 3: "Z"})]


def pair(direct):
    """Four qubits with shared-parameter ZZ rotations, as one RZZ stage or as H.CZ.H.RZ.H.CZ.H."""
    c = Circuit(4, 2)
    for q in range(4):
        c.gate("RX", [q], 0.3 + 0.2 * q, 0)
    for a, b in ((0, 1), (2, 3), (1, 2)):
        if direct:
            c.gate("RZZ", [a, b], 0.7, 1, 1.5)
        else:
            c.gate("H", [b]).gate("CZ", [a, b]).gate("H", [b]).gate("RZ", [b], 0.7, 1, 1.5)
            c.gate("H", [b]).gate("CZ", [a, b]).gate("H", [b])
        c.depolarize([a, b], 0.02)
    c.relax(2, 1e-7, 1e-4, 8e-5)
    return c


@pytest.mark.parametrize("backend", ["native", "graph", "dense", "python"])
def test_rzz_stage_equals_its_cz_decomposition_with_gradients(backend):
    shifts = [0.25, -0.4]
    direct = pair(True).prepare(OBSERVABLE, backend=backend).evaluate(shifts)
    decomposed = pair(False).prepare(OBSERVABLE, backend="native").evaluate(shifts)
    assert direct.value == pytest.approx(decomposed.value, abs=1e-12)
    assert np.allclose(direct.gradient, decomposed.gradient, atol=1e-12)


def test_rzz_stage_agrees_with_the_state_simulators():
    from quantum_circuit_simulator.statevector import StateVectorPlan
    c = pair(True)
    value, _ = StateVectorPlan(c, OBSERVABLE).evaluate([0.25, -0.4], gradients=False)
    assert value == pytest.approx(c.prepare(OBSERVABLE).evaluate([0.25, -0.4]).value, abs=1e-10)


def test_uncompiled_openqasm_rzz_is_one_stage_and_compiled_keeps_chip_gates():
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nrx(0.4) q[0];\nrzz(0.9) q[0],q[1];\n'
            'rxx(0.3) q[0],q[1];\nryy(-0.2) q[0],q[1];\nrzx(0.5) q[0],q[1];\n')
    converted = from_qasm(qasm)
    assert sum(s.name == "RZZ" for s in converted.circuit._impl.stages) == 4
    terms = [(1.0, {0: "Z"}), (0.7, {0: "X", 1: "Y"})]
    from quantum_circuit_simulator.statevector import StateVectorPlan
    exact, _ = StateVectorPlan(converted.circuit, terms).evaluate(None, gradients=False)
    import quantum_circuit_simulator.interop as interop
    ops, params = interop._lower(interop._parse_qasm2(qasm)[1])
    decomposed = interop._emit(2, ops, params, None, 1000).circuit
    assert "RZZ" not in {s.name for s in decomposed._impl.stages}
    assert decomposed.prepare(terms).evaluate().value == pytest.approx(exact, abs=1e-12)


def test_deep_truncation_flags_an_uninformative_bound():
    qasm, _ = kicked_ising(127, steps=6, theta_h=0.6)
    circuit = from_qasm(qasm).circuit
    shallow = circuit.prepare([(1.0, {62: "Z"})], truncation=Truncation(0.3)).evaluate(gradients=False)
    assert shallow.bound_informative is False
    assert shallow.error_bound == pytest.approx(1 + abs(shallow.value))
    tight = Circuit(2)
    tight.gate("RX", [0], 0.01).gate("RZZ", [0, 1], 0.02)
    small = tight.prepare([(1.0, {0: "Z", 1: "Z"})], truncation=Truncation(1e-3)).evaluate(gradients=False)
    assert small.bound_informative is True


def test_refine_tightens_until_two_levels_agree():
    qasm, _ = kicked_ising(127, steps=4, theta_h=0.5)
    circuit = from_qasm(qasm).circuit
    results, report = refine(circuit, [(1.0, {62: "Z"})], [None], ladder(1e-2, 6), tolerance=1e-3)
    assert report["converged"] and report["stopped"] == "converged"
    assert report["change"] <= 1e-3 and len(report["levels"]) >= 2
    thresholds = [level["min_coefficient"] for level in report["levels"]]
    assert thresholds == sorted(thresholds, reverse=True)
    assert results[0].value == report["levels"][-1]["values"][0]


def test_refine_keeps_the_finest_level_when_the_term_cap_stops_it():
    qasm, _ = kicked_ising(127, steps=6, theta_h=0.6)
    circuit = from_qasm(qasm, max_terms=3000).circuit
    results, report = refine(circuit, [(1.0, {62: "Z"})], [None], ladder(1e-2, 6), tolerance=1e-9)
    assert results is not None and not report["converged"]
    assert "truncated cone" in report["stopped"] and "exceeds cap 3000" in report["stopped"]


def test_cli_converge_reports_every_level(tmp_path):
    qasm, _ = kicked_ising(127, steps=4, theta_h=0.5)
    path = tmp_path / "kicked.qasm"
    path.write_text(qasm)
    out = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", "run", str(path), "--observable", "Z62",
                          "--value-only", "--min-coefficient", "1e-2", "--converge", "1e-3", "--levels", "6"],
                         capture_output=True, text=True, encoding="utf-8", env=dict(os.environ), check=True)
    data = json.loads(out.stdout)
    convergence = data["convergence"]
    assert convergence["converged"] and len(convergence["levels"]) >= 2
    assert data["results"][0]["value"] == convergence["levels"][-1]["values"][0]
    assert "bound_informative" in data["results"][0]
