"""Compiling onto chips: exact equivalence, coupling, parameters, noise and the command line."""
import json
import math
import subprocess
import sys

import pytest

from quantum_circuit_simulator import devices
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.interop import ConversionError, expectation, parse_observable, readout_expand

QASM = """OPENQASM 2.0; include "qelib1.inc"; qreg q[6];
h q; ry(0.3) q[0]; cx q[0],q[5]; cx q[1],q[4]; rzz(0.7) q[2],q[5]; cx q[3],q[0]; rx(0.4) q[4];
cz q[5],q[1]; cx q[2],q[3]; ry(-0.6) q[1]; cx q[4],q[0];"""
OBS = "Z0*Z5 + 0.5*X1*X4 - 0.3*Y2*Z3 + 0.2*X3"
CHIPS = [("line", {"n": 8}), ("heavy-hex-127", {}), ("all-to-all-ions", {"n": 6}),
         ("square-grid", {"rows": 3, "columns": 3}), ("neutral-atom-grid", {"rows": 3, "columns": 3})]


@pytest.mark.parametrize("name, params", CHIPS)
@pytest.mark.parametrize("layout", ["greedy", "trivial"])
def test_noise_free_compilation_is_exact_and_respects_coupling(name, params, layout):
    chip = devices.get(name, **params)
    compiled = compile_circuit(QASM, chip, noise=False, layout=layout)
    assert compiled.expectation(OBS).value == pytest.approx(expectation(QASM, OBS).value, abs=1e-12)
    assert chip.uncoupled_pairs(compiled.circuit) == []
    if chip.family == "trapped-ion":
        assert compiled.report["swaps"] == 0


def test_long_range_gates_on_a_line_need_swaps_and_layout_reduces_them():
    chip = devices.get("line", n=6)
    trivial = compile_circuit(QASM, chip, noise=False, layout="trivial").report
    greedy = compile_circuit(QASM, chip, noise=False, layout="greedy").report
    assert trivial["swaps"] > 0 and greedy["swaps"] <= trivial["swaps"]
    assert trivial["two_qubit_gates"] == 8 + 3 * trivial["swaps"]  # each SWAP is three two-qubit gates


def test_sabre_router_is_exact():
    pytest.importorskip("qiskit")
    chip = devices.get("heavy-hex-127")
    compiled = compile_circuit(QASM, chip, noise=False, router="sabre")
    assert compiled.expectation(OBS).value == pytest.approx(expectation(QASM, OBS).value, abs=1e-12)
    assert chip.uncoupled_pairs(compiled.circuit) == []


def test_shared_parameters_and_gradients_survive_compilation():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    a, b = Parameter("a"), Parameter("b")
    qc = QuantumCircuit(4)
    qc.h(range(4)); qc.ry(a, 0); qc.cx(0, 3); qc.rz(-b + 0.2, 3); qc.cx(1, 2); qc.ry(b, 1); qc.cx(3, 1)
    compiled = compile_circuit(qc, devices.get("line", n=5), noise=False, layout="trivial")
    values = {"a": 0.37, "b": -0.81}
    got = compiled.expectation("Z1*X3 + Z0", values=values, gradients=True)
    want = expectation(qc, "Z1*X3 + Z0", values=values, gradients=True)
    assert got.value == pytest.approx(want.value, abs=1e-12)
    assert got.gradient == pytest.approx(want.gradient, abs=1e-12)


def test_noise_lowers_values_in_order_of_scale():
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; ry(0.4) q[0]; cx q[0],q[3]; cx q[3],q[1]; cx q[1],q[2];'
    chip = devices.get("square-grid", rows=2, columns=2)
    values = [compile_circuit(qasm, chip, noise_scale=s).expectation("Z0*Z3").value for s in (0, 1, 3)]
    assert values[0] == pytest.approx(1.0, abs=1e-12) and values[0] > values[1] > values[2]


def test_idle_dephasing_reduces_coherence():
    # q0 holds |+> while q1 runs ten X gates; a short T2 dephases q0 during the wait.
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0];' + ' x q[1];' * 10
    chip = devices.get("line", n=2).with_noise(error_1q=0., error_2q=0., readout=None, t1=1e-3, t2=1e-6,
                                               time_1q=1e-7)
    with_idle = compile_circuit(qasm, chip, layout="trivial").expectation("X0").value
    without = compile_circuit(qasm, chip, layout="trivial", idle_noise=False).expectation("X0").value
    assert without == pytest.approx(1.0, abs=1e-12)
    # q0 is busy during its own H (0.1 us), then idles 0.9 us while q1 finishes.
    assert with_idle == pytest.approx(math.exp(-0.9e-6 / 1e-6), rel=1e-9)


def test_readout_error_applies_to_any_measured_axis():
    terms = readout_expand(parse_observable("X0*Y1"), (0.02, 0.03))
    beta, alpha = 1 - 0.05, 0.03 - 0.02
    assert dict((tuple(sorted(p.items())), c) for c, p in terms) == pytest.approx({
        (): alpha * alpha, ((0, "X"),): beta * alpha, ((1, "Y"),): alpha * beta, ((0, "X"), (1, "Y")): beta * beta})
    chip = devices.get("line", n=1).with_noise(error_1q=0., readout=(0.02, 0.03))
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
    assert expectation(qasm, "X0", device=chip).value == pytest.approx(alpha + beta * 1.0, abs=1e-12)


def test_compile_refuses_annealers_and_oversized_circuits():
    with pytest.raises(ConversionError, match="annealer"):
        compile_circuit(QASM, devices.get("chimera", m=1))
    with pytest.raises(ConversionError, match="larger device"):
        try:
            compile_circuit(QASM, devices.get("line", n=4))
        except ConversionError as exc:
            raise ConversionError(exc.hint) from exc


def cli(*args):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], capture_output=True, text=True, encoding="utf-8", timeout=60)
    return done.returncode, json.loads(done.stdout)


def test_compile_and_run_compile_commands(tmp_path):
    path = tmp_path / "c.qasm"
    path.write_text(QASM)
    code, out = cli("compile", str(path), "--device", "line", "--device-param", "n=8")
    assert code == 0 and out["compile"]["swaps"] >= 0 and len(out["compile"]["final_layout"]) == 6
    code, ideal = cli("run", str(path), "--observable", OBS, "--value-only")
    code2, routed = cli("run", str(path), "--observable", OBS, "--device", "line", "--device-param", "n=8",
                        "--compile", "--noise-scale", "0", "--value-only")
    assert code == code2 == 0
    assert routed["results"][0]["value"] == pytest.approx(ideal["results"][0]["value"], abs=1e-12)
    code, err = cli("compile", str(path))
    assert code == 2 and "--device" in err["error"]["hint"]


@pytest.mark.parametrize("chip, fam, n", [
    (("heavy-hex-127", {}), "ghz", 16), (("heavy-hex-127", {}), "qaoa-ring", 12),
    (("heavy-hex-127", {}), "ising-trotter", 10), (("square-grid", {"rows": 4, "columns": 4}), "ghz", 16),
    (("square-grid", {"rows": 4, "columns": 4}), "qaoa-ring", 12), (("square-grid", {"rows": 6, "columns": 6}), "qaoa-ring", 8),
])
def test_easy_placements_need_no_swaps(chip, fam, n):
    from quantum_circuit_simulator.circuits import family
    qasm, observable = family(fam, n=n)
    device = devices.get(chip[0], **chip[1])
    compiled = compile_circuit(qasm, device, noise=False)
    assert compiled.report["layout"] == "exact" and compiled.report["swaps"] == 0
    assert compiled.expectation(observable).value == pytest.approx(expectation(qasm, observable).value, abs=1e-12)


def test_rings_shorter_than_heavy_hex_cycles_fall_back_to_the_best_greedy_start():
    from quantum_circuit_simulator.circuits import family
    qasm, observable = family("qaoa-ring", n=6)  # heavy-hex's shortest cycle has 12 qubits
    compiled = compile_circuit(qasm, devices.get("heavy-hex-127"), noise=False)
    assert compiled.report["layout"] == "greedy" and compiled.report["swaps"] > 0
    assert compiled.expectation(observable).value == pytest.approx(expectation(qasm, observable).value, abs=1e-12)
    # A qubit interacting with more partners than any chip qubit has neighbours cannot be placed exactly.
    star = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5];' + "".join(f" cz q[0],q[{i}];" for i in range(1, 5))
    assert compile_circuit(star, devices.get("heavy-hex-127"), noise=False).report["layout"] == "greedy"
