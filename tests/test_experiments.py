"""Circuit families, the solver router and sweeps over chips, noise and size."""
import json
import subprocess
import sys

import pytest

from quantum_circuit_simulator import Circuit
from quantum_circuit_simulator.circuits import FAMILIES, family
from quantum_circuit_simulator.experiments import solve, sweep
from quantum_circuit_simulator.interop import expectation, parse_observable


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_families_match_qiskit_statevector(name):
    pytest.importorskip("qiskit")
    from qiskit import qasm2
    from qiskit.quantum_info import SparsePauliOp, Statevector
    qasm, observable = family(name, n=5)
    reference = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    labels = [("".join(p.get(q, "I") for q in reversed(range(5))), c) for c, p in parse_observable(observable)]
    want = Statevector(reference).expectation_value(SparsePauliOp.from_list(labels)).real
    assert expectation(qasm, observable).value == pytest.approx(want, abs=1e-12)


def test_ghz_correlation_is_one():
    qasm, observable = family("ghz", n=9)
    assert expectation(qasm, observable).value == pytest.approx(1.0, abs=1e-12)


def _layered(n, layers, max_terms):
    c = Circuit(n, max_terms=max_terms)
    for layer in range(layers):
        for q in range(n):
            c.gate("RY", [q], 0.3 + 0.07 * q + 0.1 * layer).gate("RZ", [q], 0.2 - 0.05 * q)
        for q in range(layer % 2, n - 1, 2):
            c.gate("CZ", [q, q + 1])
    return c


def test_router_prefers_exact_and_degrades_visibly(monkeypatch):
    from quantum_circuit_simulator import experiments
    obs = [(1.0, {2: "Z"}), (0.5, {1: "X", 2: "X"})]
    exact = solve(_layered(5, 2, 100_000), obs, cross_check=True)
    assert exact["method"] == "native" and exact["exact"] and exact["cross_check"]["agrees"]
    # A tiny term cap forces the router past native; 5 wires fit the state vector.
    reference = solve(_layered(5, 4, 100_000), obs)["value"]
    small = solve(_layered(5, 4, 4), obs, gradients=True, cross_check=True)
    assert small["method"] == "state_vector" and small["exact"] and small["reasons"]
    assert small["value"] == pytest.approx(reference, abs=1e-12) and small["cross_check"]["method"] == "dense"
    # Without state simulation, 5 wires fit the dense backend.
    monkeypatch.setattr(experiments, "ROUTER_STATEVECTOR_QUBITS", 0)
    dense = solve(_layered(5, 4, 4), obs)
    assert dense["method"] == "dense" and dense["value"] == pytest.approx(reference, abs=1e-12)
    # Ten wires exceed the dense backend, so the router truncates and reports convergence.
    # Exact needs 131,088 terms here; truncation needs 72,526 (1e-4) and 114,251 (1e-5).
    wide = solve(_layered(10, 7, 120_000), [(1.0, {5: "Z"})])
    assert not wide["exact"] and wide["method"] == "native truncated at 1e-05"
    exact_value = solve(_layered(10, 7, 1_000_000), [(1.0, {5: "Z"})])["value"]
    assert abs(wide["value"] - exact_value) <= wide["error_bound"]
    assert wide["convergence_change"] is not None


def test_sweep_rows_ideal_and_noise():
    table = sweep("ghz", sizes=[4, 8], devices=["line:n=8", "all-to-all-ions:n=6"], noise_scales=[0, 1])
    rows = table.rows
    assert len(rows) == 2 * (1 + 2 * 2)
    assert {r["device"] for r in rows if r["size"] == 4} == {"ideal", "line-8", "all-to-all-ions-6"}
    for r in rows:
        if r["device"] == "ideal" or (r["noise_scale"] == 0 and "error" not in r):
            assert r["value"] == pytest.approx(1.0, abs=1e-12)
        elif "error" not in r:
            assert r["deviation"] < 0
    too_small = [r for r in rows if r["size"] == 8 and r["device"] == "all-to-all-ions-6"]
    assert too_small and all("larger device" in r["error"] or "has 6" in r["error"] for r in too_small)
    assert table.markdown().startswith("| size | device |") and table.csv().startswith("size,device")


def test_sweep_of_an_openqasm_circuit_with_a_custom_device():
    from quantum_circuit_simulator import devices
    chip = devices.get("square-grid", rows=2, columns=3).with_noise(error_2q=1e-2)
    qasm, observable = family("qaoa-ring", n=6)
    rows = sweep(qasm, observable, devices=[chip, chip.to_dict()], noise_scales=[1]).rows
    assert rows[1]["value"] == pytest.approx(rows[2]["value"], abs=1e-14)  # Device and its JSON agree


def cli(*args):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], capture_output=True, text=True, encoding="utf-8", timeout=120)
    return done.returncode, done.stdout


def test_sweep_command():
    code, out = cli("sweep", "--family", "qaoa-ring", "--sizes", "6", "--device", "square-grid:rows=2,columns=3",
                    "--noise-scales", "0,1")
    rows = json.loads(out)["rows"]
    assert code == 0 and len(rows) == 3 and rows[1]["deviation"] == pytest.approx(0, abs=1e-12)
    code, out = cli("sweep", "--family", "ghz", "--sizes", "4", "--device", "line", "--format", "markdown")
    assert code == 0 and out.startswith("| size | device |")
    code, out = cli("sweep", "--family", "ghz", "--device", "line")
    assert code == 2 and "--sizes" in json.loads(out)["error"]["hint"]
