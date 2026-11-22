"""Majorana tetron chips: the published Majorana 2 cell, growing arrays, measurement costs, parity
flips and CAD."""
import math

import pytest

from quantum_circuit_simulator import cad, design, devices
from quantum_circuit_simulator.compiler import compile_circuit, tetron_cost

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_majorana_2_carries_published_values_with_sources():
    chip = devices.get("microsoft_majorana_2")
    model = chip.parameters["topological"]
    assert chip.family == "topological" and chip.n_qubits == 4 and len(chip.edges) == 4
    assert model["tau_z_s"] == 22.0 and model["measurement_time_s"] == 10e-6
    assert set(model) - {"synthesis_precision"} <= set(chip.parameters["sources"])
    # A one-qubit Clifford is one Z-type and one X-type measurement.
    assert chip.error_1q == pytest.approx(1 - (1 - 0.005) * (1 - 0.16))
    assert devices.get("microsoft_majorana_2", tau_x_s=1e-3).parameters["topological"]["tau_x_s"] == 1e-3
    assert devices.Device.from_dict(chip.to_dict()).parameters["topological"] == model


def test_tetron_arrays_grow_and_keep_their_model():
    chip = devices.get("tetron-array", rows=27, columns=13)
    assert chip.n_qubits == 351
    grown = design.expand(devices.get("microsoft_majorana_2"), 100)
    assert grown.n_qubits >= 100 and grown.family == "topological"
    assert grown.parameters["topological"]["tau_x_s"] == 14.5e-6


def test_magic_state_counts():
    model = {"synthesis_precision": 1e-3}
    assert tetron_cost("H", None, -1, model) == 0
    assert tetron_cost("RZ", math.pi / 2, -1, model) == 0       # S: a Clifford
    assert tetron_cost("RZ", math.pi / 4, -1, model) == 1       # T: one magic state
    assert tetron_cost("RZ", 0.3, -1, model) == 30              # synthesized: ceil(3 log2 1000)
    assert tetron_cost("RZ", math.pi / 4, 0, model) == 30       # parameterised: synthesized


def test_idle_parity_flips_match_the_poisson_formula():
    """An idle X-basis qubit loses <X> to phase flips: <X> = exp(-2 t / tau_x), here with error-free gates."""
    chip = devices.get("tetron-array", rows=1, columns=2, measurement_error_z=0.0, measurement_error_x=0.0,
                       tau_x_s=50e-6, tau_z_s=1e9)
    qasm = HEADER + "qreg q[2];\nh q[0];\n" + "x q[1];\n" * 10 + "cz q[0],q[1];\ncz q[0],q[1];\n"
    compiled = compile_circuit(qasm, chip, layout="trivial")
    # The two CZs cancel. Flips during a gate act after it, so qubit 0 dephases for the whole circuit,
    # its own H included.
    t = compiled.report["duration_s"]
    assert compiled.expectation("X0").value == pytest.approx(math.exp(-2 * t / 50e-6), rel=1e-9)


def test_published_majorana_2_cannot_hold_a_bell_pair_but_targets_can():
    bell = HEADER + "qreg q[2];\nh q[0];\ncx q[0],q[1];\n"
    published = compile_circuit(bell, devices.get("microsoft_majorana_2"), layout="trivial")
    target = compile_circuit(bell, devices.get("tetron-array"), layout="trivial")
    assert abs(published.expectation("Z0*Z1").value) < 0.05
    assert target.expectation("Z0*Z1").value > 0.95


def test_tetron_cad_uses_published_dimensions(tmp_path):
    chip = devices.get("microsoft_majorana_2")
    shapes = cad.layout(chip)
    wires = [p for layer, p in shapes["polygons"] if layer == cad.LAYERS["qubit"]]
    lengths = sorted(round(max(x for x, _ in p) - min(x for x, _ in p), 6) for p in wires)
    assert lengths.count(3.5) == 8                                    # two 3.5 um wires per tetron
    assert len([1 for layer, _ in shapes["polygons"] if layer == cad.LAYERS["readout"]]) == 4 * 5
    for ext in ("gds", "svg", "dxf", "stl"):
        cad.write(chip, tmp_path / f"majorana.{ext}")
        assert (tmp_path / f"majorana.{ext}").stat().st_size > 0


def test_noise_budget_names_the_dominant_synthesis_cost():
    """A synthesized rotation's error splits into its T gates' parity measurements and magic states,
    so a sweep can see which one to vary."""
    qasm = HEADER + "qreg q[1];\nrx(0.3) q[0];\n"
    published = compile_circuit(qasm, devices.get("microsoft_majorana_2"), layout="trivial")
    totals = published.report["noise_budget"]["totals"]
    assert totals["magic_states"] == 30
    assert totals["synthesis_measurements"] == pytest.approx(30 * (1 - (1 - 0.005) * (1 - 0.16)))
    assert totals["magic_state"] == pytest.approx(30 * 0.16)
    better = compile_circuit(qasm, devices.get("microsoft_majorana_2", magic_state_error=1e-6), layout="trivial")
    better_totals = better.report["noise_budget"]["totals"]
    assert better_totals["synthesis_measurements"] > 100 * better_totals["magic_state"]
