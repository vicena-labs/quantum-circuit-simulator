"""Per-qubit noise, crosstalk and growing chips, checked against exact values."""
import json
import math
import subprocess
import sys

import pytest

from quantum_circuit_simulator import design, devices
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.experiments import sweep

HEADER = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[{n}];'


def quiet(n, **changes):
    """A line chip with no noise except what a test sets."""
    base = dict(error_1q=0., error_2q=0., readout=None, t1=1.0, t2=1.0, time_1q=1e-7, time_2q=3e-7)
    return devices.get("line", n=n).with_noise(**{**base, **changes})


def value(qasm, chip, observable, **kw):
    # The exact formulas below use the depolarizing gate model; the thermal model has its own test.
    return compile_circuit(qasm, chip, layout="trivial", gate_model="depolarizing", **kw).expectation(observable).value


def test_per_qubit_readout_error_and_one_qubit_error():
    chip = quiet(2, readout=(0.01, 0.02), qubit_overrides={1: {"readout": (0.05, 0.10), "error_1q": 0.004}})
    qasm = HEADER.format(n=2) + " x q[0]; x q[1];"
    beta0, alpha0 = 1 - 0.03, 0.02 - 0.01
    beta1, alpha1 = 1 - 0.15, 0.10 - 0.05
    assert value(qasm, chip, "Z0") == pytest.approx(alpha0 - beta0, abs=1e-12)
    # X with one-qubit infidelity r: depolarizing p = 3r/2 scales Z by 1 - 4p/3 = 1 - 2r.
    assert value(qasm, chip, "Z1") == pytest.approx(alpha1 - beta1 * (1 - 2 * 0.004), abs=1e-12)


def test_per_qubit_t2_dephases_only_that_qubit():
    chip = quiet(3, t2=1e-5, qubit_overrides={1: {"t2": 1e-6}})
    qasm = HEADER.format(n=3) + " h q[0]; h q[1];" + " x q[2];" * 10   # q0 and q1 idle 0.9 us
    assert value(qasm, chip, "X0") == pytest.approx(math.exp(-0.9e-6 / 1e-5), rel=1e-9)
    assert value(qasm, chip, "X1") == pytest.approx(math.exp(-0.9e-6 / 1e-6), rel=1e-9)


def test_spectator_error_hits_started_neighbours_of_two_qubit_gates():
    chip = quiet(3, crosstalk={"spectator_error": 0.01})
    qasm = HEADER.format(n=3) + " h q[2]; cz q[0],q[1];"   # q2 neighbours q1 and has started
    assert value(qasm, chip, "X2", idle_noise=False) == pytest.approx(1 - 2 * 0.01, abs=1e-12)
    untouched = HEADER.format(n=3) + " cz q[0],q[1]; h q[2];"   # q2 starts after the gate
    assert value(untouched, chip, "X2", idle_noise=False) == pytest.approx(1.0, abs=1e-12)


def test_zz_crosstalk_accumulates_while_qubits_idle():
    # q0 in |1>, q1 in |+>; both idle for 1 us while q3 (far from them) runs ten X gates.
    chip = quiet(4, crosstalk={"zz_hz": 1e5})
    qasm = HEADER.format(n=4) + " x q[0]; h q[1];" + " x q[3];" * 10
    theta = 2 * math.pi * 1e5 * 1e-6
    assert value(qasm, chip, "X1", idle_noise=False) == pytest.approx(math.cos(theta), abs=1e-12)
    assert value(qasm, chip, "X1", idle_noise=False, noise_scale=2) == pytest.approx(math.cos(2 * theta), abs=1e-12)


def test_zz_crosstalk_continues_on_spectators_through_a_gate():
    # q1 in |1> and q2 in |+> from t = 0; the CZ on (q0, q1) runs from 0.1 to 0.4 us.
    chip = quiet(3, crosstalk={"zz_hz": 1e5})
    qasm = HEADER.format(n=3) + " x q[1]; h q[2]; cz q[0],q[1];"
    assert value(qasm, chip, "X2", idle_noise=False) == pytest.approx(math.cos(2 * math.pi * 1e5 * 4e-7), abs=1e-12)


def test_a_couplers_own_gate_replaces_its_zz():
    # Pair (q0, q1) accumulates ZZ for 0.1 us before its CZ and none during it.
    chip = quiet(2, crosstalk={"zz_hz": 1e5})
    qasm = HEADER.format(n=2) + " x q[0]; h q[1]; cz q[0],q[1];"
    assert value(qasm, chip, "X1", idle_noise=False) == pytest.approx(-math.cos(2 * math.pi * 1e5 * 1e-7), abs=1e-12)


def test_overrides_and_crosstalk_round_trip_scale_and_validate():
    chip = design.with_crosstalk(design.vary_qubits(devices.get("square-grid", rows=2, columns=3), 0.3, seed=4),
                                 spectator_error=1e-3, zz_hz=2e4)
    restored = devices.Device.from_dict(json.loads(chip.to_json()))
    assert restored.to_dict() == chip.to_dict()
    double = chip.scaled_noise(2)
    assert double.crosstalk == pytest.approx({"spectator_error": 2e-3, "zz_hz": 4e4})
    assert double.qubit(3, "error_1q") == pytest.approx(2 * chip.qubit(3, "error_1q"))
    assert double.qubit(3, "t1") == chip.qubit(3, "t1")
    for bad in ({9: {"t1": 1e-4}}, {0: {"gate_time": 1}}, {0: {"readout": (0.6, 0.1)}}):
        with pytest.raises(ValueError):
            chip.with_noise(qubit_overrides=bad)
    with pytest.raises(ValueError):
        chip.with_noise(crosstalk={"leakage": 0.1})


def test_vary_qubits_is_reproducible_and_physical():
    chip = devices.get("square-grid", rows=5, columns=5)
    a, b = design.vary_qubits(chip, 0.25, seed=7), design.vary_qubits(chip, 0.25, seed=7)
    assert a.qubit_overrides == b.qubit_overrides and len(a.qubit_overrides) == 25
    assert all(v["t2"] <= 2 * v["t1"] for v in a.qubit_overrides.values())
    errors = [v["error_1q"] for v in a.qubit_overrides.values()]
    assert len(set(errors)) == 25 and 0.5 * chip.error_1q < sum(errors) / 25 < 2 * chip.error_1q


@pytest.mark.parametrize("spec, target", [
    (("heavy-hex-127", {}), 300), (("square-grid", {"rows": 4, "columns": 6}), 200), (("line", {"n": 4}), 30),
    (("all-to-all-ions", {"n": 8}), 20), (("neutral-atom-grid", {"rows": 3, "columns": 3}), 40), (("chimera", {"m": 2}), 300),
])
def test_expand_keeps_the_architecture(spec, target):
    chip = devices.get(spec[0], **spec[1]).with_noise(error_2q=7e-3)
    grown = design.expand(chip, target)
    facts = grown.summary()
    assert grown.n_qubits >= target and facts["connected"] and grown.family == chip.family
    # Degrees follow the architecture: heavy-hex 3, grids 4, Chimera 6 once it has interior cells.
    expected = {"heavy-hex-127": 3, "square-grid": 4, "line": 2, "neutral-atom-grid": 8, "chimera": 6}
    if spec[0] in expected:
        assert facts["max_degree"] == expected[spec[0]]
    assert grown.error_2q == 7e-3 and grown.parameters["expanded_from"] == chip.name
    if spec[0] == "square-grid":
        r, c = grown.parameters["catalog_parameters"]["rows"], grown.parameters["catalog_parameters"]["columns"]
        assert abs(c / r - 6 / 4) < 0.2


def test_expand_modular_and_custom_chips():
    modular = design.modular(devices.get("square-grid", rows=2, columns=2), 3, links="ring", link_error=0.03)
    grown = design.expand(modular, 30)
    assert grown.n_qubits == 32 and grown.parameters["modules"] == 8 and set(grown.coupler_errors.values()) == {0.03}
    custom = design.from_edges("tri", 3, [(0, 1), (1, 2), (0, 2)], error_2q=1e-2)
    tiled = design.expand(custom, 10)
    assert tiled.n_qubits == 12 and tiled.summary()["connected"]
    assert [c.n_qubits >= n for c, n in zip(design.series(custom, [3, 6, 9]), [3, 6, 9])] == [True] * 3


def test_sweep_grows_the_chip_to_fit_each_size():
    rows = sweep("ghz", sizes=[10, 40], devices=["fit:square-grid:rows=2,columns=2"], noise_scales=[0]).rows
    chips = [r["device"] for r in rows if r["device"] != "ideal"]
    assert chips == ["square-grid-4x4", "square-grid-7x7"]
    assert all(r["value"] == pytest.approx(1.0, abs=1e-12) and r["swaps"] == 0 for r in rows if r["device"] != "ideal")


def test_design_and_sweep_commands_with_new_options(tmp_path):
    out = tmp_path / "chip.json"
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", "design", "--base", "heavy-hex-127", "--expand-to", "300",
                           "--qubit-spread", "0.2", "--spectator-error", "1e-3", "--zz-hz", "5e4", "--out", str(out)],
                          capture_output=True, text=True, timeout=120)
    facts = json.loads(done.stdout)
    assert done.returncode == 0 and facts["n_qubits"] >= 300 and facts["qubits_with_own_noise"] == facts["n_qubits"]
    assert facts["crosstalk"] == {"spectator_error": 1e-3, "zz_hz": 5e4}
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", "sweep", "--family", "ghz", "--sizes", "6,20",
                           "--device", f"fit:{out}", "--noise-scales", "1"], capture_output=True, text=True, timeout=300)
    rows = json.loads(done.stdout)["rows"]
    assert done.returncode == 0 and all("error" not in r for r in rows) and len(rows) == 4
