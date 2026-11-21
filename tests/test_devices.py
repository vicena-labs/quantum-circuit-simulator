"""The chip catalogue: layouts, JSON descriptions, noise conversion and use in simulation."""
import json
import subprocess
import sys

import numpy as np
import pytest

from quantum_circuit_simulator import Circuit, devices
from quantum_circuit_simulator._core._reference import depolarizing_ptm
from quantum_circuit_simulator.interop import ConversionError, expectation

OFFLINE = [name for name in devices.CATALOG if name not in ("pegasus", "zephyr", "dwave-advantage", "dwave-advantage2")]


@pytest.mark.parametrize("name", OFFLINE)
def test_every_catalogue_device_builds_connected_and_round_trips(name):
    small = {"heavy-hex": dict(rows=3, row_length=7), "square-grid": dict(rows=3, columns=4), "line": dict(n=5),
             "all-to-all-ions": dict(n=6), "neutral-atom-grid": dict(rows=3, columns=3), "chimera": dict(m=2)}
    chip = devices.get(name, **small.get(name, {}))
    facts = chip.summary()
    assert facts["connected"] and facts["couplers"] == len(chip.edges) > 0
    assert devices.Device.from_dict(json.loads(chip.to_json())).to_dict() == chip.to_dict()
    assert chip.family in {entry["family"] for entry in devices.catalog() if entry["name"] == name}


def test_heavy_hex_127_is_the_eagle_layout():
    chip = devices.get("heavy-hex-127")
    assert (chip.n_qubits, len(chip.edges), chip.summary()["max_degree"]) == (127, 144, 3)
    for edge in [(0, 14), (14, 18), (4, 15), (15, 22), (20, 33), (33, 39), (96, 109), (109, 114), (12, 17), (17, 30)]:
        assert edge in chip.edges
    assert chip.edges == devices.get("heavy-hex", rows=7, row_length=15).edges
    with pytest.raises(ValueError, match="odd number of rows"):
        devices.get("heavy-hex", rows=4, row_length=15)


def test_layout_counts():
    grid = devices.get("square-grid", rows=4, columns=5)
    assert (grid.n_qubits, len(grid.edges)) == (20, 4 * 4 + 5 * 3)
    assert len(devices.get("all-to-all-ions", n=10).edges) == 45
    assert len(devices.get("neutral-atom-grid", rows=3, columns=3, radius=1.0).edges) == 12
    assert len(devices.get("neutral-atom-grid", rows=3, columns=3, radius=1.5).edges) == 20  # with diagonals
    chimera = devices.get("chimera", m=2, n=3, t=4)
    assert chimera.n_qubits == 48 and len(chimera.edges) == 16 * 6 + 4 * 3 * 1 + 4 * 2 * 2


def test_pegasus_and_zephyr_build_through_dwave_networkx():
    pytest.importorskip("dwave_networkx")
    pegasus, zephyr = devices.get("pegasus", m=4), devices.get("zephyr", m=2, t=4)
    assert pegasus.family == zephyr.family == "annealer"
    assert pegasus.summary()["connected"] and zephyr.summary()["connected"]
    assert pegasus.summary()["max_degree"] <= 15 and zephyr.summary()["max_degree"] <= 20


def test_chimera_matches_dwave_networkx():
    dnx = pytest.importorskip("dwave_networkx")
    ours = devices.get("chimera", m=3, n=2, t=4)
    theirs = dnx.chimera_graph(3, 2, 4)
    assert ours.edges == tuple(sorted(tuple(sorted(e)) for e in theirs.edges))


@pytest.mark.parametrize("k", [1, 2])
def test_infidelity_conversion_reproduces_the_reported_gate_fidelity(k):
    # Average gate infidelity of a channel with Pauli transfer matrix R on d = 2^k
    # levels is 1 - (Tr R / d + 1) / (d + 1).
    chip = devices.get("square-grid", rows=2, columns=2).with_noise(error_1q=4e-4, error_2q=7e-3)
    noise = chip.noise_model()
    p, reported = (noise.depolarizing_1q, 4e-4) if k == 1 else (noise.depolarizing_2q, 7e-3)
    d = 2 ** k
    trace = np.trace(np.asarray(depolarizing_ptm(p, k)))
    assert 1 - (trace / d + 1) / (d + 1) == pytest.approx(reported, rel=1e-12)


def test_noise_scaling_and_annealers_have_no_gate_noise():
    chip = devices.get("all-to-all-ions", n=4)
    double = chip.scaled_noise(2.0)
    assert double.error_2q == pytest.approx(2 * chip.error_2q)
    assert double.readout == pytest.approx(tuple(2 * p for p in chip.readout))
    with pytest.raises(ConversionError):
        devices.get("chimera", m=1).noise_model()
    with pytest.raises(ValueError):
        chip.with_noise(error_3q=0.1)


def test_uncoupled_pairs_counts_gates_not_noise():
    chip = devices.get("line", n=4)
    c = Circuit(4).gate("H", [0]).gate("CZ", [0, 1]).gate("CZ", [0, 2])
    c.depolarize([0, 2], 0.01)
    assert chip.uncoupled_pairs(c) == [(0, 2)]


BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nry(0.7) q[0];\ncx q[0], q[1];\ncx q[0], q[2];\n'


def test_device_noise_flows_into_expectation_and_estimator():
    chip = devices.get("line", n=3)
    clean = expectation(BELL, "Z0*Z1").value
    noisy = expectation(BELL, "Z0*Z1", device=chip).value
    assert 0 < noisy < clean
    assert expectation(BELL, "Z0*Z1", device=chip.scaled_noise(3.0)).value < noisy
    with pytest.raises(ConversionError, match="line-2 has 2") as info:
        expectation(BELL, "Z0", device=devices.get("line", n=2))
    assert "larger device" in info.value.hint
    qiskit = pytest.importorskip("qiskit")
    from quantum_circuit_simulator.estimator import Estimator
    from qiskit import qasm2
    evs = Estimator(device=chip).run([(qasm2.loads(BELL), "IZZ")]).result()[0].data.evs
    assert float(evs) == pytest.approx(noisy, abs=1e-14)


def cli(*args):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], capture_output=True, text=True, encoding="utf-8", timeout=60)
    return done.returncode, json.loads(done.stdout)


def test_device_commands_and_device_flag(tmp_path):
    code, listing = cli("devices")
    assert code == 0 and {"heavy-hex-127", "all-to-all-ions", "pegasus"} <= {d["name"] for d in listing}
    code, facts = cli("device", "square-grid", "--param", "rows=3", "--param", "columns=3")
    assert code == 0 and facts["n_qubits"] == 9 and facts["couplers"] == 12
    code, full = cli("device", "line", "--param", "n=3", "--edges")
    assert code == 0 and full["edges"] == [[0, 1], [1, 2]]
    path = tmp_path / "chip.json"
    path.write_text(json.dumps(full))
    qasm = tmp_path / "c.qasm"
    qasm.write_text(BELL)
    code, out = cli("run", str(qasm), "--observable", "Z0*Z1", "--device", str(path))
    assert code == 0 and out["device"]["uncoupled_two_qubit_operations"] == 1  # cx q[0], q[2] on a line
    assert out["device"]["noise_applied"] and out["results"][0]["value"] < 1
    code, err = cli("device", "no-such-chip")
    assert code == 2 and "quantum-circuit-simulator devices" in err["error"]["hint"]


def test_device_files_from_before_the_rename_still_load():
    data = devices.get("line", n=3).to_dict()
    data["schema"] = "pauli-cones-device/1"
    assert devices.Device.from_dict(data).n_qubits == 3
