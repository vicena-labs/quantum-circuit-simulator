"""Real chips from calibration snapshots, the benchmark circuit library, CAD export and annealing."""
import json
import math
import struct
import subprocess
import sys

import numpy as np
import pytest

from quantum_circuit_simulator import cad, devices
from quantum_circuit_simulator.circuits import kicked_ising, library, library_names, load
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.interop import expectation, from_qasm

REAL = ["ibm_fez", "ibm_torino", "ibm_sherbrooke", "ibm_manila", "google_willow_pink", "google_weber"]


@pytest.mark.parametrize("name", REAL)
def test_real_chips_keep_every_qubit_and_coupler(name):
    chip = devices.get(name)
    assert chip.parameters["real_device"] and chip.summary()["connected"]
    assert len(chip.qubit_overrides) == chip.n_qubits and len(chip.coupler_errors) >= 0.9 * len(chip.edges)
    assert len(chip.coordinates) == chip.n_qubits
    xy = chip.positions()
    if name != "ibm_manila":   # Heron, Eagle and Google grids are drawn on their lattices
        assert all(math.isclose(math.dist(xy[a], xy[b]), 1.0) for a, b in chip.edges)
    assert 0 < chip.error_2q < 0.05 and chip.time_2q and chip.t1
    assert devices.Device.from_dict(json.loads(chip.to_json())).to_dict() == chip.to_dict()
    with pytest.raises(ValueError, match="real chip"):
        devices.get(name, rows=2)


def test_catalogue_lists_layouts_and_real_chips():
    entries = devices.catalog()
    real = [e for e in entries if e["kind"] == "real"]
    assert len(real) >= 60 and {"ibm_fez", "google_willow_pink"} <= {e["name"] for e in real}
    assert all(e["snapshot"] for e in real if e["name"].startswith("ibm_"))


def test_a_real_chip_runs_a_compiled_circuit_with_its_own_noise():
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3]; cx q[3],q[4];'
    fez = compile_circuit(qasm, devices.get("ibm_fez"))
    value = fez.expectation("Z0*Z4").value
    assert fez.report["swaps"] == 0 and 0.7 < value < 1.0
    assert compile_circuit(qasm, devices.get("ibm_fez"), noise=False).expectation("Z0*Z4").value == pytest.approx(1.0)


def test_thermal_and_depolarizing_gate_models_agree_on_depolarizing_invariant_quantities():
    # With T1 = T2 effectively infinite the thermal model is pure depolarizing.
    chip = devices.get("line", n=3).with_noise(t1=1e3, t2=1e3, readout=None)
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; h q[0]; cx q[0],q[1]; cx q[1],q[2];'
    thermal = compile_circuit(qasm, chip, layout="trivial").expectation("Z0*Z2").value
    depolarizing = compile_circuit(qasm, chip, layout="trivial", gate_model="depolarizing").expectation("Z0*Z2").value
    assert thermal == pytest.approx(depolarizing, abs=1e-6)


def test_matches_qiskit_aer_on_an_ibm_snapshot():
    aer = pytest.importorskip("qiskit_aer")
    runtime = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from qiskit import QuantumCircuit, transpile
    fake, chain = runtime.FakeLagosV2(), [0, 1, 3, 5, 6]
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    t = transpile(qc, fake, initial_layout=chain, optimization_level=0, seed_transpiler=1)
    t.measure_all()
    counts = aer.AerSimulator.from_backend(fake, method="density_matrix").run(t, shots=200_000, seed_simulator=3) \
        .result().get_counts()
    reference = sum(c * (-1) ** (int(b[::-1][0]) + int(b[::-1][6])) for b, c in counts.items()) / 200_000
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[5]; h q[0];' + "".join(f" cx q[{i}],q[{i+1}];" for i in range(4))
    ours = compile_circuit(qasm, devices.get("ibm_lagos"), layout=chain, idle_noise=False).expectation("Z0*Z4").value
    assert abs(ours - reference) < 0.01   # Aer's backend model has no idle noise


def test_library_circuits_load_and_simulate():
    names = library_names()
    assert len(library()) > 100 and any(n.startswith("mqt/") for n in names) and any(n.startswith("supermarq/") for n in names)
    qft = load("mqt/qft", 8)
    converted = from_qasm(qft)
    assert converted.circuit.n_qubits >= 8
    ghz = load("mqt/ghz", 8)
    n = from_qasm(ghz).circuit.n_qubits
    assert expectation(ghz, f"Z0*Z{n - 1}").value == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(KeyError, match="available"):
        load("mqt/nonexistent")


def test_kicked_ising_reproduces_the_utility_experiment_structure():
    qasm, observable = kicked_ising(steps=2)
    c = from_qasm(qasm).circuit
    assert c.n_qubits == 127 and observable.count("Z") == 127
    # theta_h = 0 is a Clifford evolution of |0...0>: the magnetisation stays 1.
    exact, _ = kicked_ising(steps=3, theta_h=0.0)
    assert expectation(exact, " + ".join(f"{1 / 127}*Z{q}" for q in range(127))).value == pytest.approx(1.0, abs=1e-12)


def test_cad_files_are_valid(tmp_path):
    chip = devices.get("ibm_torino")
    info = cad.write(chip, tmp_path / "torino.gds")
    data = (tmp_path / "torino.gds").read_bytes()
    assert struct.unpack(">HH", data[:4]) == (6, 0x0002) and data.endswith(struct.pack(">HH", 4, 0x0400))
    assert info["qubits"] == 133 and info["couplers"] == 150
    gdstk = pytest.importorskip("gdstk")
    cell = gdstk.read_gds(str(tmp_path / "torino.gds")).top_level()[0]
    assert len(cell.paths) == 150 and len(cell.labels) == 133
    cad.write(chip, tmp_path / "torino.svg")
    assert (tmp_path / "torino.svg").read_text().startswith("<svg")
    cad.write(devices.get("all-to-all-ions", n=6), tmp_path / "ions.dxf")
    assert "ENTITIES" in (tmp_path / "ions.dxf").read_text()
    assert cad._real8(1e-3)[0] == 0x3E and cad._real8(1e-9)[0] == 0x39   # GDSII excess-64 exponents
    # STL: closed, outward-facing solids, so the signed volume is the slab plus the raised metal.
    cad.write(chip, tmp_path / "torino.stl")
    stl = (tmp_path / "torino.stl").read_bytes()
    count = struct.unpack("<I", stl[80:84])[0]
    assert len(stl) == 84 + 50 * count
    v = np.frombuffer(stl[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"]
    v = v.reshape(-1, 3, 3).astype(float)
    volume = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6
    width, height = info["size_um"]
    assert width * height * cad.SUBSTRATE_UM < volume < width * height * cad.SUBSTRATE_UM * 1.05


def test_annealing_finds_ground_states_directly_and_on_a_chip():
    pytest.importorskip("dimod")
    pytest.importorskip("dwave.samplers")
    pytest.importorskip("minorminer")
    pytest.importorskip("dwave.embedding")
    from quantum_circuit_simulator import annealing, hamiltonians as hm
    problem = annealing.Problem.from_hamiltonian(hm.maxcut(hm.random_regular(12, 3, seed=4)))
    ground, _ = problem.ground_state()
    direct = annealing.anneal(problem, sampler="sa", num_reads=50, seed=1)
    assert direct["best_energy"] == pytest.approx(ground) and direct["ground_energy"] == pytest.approx(ground)
    chip = devices.get("chimera", m=4)
    embedded = annealing.anneal(problem, chip, sampler="sqa", num_reads=50, seed=2, control_noise=0.01)
    assert embedded["best_energy"] == pytest.approx(ground) and embedded["embedding"]["physical_qubits"] >= 12
    assert 0 <= embedded["chain_break_fraction"] <= 1
    qubo = annealing.Problem.from_qubo({(0, 0): -1, (1, 1): -1, (0, 1): 2})
    assert qubo.ground_state()[0] == -1
    assert annealing.Problem.from_dict(qubo.to_dict()).ground_state()[0] == -1
    with pytest.raises(ValueError, match="gate-model"):
        annealing.anneal(problem, devices.get("line", n=20))


def cli(*args):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], capture_output=True, text=True,
                          timeout=300)
    return done.returncode, json.loads(done.stdout)


def test_cli_lists_library_and_writes_cad(tmp_path):
    code, out = cli("circuits", "--suite", "MQT Bench", "--search", "qft")
    assert code == 0 and out["schema"] == "quantum-circuit-simulator-circuits/1" and out["library"]
    code, out = cli("circuits", "mqt/ghz", "--size", "4", "--out", str(tmp_path / "ghz.qasm"))
    assert code == 0 and (tmp_path / "ghz.qasm").read_text().startswith("OPENQASM 2.0")
    code, out = cli("cad", "google_willow_pink", "--out", str(tmp_path / "willow.gds"))
    assert code == 0 and out["qubits"] == 105 and (tmp_path / "willow.gds").stat().st_size > 10_000


@pytest.mark.parametrize("name, qubits, degree", [("dwave-advantage", 5640, 15), ("dwave-advantage2", 4800, 20)])
def test_dwave_advantage_devices(name, qubits, degree):
    pytest.importorskip("dwave_networkx")
    chip = devices.get(name)
    assert chip.family == "annealer" and chip.n_qubits == qubits and chip.summary()["max_degree"] == degree
    assert chip.parameters["annealer"]["j_range"] == [-1.0, 1.0] and len(chip.coordinates) == qubits
