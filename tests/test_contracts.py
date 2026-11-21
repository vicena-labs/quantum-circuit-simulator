"""Output contracts: chip coordinates, compile schedules and noise budgets, envelopes, progress and checkpoints."""
import json
import math
import subprocess
import sys

import pytest

from quantum_circuit_simulator import design, devices
from quantum_circuit_simulator.circuits import family
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.contracts import Checkpoint
from quantum_circuit_simulator.experiments import sweep


def neighbour_distances(chip):
    xy = chip.positions()
    return [math.dist(xy[a], xy[b]) for a, b in chip.edges]


@pytest.mark.parametrize("name, params", [("heavy-hex-127", {}), ("square-grid", {"rows": 3, "columns": 4}),
                                          ("line", {"n": 5}), ("neutral-atom-grid", {"rows": 3, "columns": 3})])
def test_catalogue_chips_carry_lattice_coordinates(name, params):
    chip = devices.get(name, **params)
    xy = chip.coordinates
    assert len(xy) == chip.n_qubits and len(set(xy)) == chip.n_qubits
    distances = neighbour_distances(chip)
    limit = 1.5 if name == "neutral-atom-grid" else 1.0
    assert all(d <= limit + 1e-9 for d in distances)
    assert devices.Device.from_dict(json.loads(chip.to_json())).coordinates == xy


def test_design_operations_keep_coordinates_and_custom_chips_get_a_layout():
    grid = devices.get("square-grid", rows=2, columns=2)
    tiled = design.modular(grid, 3, links="line")
    assert len(set(tiled.coordinates)) == 12 and max(x for x, _ in tiled.coordinates) == 7
    trimmed = design.remove_qubits(devices.get("line", n=5), [2])
    assert trimmed.coordinates == ((0, 0), (1, 0), (3, 0), (4, 0))
    ring = design.from_edges("ring", 8, [(i, (i + 1) % 8) for i in range(8)])
    assert ring.coordinates == () and len(set(ring.positions())) == 8
    assert all(0.7 < d < 1.4 for d in neighbour_distances(ring))
    assert len(design.expand(grid, 20).coordinates) == design.expand(grid, 20).n_qubits


def test_compile_schedule_and_noise_budget_are_consistent():
    qasm, _ = family("qaoa-ring", n=6)
    chip = design.with_crosstalk(devices.get("square-grid", rows=2, columns=3), spectator_error=1e-3, zz_hz=2e4)
    compiled = compile_circuit(qasm, chip)
    report, schedule = compiled.report, compiled.schedule
    # Gates on a qubit never overlap, and the last one ends at the circuit's duration.
    ends = {}
    for gate in schedule:
        for q in gate["qubits"]:
            assert gate["start_s"] >= ends.get(q, 0.0) - 1e-15
            ends[q] = gate["start_s"] + gate["duration_s"]
    assert max(ends.values()) == pytest.approx(report["duration_s"])
    budget = report["noise_budget"]
    qubits, couplers, totals = budget["qubits"], budget["couplers"], budget["totals"]
    assert sum(c.get("gate_error", 0) for c in couplers.values()) == pytest.approx(totals["gate_2q"])
    assert sum(q.get("idle_error", 0) for q in qubits.values()) == pytest.approx(totals["idle"])
    assert sum(c.get("gates", 0) for c in couplers.values()) == report["two_qubit_gates"]
    assert totals["zz_phase_rad"] > 0 and 0 < budget["estimated_fidelity"] < 1
    assert compile_circuit(qasm, chip, noise=False).report["noise_budget"]["estimated_fidelity"] == 1.0


def cli(*args, stdin=None):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], input=stdin, capture_output=True,
                          text=True, encoding="utf-8", timeout=300)
    return done.returncode, json.loads(done.stdout), done.stderr


QASM = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];'


@pytest.mark.parametrize("args, kind", [
    (["hamiltonian", "--model", "heisenberg", "--size", "4"], "hamiltonian"),
    (["optimize", "--model", "transverse-field-ising", "--size", "4", "--steps", "20"], "optimize"),
    (["optimize", "--algorithm", "qaoa", "--model", "maxcut", "--lattice", "regular", "--size", "6,3",
      "--steps", "20"], "optimize"),
    (["sample", "-", "--shots", "200", "--seed", "1", "--device", "line:n=4"], "sample"),
    (["statevector", "-"], "statevector"),
    (["zne", "-", "--observable", "Z0*Z3", "--device", "line:n=4"], "zne"),
    (["shot-noise", "-", "--observable", "Z0*Z1 + X0*X1*X2*X3", "--shots", "1000"], "shot-noise"),
    (["compile", "-", "--device", "line:n=4", "--schedule"], "compile"),
    (["run", "-", "--observable", "Z0*Z3"], "result"),
])
def test_every_command_returns_a_versioned_envelope_with_provenance(args, kind):
    code, out, _ = cli(*args, stdin=QASM if "-" in args else None)
    assert code == 0 and out["ok"] and out["schema"] == f"quantum-circuit-simulator-{kind}/1"
    assert {"version", "request_sha256", "created_utc"} <= set(out["provenance"])
    if kind == "compile":
        assert out["compile"]["schedule"] and out["compile"]["noise_budget"]["totals"]


def test_sweep_progress_and_checkpoint_resume(tmp_path):
    path = tmp_path / "sweep.jsonl"
    args = ["sweep", "--family", "ghz", "--sizes", "4,6", "--device", "line:n=6", "--noise-scales", "0,1",
            "--checkpoint", str(path)]
    code, first, err = cli(*args, "--progress")
    events = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    assert code == 0 and len(events) == len(first["rows"]) == 6 and events[-1]["done"] == 6
    code, second, _ = cli(*args)
    assert [r["value"] for r in second["rows"]] == [r["value"] for r in first["rows"]]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1 + 6
    with pytest.raises(ValueError, match="different request"):
        Checkpoint(path, {"something": "else"})


def test_sweep_resumes_without_recomputing(tmp_path):
    path = tmp_path / "s.jsonl"
    rows = sweep("ghz", sizes=[4], devices=["line:n=4"], noise_scales=[1], checkpoint=path).rows
    lines = path.read_text(encoding="utf-8").splitlines()
    stored = json.loads(lines[2])
    stored["row"]["value"] = 123.0          # a resumed point is read, not recomputed
    path.write_text("\n".join(lines[:2] + [json.dumps(stored)]) + "\n")
    again = sweep("ghz", sizes=[4], devices=["line:n=4"], noise_scales=[1], checkpoint=path).rows
    assert again[1]["value"] == 123.0 and again[0]["value"] == rows[0]["value"]
