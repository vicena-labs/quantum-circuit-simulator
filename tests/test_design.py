"""Chip design: shapes, modules, per-coupler errors, physics-derived errors and comparisons."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from quantum_circuit_simulator import design, devices
from quantum_circuit_simulator._core._reference import relaxation_ptm
from quantum_circuit_simulator.compiler import compile_circuit


@pytest.mark.parametrize("t, t1, t2", [(30e-9, 300e-6, 200e-6), (5e-6, 20e-6, 15e-6)])
def test_coherence_limit_matches_the_relaxation_channel(t, t1, t2):
    # Average gate infidelity of a channel with PTM R on d levels: 1 - (Tr R / d + 1) / (d + 1).
    r1 = np.asarray(relaxation_ptm(t, t1, t2))
    assert design.coherence_limited_error(t, t1, t2, 1) == pytest.approx(1 - (np.trace(r1) / 2 + 1) / 3, rel=1e-12)
    r2 = np.kron(r1, r1)
    assert design.coherence_limited_error(t, t1, t2, 2) == pytest.approx(1 - (np.trace(r2) / 4 + 1) / 5, rel=1e-12)


def test_with_physics_derives_errors_and_rejects_unphysical_t2():
    chip = design.with_physics(devices.get("line", n=4), t1=100e-6, t2=80e-6, time_1q=50e-9, time_2q=400e-9,
                               control_error_2q=1e-3)
    assert chip.error_2q == pytest.approx(design.coherence_limited_error(400e-9, 100e-6, 80e-6, 2) + 1e-3)
    assert chip.error_1q == pytest.approx(design.coherence_limited_error(50e-9, 100e-6, 80e-6, 1))
    with pytest.raises(ValueError, match="2 T1"):
        design.with_physics(chip, t1=10e-6, t2=30e-6, time_1q=1e-8, time_2q=1e-7)


def test_remove_qubits_renumbers_and_keeps_coupler_errors():
    chip = design.add_couplers(devices.get("line", n=5), [(0, 4)], error=0.05)
    smaller = design.remove_qubits(chip, [2])
    assert smaller.n_qubits == 4 and set(smaller.edges) == {(0, 1), (2, 3), (0, 3)}
    assert smaller.coupler_errors == {(0, 3): 0.05}
    assert smaller.parameters["renumbering"] == {"0": 0, "1": 1, "3": 2, "4": 3}
    with pytest.raises(ValueError):
        design.remove_couplers(chip, [(1, 3)])


def test_modular_chip_structure_and_link_errors():
    module = devices.get("square-grid", rows=2, columns=2)
    ring = design.modular(module, 3, links="ring", link_error=0.03)
    assert ring.n_qubits == 12 and len(ring.edges) == 3 * 4 + 3 and ring.summary()["connected"]
    assert set(ring.coupler_errors.values()) == {0.03} and len(ring.coupler_errors) == 3
    line = design.modular(module, 3, links="line", links_per_pair=2)
    assert len(line.edges) == 3 * 4 + 2 * 2 and not line.coupler_errors
    assert len(design.modular(module, 4, links="all").edges) == 4 * 4 + 6
    restored = devices.Device.from_dict(json.loads(ring.to_json()))
    assert restored.to_dict() == ring.to_dict()


def test_compiler_applies_each_couplers_own_error():
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
    base = devices.get("line", n=2).with_noise(error_1q=0., error_2q=1e-3, readout=None)
    bad = base.with_noise(coupler_errors={(0, 1): 0.1})
    good_value = compile_circuit(qasm, base, layout="trivial", idle_noise=False,
                                 gate_model="depolarizing").expectation("Z0*Z1").value
    bad_value = compile_circuit(qasm, bad, layout="trivial", idle_noise=False,
                                gate_model="depolarizing").expectation("Z0*Z1").value
    # One CZ under two-qubit depolarizing p = 5r/4 scales Z0 Z1 by 1 - 16p/15.
    assert good_value == pytest.approx(1 - 16 * (1e-3 * 5 / 4) / 15, abs=1e-12)
    assert bad_value == pytest.approx(1 - 16 * (0.1 * 5 / 4) / 15, abs=1e-12)
    assert bad.scaled_noise(2.0).coupler_errors[(0, 1)] == pytest.approx(0.2)


def test_compare_ranks_a_quieter_chip_first():
    noisy = devices.get("square-grid", rows=3, columns=3)
    quiet = design.with_physics(noisy, t1=1e-3, t2=1e-3, time_1q=20e-9, time_2q=100e-9)
    from dataclasses import replace
    quiet = replace(quiet, name="quiet-grid")
    rows = design.compare([noisy, quiet], family="ghz", sizes=[4, 6]).rows
    assert [r["device"] for r in rows] == ["quiet-grid", "square-grid-3x3"]
    assert rows[0]["mean_abs_deviation"] < rows[1]["mean_abs_deviation"] and rows[0]["sizes_failed"] == 0


def cli(*args):
    done = subprocess.run([sys.executable, "-m", "quantum_circuit_simulator", *args], capture_output=True, text=True, encoding="utf-8", timeout=120)
    return done.returncode, done.stdout


def test_design_and_compare_commands(tmp_path):
    out = tmp_path / "chip.json"
    code, text = cli("design", "--base", "square-grid:rows=2,columns=2", "--modules", "3", "--link-error", "0.03",
                     "--remove-qubits", "0", "--t1", "2e-4", "--t2", "1e-4", "--time-1q", "3e-8", "--time-2q",
                     "2e-7", "--name", "tri-module", "--out", str(out))
    facts = json.loads(text)
    assert code == 0 and facts["name"] == "tri-module" and facts["n_qubits"] == 11 and out.exists()
    code, text = cli("compare", str(out), "square-grid:rows=3,columns=4", "--family", "ghz", "--sizes", "4,6")
    rows = json.loads(text)["rows"]
    assert code == 0 and {r["device"] for r in rows} == {"tri-module", "square-grid-3x4"}
    code, text = cli("design", "--base", "line", "--t1", "1e-4")
    assert code == 2 and "design --help" in json.loads(text)["error"]["hint"]
