"""Fault-tolerant resource estimates: the surface-code model by hand, and T counting."""
from __future__ import annotations

import math

import pytest

from quantum_circuit_simulator import devices, resources
from quantum_circuit_simulator.compiler import compile_circuit

QASM = """OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2];
t q[0]; tdg q[1]; rz(pi/4) q[0]; rz(0.3) q[1]; s q[1]; h q[0]; cx q[0],q[1];
measure q -> c;
"""


def test_logical_error_rate_and_distance_are_monotone_and_match_hand_values():
    # p_L(d) = 0.03 (p/0.01)^((d+1)/2): at p = 1e-3, d = 11 gives 3e-8 and d = 13 gives 3e-9.
    assert resources.code_distance(3.01e-8, 1e-3) == 11
    assert resources.code_distance(2.99e-8, 1e-3) == 13
    assert resources.code_distance(3.01e-4, 1e-3) == 3
    targets = [10.0 ** -k for k in range(3, 16)]
    distances = [resources.code_distance(t, 1e-3) for t in targets]
    assert distances == sorted(distances) and all(d % 2 for d in distances)
    worse = [resources.code_distance(1e-12, p) for p in (1e-4, 1e-3, 3e-3, 6e-3)]
    assert worse == sorted(worse)
    with pytest.raises(ValueError):
        resources.code_distance(1e-9, 0.02)


def test_worked_example_by_hand():
    """Q = 10 qubits, 1000 T gates, 10 measurements, p = 1e-3, budget 1e-3, 1 us rounds.

    No rotations, so the budget splits in two: 5e-4 logical, 5e-4 distillation.
    Patches: 2*10 + ceil(sqrt(80)) + 1 = 30. Logical cycles: 1000 + 10 = 1010.
    Logical target per patch-cycle: 5e-4 / (30 * 1010) = 1.65e-8, so d = 13
    (p_L(11) = 3e-8 is too large, p_L(13) = 3e-9). Algorithm qubits:
    30 * 2 * 13^2 = 10,140. One 15-to-1 level: 35e-9 + 121 * 3e-9 = 3.98e-7
    <= 5e-4 / 1000 = 5e-7. Factories: ceil(11 * 1000 / 1010) = 11, each 11
    patches: 11 * 11 * 338 = 40,898 qubits. Runtime: 1010 * 13 * 1 us = 13.13 ms.
    """
    out = resources.estimate({"logical_qubits": 10, "t_gates": 1000, "rotations": 0, "measurements": 10},
                             physical_error=1e-3, budget=1e-3, cycle_time_s=1e-6)
    assert out["logical_qubits"] == 30
    assert out["logical_cycles"] == 1010
    assert out["code_distance"] == 13
    assert out["physical_qubits_algorithm"] == 10_140
    assert out["factory_levels"] == 1
    assert out["factory_output_error"] == pytest.approx(3.98e-7)
    assert out["t_factories"] == 11
    assert out["physical_qubits_factories"] == 40_898
    assert out["physical_qubits"] == 51_038
    assert out["runtime_s"] == pytest.approx(0.01313)
    assert out["assumptions"] and any("2211.07629" in c for c in out["citations"])


def test_estimate_grows_with_work_and_shrinks_with_better_hardware():
    base = {"logical_qubits": 50, "t_gates": 10 ** 6, "rotations": 100, "measurements": 50}
    small = resources.estimate(base)
    bigger = resources.estimate({**base, "t_gates": 10 ** 9})
    better = resources.estimate(base, physical_error=1e-4)
    assert bigger["code_distance"] >= small["code_distance"]
    assert bigger["runtime_s"] > small["runtime_s"]
    assert better["code_distance"] < small["code_distance"]
    assert better["physical_qubits"] < small["physical_qubits"]


def test_t_counting_on_small_circuits():
    counts = resources.logical_counts(QASM, rotation_error=1e-3)
    # t, tdg and rz(pi/4) are T gates; rz(0.3) is synthesized; s, h, cx (H CZ H) are Clifford.
    assert counts["t_gates"] == 3
    assert counts["rotations"] == 1
    assert counts["t_per_rotation"] == math.ceil(3 * math.log2(1e3)) == 30
    assert counts["t_count"] == 33
    assert counts["cliffords"] == 5
    assert counts["measurements"] == 2
    assert counts["logical_qubits"] == 2
    # The budget's synthesis share sets the precision when none is given: 1e-3 / 3 over one rotation.
    assert resources.logical_counts(QASM)["rotation_error"] == pytest.approx(1e-3 / 3)


def test_t_count_agrees_with_tetron_compiler():
    chip = devices.get("tetron-array")
    precision = chip.parameters["topological"]["synthesis_precision"]
    compiled = compile_circuit(QASM, chip)
    magic = compiled.report["noise_budget"]["totals"]["magic_states"]
    assert resources.logical_counts(QASM, rotation_error=precision)["t_count"] == magic


def test_tetron_device_sets_error_and_cycle_time():
    chip = devices.get("tetron-array")
    topo = chip.parameters["topological"]
    out = resources.estimate(QASM, device=chip)
    assert out["physical_error"] == max(topo["measurement_error_z"], topo["measurement_error_x"])
    assert out["cycle_time_s"] == resources.TETRON_MEASUREMENTS_PER_ROUND * topo["measurement_time_s"]
    assert any("Majorana" in a for a in out["assumptions"])
    with pytest.raises(ValueError, match="threshold"):
        resources.estimate(QASM, device=devices.get("microsoft_majorana_2"))   # 16% X-measurement error
    with pytest.raises(ValueError, match="topological"):
        resources.estimate(QASM, device=devices.get("heavy-hex-127"))
