"""Clifford circuits at any size through Stim (optional: pip install "quantum-circuit-simulator[stim]").

Expectation values of Clifford circuits need nothing extra: Pauli propagation
maps each Pauli to a single Pauli, so the native solver is already exact and
polynomial for them. Sampling bitstrings from a Clifford circuit beyond the
state-vector limit is what Stim adds. Each gate is converted through its
stabilizer tableau, so any Clifford unitary works, including rotations by
multiples of pi/2. Pauli noise channels map exactly; T1/T2 relaxation is not a
Pauli channel and is replaced by its Pauli twirl, which keeps every Pauli
expectation's decay but not the drift towards |0>, so such samples are
reported as approximate.
"""
from __future__ import annotations

import math

import numpy as np


def is_clifford(circuit):
    """Whether every gate of a circuit (bound at its current angles) is Clifford."""
    impl = circuit._impl if hasattr(circuit, "_impl") else circuit
    return all(_clifford_gate(s) for s in impl.stages if not _is_noise(s))


def _is_noise(stage):
    from .statevector import NOISE
    return stage.name in NOISE


def _clifford_gate(stage):
    if stage.name in ("RX", "RY", "RZ", "RZZ"):
        return abs(stage.angle / (math.pi / 2) - round(stage.angle / (math.pi / 2))) < 1e-9
    if stage.name == "FSIM":
        return False
    if stage.name == "UNITARY":
        return _tableau(stage.data[0]) is not None
    return True


def sample_clifford(impl, shots, rng):
    """Bitstring integers from a Clifford circuit, and whether the distribution is exact."""
    try:
        import stim
    except ImportError as exc:
        raise ImportError("sampling beyond 28 qubits needs a Clifford circuit and Stim: "
                          'pip install "quantum-circuit-simulator[stim]"') from exc
    circuit, exact = to_stim(impl)
    circuit.append("M", list(range(impl.nqubits)))
    sampler = circuit.compile_sampler(seed=int(rng.integers(2 ** 63)))
    bits = sampler.sample(shots)
    n = impl.nqubits
    if n <= 62:
        return (bits.astype(np.int64) << np.arange(n, dtype=np.int64)).sum(axis=1), exact
    return [sum(int(b) << q for q, b in enumerate(row)) for row in bits], exact


def to_stim(impl):
    """A stim.Circuit for a bound Clifford circuit, and whether its noise is exact (no T1/T2 twirl)."""
    import stim
    from .statevector import _unitary
    out = stim.Circuit()
    exact = True
    for stage in impl.stages:
        w = list(stage.wires)
        if stage.name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            out.append(stage.name, w, stage.data[0])
        elif stage.name == "DEPOLARIZE":
            out.append("DEPOLARIZE1" if len(w) == 1 else "DEPOLARIZE2", w, stage.data[0])
        elif stage.name == "T1_T2":
            t, t1, t2 = stage.data
            # Pauli twirl of relaxation: px = py = (1 - e^{-t/T1}) / 4, pz from the T2 decay.
            px = (1 - math.exp(-t / t1)) / 4
            pz = (1 - math.exp(-t / t2)) / 2 - px
            out.append("PAULI_CHANNEL_1", w, [px, px, max(0.0, pz)])
            exact = False
        else:
            if not _clifford_gate(stage):
                raise ValueError(f"{stage.name} at angle {stage.angle} is not Clifford; Stim cannot simulate it")
            piece = _tableau(_unitary(stage)[0])
            for instruction in piece:
                out.append(instruction.name, [w[t.value] for t in instruction.targets_copy()],
                           instruction.gate_args_copy())
    return out, exact


_CACHE = {}


def _tableau(u):
    """The Clifford circuit of a 2x2 or 4x4 unitary on local qubits 0 (and 1), or None if not Clifford."""
    import stim
    key = np.round(np.asarray(u, complex), 12).tobytes()
    if key not in _CACHE:
        try:
            _CACHE[key] = stim.Tableau.from_unitary_matrix(np.asarray(u, complex), endian="little").to_circuit()
        except ValueError:
            _CACHE[key] = None
    return _CACHE[key]
