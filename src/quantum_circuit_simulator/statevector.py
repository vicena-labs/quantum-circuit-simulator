"""State simulation: exact states, density matrices, noisy trajectories and sampling.

Pauli propagation answers expectation values; some questions need the state
itself or measured bitstrings. This module simulates a `Circuit`
(or anything `convert` accepts) directly:

- `statevector` returns the amplitudes of a noise-free circuit (up to 28 qubits).
- `StateVectorPlan` evaluates observables exactly with adjoint gradients on
  noise-free circuits, and on noisy ones through the density matrix (up to 12
  qubits). It is the reference the router checks Pauli propagation against.
- `sample` draws measured bitstrings, noise included: exactly from the density
  matrix when it fits, otherwise one quantum trajectory per shot (noise
  channels unravelled into Kraus operators), with readout errors applied as
  classical bit flips. Clifford circuits beyond the state-vector limit go to
  Stim when it is installed.

Wire 0 is the least significant bit of every basis index and bitstring
integer; bitstrings are printed with wire 0 rightmost, as in Qiskit.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import reduce

import numpy as np

from ._core._reference import GATE1, GATE2, X, Y, Z, fsim, rotation, rzz

MAX_STATEVECTOR_QUBITS = 28
MAX_DENSITY_QUBITS = 12
AUTO_DENSITY_QUBITS = 10   # above this, trajectories are cheaper than a 4^n density matrix
PAULI = {"I": np.eye(2, dtype=complex), "X": X, "Y": Y, "Z": Z}
NOISE = {"DEPOLARIZE", "X_ERROR", "Y_ERROR", "Z_ERROR", "T1_T2"}


# ---------------------------------------------------------------- helpers shared with Hamiltonians
def pauli_matrix(pauli, n):
    """Dense matrix of a Pauli word {wire: axis} on n wires (wire 0 least significant)."""
    return reduce(np.kron, [PAULI[pauli.get(q, "I")] for q in reversed(range(n))], np.eye(1, dtype=complex))


def apply_pauli_sum(terms, vector, n):
    """H |v> for H a list of (coefficient, {wire: axis}), without forming a matrix."""
    state = vector.reshape((2,) * n)
    out = np.zeros_like(state)
    for c, pauli in terms:
        out += c * _apply_word(state, pauli, n)
    return out.reshape(-1)


def _apply_word(state, pauli, n):
    out = state
    for q, axis in pauli.items():
        out = _apply1(out, PAULI[axis], q, n)
    return out


# Wire q is axis -(q + 1), counted from the end, so leading batch axes (one per
# trajectory) need no special case.
def _apply1(state, u, q, n):
    axis = -(q + 1)
    return np.moveaxis(np.tensordot(u, state, axes=([1], [axis])), 0, axis)


def _apply2(state, u, a, b, n):
    """Apply a 4x4 unitary whose index is bit(a) + 2 bit(b)."""
    ua, ub = -(a + 1), -(b + 1)
    t = u.reshape(2, 2, 2, 2)        # [b_out, a_out, b_in, a_in]
    out = np.tensordot(t, state, axes=([2, 3], [ub, ua]))
    return np.moveaxis(out, [0, 1], [ub, ua])


# ---------------------------------------------------------------- stages as operators
def _unitary(stage):
    """The unitary of a gate stage at its bound angle, and its derivative in the shift."""
    name = stage.name
    if name in GATE1:
        return GATE1[name], None
    if name in GATE2:
        return GATE2[name], None
    if name in ("RX", "RY", "RZ"):
        u, du = rotation(name[1], stage.angle)
        return u, du * stage.scale
    if name == "RZZ":
        u, du = rzz(stage.angle)
        return u, du * stage.scale
    if name == "FSIM":
        u, du, _ = fsim(stage.angle, stage.phi or 0.0)
        return u, du * stage.scale
    if name == "UNITARY":
        return stage.data[0], None
    raise ValueError(f"stage {name} is not a gate")


def _kraus(stage):
    """Kraus operators of a noise stage as (probability-weighted) lists."""
    name = stage.name
    if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        p = stage.data[0]
        return [math.sqrt(1 - p) * PAULI["I"], math.sqrt(p) * PAULI[name[0]]]
    if name == "DEPOLARIZE":
        p = stage.data[0]
        k = len(stage.wires)
        words = [w for w in _words(k) if any(a != "I" for a in w)]
        ops = [math.sqrt(1 - p) * np.eye(2 ** k, dtype=complex)]
        ops += [math.sqrt(p / len(words)) * pauli_matrix(dict(enumerate(w)), k) for w in words]
        return ops
    if name == "T1_T2":
        t, t1, t2 = stage.data
        gamma = 1 - math.exp(-t / t1)
        # Amplitude damping leaves coherence sqrt(1 - gamma); extra dephasing brings it to exp(-t/T2).
        coherence = math.exp(-t / t2) / math.sqrt(1 - gamma) if gamma < 1 else 0.0
        pz = (1 - min(1.0, coherence)) / 2
        damp = [np.array([[1, 0], [0, math.sqrt(1 - gamma)]], complex), np.array([[0, math.sqrt(gamma)], [0, 0]], complex)]
        return [math.sqrt(1 - pz) * k for k in damp] + [math.sqrt(pz) * (Z @ k) for k in damp]
    raise ValueError(f"stage {name} is not a supported channel")


def _words(k):
    return [(a,) if k == 1 else (a, b) for a in "IXYZ" for b in ("IXYZ" if k == 2 else "I")][: 4 ** k]


def _bound(circuit, shifts):
    impl = circuit._impl if hasattr(circuit, "_impl") else circuit
    if shifts is None:
        shifts = [0.0] * impl.nparams
    return impl.shifted(list(shifts)) if impl.nparams else impl


def is_noisy(circuit):
    impl = circuit._impl if hasattr(circuit, "_impl") else circuit
    return any(s.name in NOISE for s in impl.stages)


def _gate_stages(impl):
    for stage in impl.stages:
        if stage.name in NOISE:
            raise ValueError("the circuit has noise channels; use the density matrix or trajectories")
        yield stage


# ---------------------------------------------------------------- pure states
def statevector(circuit, shifts=None, values=None):
    """Amplitudes of a noise-free circuit from |0..0>, as a flat array (wire 0 least significant)."""
    circuit, shifts = _resolve(circuit, shifts, values)
    impl = _bound(circuit, shifts)
    n = impl.nqubits
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"{n} qubits exceed the state-vector limit of {MAX_STATEVECTOR_QUBITS}")
    return _evolve(_zero(n), _gate_stages(impl), n).reshape(-1)


def _resolve(circuit, shifts, values):
    """A `Circuit` and shift vector from any circuit form and its parameter values."""
    if hasattr(circuit, "_impl"):
        return circuit, shifts
    from .interop import convert
    converted = convert(circuit, values=values)
    if converted.parameters and shifts is None:
        shifts = converted.shifts(values)
    return converted.circuit, shifts


def _zero(n):
    state = np.zeros((2,) * n, complex)
    state[(0,) * n] = 1.0
    return state


def _evolve(state, stages, n):
    for stage in stages:
        state = _apply_gate(state, stage, _unitary(stage)[0], n)
    return state


def _apply_gate(state, stage, u, n):
    if len(stage.wires) == 1:
        return _apply1(state, u, stage.wires[0], n)
    return _apply2(state, u, stage.wires[0], stage.wires[1], n)


class StateVectorPlan:
    """Exact observables of a circuit by state simulation, with adjoint gradients when noise-free.

    Noise-free circuits use the state vector (up to 28 qubits); noisy ones the
    density matrix (up to 12 qubits). Readout errors act on the observable,
    as in the Pauli solvers.
    """

    def __init__(self, circuit, terms):
        from .interop import readout_expand
        self.circuit = circuit
        impl = circuit._impl
        self.n = impl.nqubits
        self.noisy = is_noisy(circuit)
        limit = MAX_DENSITY_QUBITS if self.noisy else MAX_STATEVECTOR_QUBITS
        if self.n > limit:
            kind = "density-matrix" if self.noisy else "state-vector"
            raise ValueError(f"{self.n} qubits exceed the {kind} limit of {limit}")
        self.terms = readout_expand(list(terms), dict(impl.readout))

    def evaluate(self, shifts=None, *, gradients=True):
        impl = _bound(self.circuit, shifts)
        if self.noisy:
            rho = density_matrix_of(impl)
            value = sum(c * _expect_rho(rho, p, self.n) for c, p in self.terms)
            if gradients and impl.nparams:
                raise ValueError("the state-vector backend gives gradients of noise-free circuits only")
            return float(value), np.zeros(0)
        stages = list(_gate_stages(impl))
        psi = _evolve(_zero(self.n), stages, self.n)
        h_psi = sum(c * _apply_word(psi, p, self.n) for c, p in self.terms)
        value = float(np.vdot(psi, h_psi).real)
        gradient = np.zeros(impl.nparams)
        if gradients and impl.nparams:
            # Adjoint differentiation: walk back, keeping |psi_k> and <lambda_k| = <psi| H U_N .. U_k+1.
            lam = h_psi
            for stage in reversed(stages):
                u, du = _unitary(stage)
                psi = _apply_gate(psi, stage, u.conj().T, self.n)
                if stage.param >= 0:
                    tangent = _apply_gate(psi, stage, du, self.n)
                    gradient[stage.param] += 2 * np.vdot(lam, tangent).real
                lam = _apply_gate(lam, stage, u.conj().T, self.n)
        return value, gradient


def _expect_rho(rho, pauli, n):
    """Tr(P rho): P acts on the ket wires, which are wires n..2n-1 of the 2n-wire tensor."""
    shifted = {q + n: a for q, a in pauli.items()}
    return np.trace(_apply_word(rho.reshape((2,) * (2 * n)), shifted, 2 * n).reshape(2 ** n, 2 ** n)).real


def density_matrix(circuit, shifts=None, values=None):
    """The density matrix of a (noisy) circuit from |0..0>, up to 12 qubits."""
    circuit, shifts = _resolve(circuit, shifts, values)
    return density_matrix_of(_bound(circuit, shifts))


def density_matrix_of(impl):
    n = impl.nqubits
    if n > MAX_DENSITY_QUBITS:
        raise ValueError(f"{n} qubits exceed the density-matrix limit of {MAX_DENSITY_QUBITS}")
    # rho as a (2,)*2n tensor: ket wires on axes 0..n-1, bra wires on n..2n-1.
    rho = np.zeros((2,) * (2 * n), complex)
    rho[(0,) * (2 * n)] = 1.0
    for stage in impl.stages:
        ops = _kraus(stage) if stage.name in NOISE else [_unitary(stage)[0]]
        rho = sum(_apply_both(rho, stage, k, n) for k in ops)
    return rho.reshape(2 ** n, 2 ** n)


def _apply_both(rho, stage, k, n):
    """K rho K^dagger on the (2,)*2n tensor."""
    ket = _apply_gate(rho, _Shift(stage, n), k, 2 * n)            # ket axes are the high wires
    return _apply_gate(ket, stage, k.conj(), 2 * n)                # bra axes: K* on the low wires


class _Shift:
    """A stage whose wires address the ket half of a 2n-wire density tensor."""
    def __init__(self, stage, n):
        self.wires = tuple(q + n for q in stage.wires)


# ---------------------------------------------------------------- sampling
def sample(circuit, shots=1024, *, shifts=None, values=None, seed=None, method="auto", max_trajectories=None):
    """Measured bitstrings of every wire, noise and readout errors included.

    Returns a dict with `counts` ({bitstring: count}, wire 0 rightmost),
    `method` and `exact_distribution`. method "auto" picks the state vector for
    noise-free circuits up to 28 qubits, the density matrix for noisy ones up
    to 10 qubits, trajectories for noisy ones up to 28 qubits, and Stim for
    larger Clifford circuits. Trajectories draw one noise realisation per shot
    unless `max_trajectories` caps them, in which case each realisation serves
    several shots (unbiased, but correlated).
    """
    circuit, shifts = _resolve(circuit, shifts, values)
    impl = _bound(circuit, shifts)
    n = impl.nqubits
    rng = np.random.default_rng(seed)
    shots = int(shots)
    if shots < 1:
        raise ValueError("shots must be positive")
    noisy = is_noisy(circuit)
    if method == "auto":
        if n <= MAX_STATEVECTOR_QUBITS and not noisy:
            method = "statevector"
        elif noisy and n <= AUTO_DENSITY_QUBITS:
            method = "density_matrix"
        elif noisy and n <= MAX_STATEVECTOR_QUBITS:
            method = "trajectories"
        else:
            method = "stim"
    if method == "statevector":
        probabilities = np.abs(statevector(circuit, shifts)) ** 2
        outcomes = rng.choice(2 ** n, size=shots, p=probabilities / probabilities.sum())
        exact = True
    elif method == "density_matrix":
        probabilities = np.clip(np.diag(density_matrix_of(impl)).real, 0, None)
        outcomes = rng.choice(2 ** n, size=shots, p=probabilities / probabilities.sum())
        exact = True
    elif method == "trajectories":
        outcomes, exact = _trajectories(impl, shots, rng, max_trajectories), max_trajectories is None
    elif method == "stim":
        from .stabilizer import sample_clifford
        outcomes, exact = sample_clifford(impl, shots, rng)
    else:
        raise ValueError("method must be auto, statevector, density_matrix, trajectories or stim")
    outcomes = _readout_flips(np.asarray(outcomes, dtype=object if n > 62 else np.int64), impl.readout, n, rng)
    counts = Counter(format(int(o), f"0{n}b") for o in outcomes)
    return {"counts": dict(counts.most_common()), "shots": shots, "n_qubits": n, "method": method,
            "exact_distribution": exact,
            "note": "bitstrings print wire 0 rightmost; readout errors applied as classical flips"}


def _trajectories(impl, shots, rng, max_trajectories):
    """Outcomes from independent quantum trajectories, simulated together as one batch."""
    n = impl.nqubits
    runs = shots if max_trajectories is None else max(1, min(shots, int(max_trajectories)))
    outcomes = []
    # Batches bound memory: each trajectory holds 2^n amplitudes.
    batch = max(1, min(runs, (1 << 24) >> n))
    for first in range(0, runs, batch):
        size = min(batch, runs - first)
        state = np.zeros((size,) + (2,) * n, complex)
        state[(slice(None),) + (0,) * n] = 1.0
        for stage in impl.stages:
            if stage.name in NOISE:
                state = _jump_batch(state, stage, _kraus(stage), n, rng)
            else:
                state = _apply_gate(state, stage, _unitary(stage)[0], n)
        p = np.abs(state.reshape(size, -1)) ** 2
        p /= p.sum(axis=1, keepdims=True)
        # One outcome per trajectory by inverse-CDF sampling, row by row.
        draws = (p.cumsum(axis=1) < rng.random((size, 1))).sum(axis=1)
        outcomes.extend(np.minimum(draws, 2 ** n - 1))
    if runs < shots:
        outcomes = list(rng.choice(np.asarray(outcomes), size=shots, replace=True))
    return outcomes


def _jump_batch(state, stage, ops, n, rng):
    """Apply one Kraus operator per trajectory, each chosen with its Born probability."""
    candidates = np.stack([_apply_gate(state, stage, k, n) for k in ops])     # (ops, batch, ...)
    weights = (np.abs(candidates) ** 2).reshape(len(ops), candidates.shape[1], -1).sum(axis=2)
    cumulative = weights.cumsum(axis=0) / weights.sum(axis=0)
    pick = (cumulative < rng.random(weights.shape[1])).sum(axis=0)
    pick = np.minimum(pick, len(ops) - 1)
    chosen = candidates[pick, np.arange(weights.shape[1])]
    norm = np.sqrt(weights[pick, np.arange(weights.shape[1])])
    return chosen / norm.reshape((-1,) + (1,) * n)


def _readout_flips(outcomes, readout, n, rng):
    """Flip each wire's bit with P(1|0) = p01 and P(0|1) = p10."""
    if not readout:
        return outcomes
    if n > 62:
        out = []
        for o in outcomes:
            o = int(o)
            for q, (p01, p10) in readout.items():
                if rng.random() < (p10 if (o >> q) & 1 else p01):
                    o ^= 1 << q
            out.append(o)
        return out
    out = np.asarray(outcomes, dtype=np.int64).copy()
    for q, (p01, p10) in readout.items():
        bit = (out >> q) & 1
        flip = rng.random(out.size) < np.where(bit == 1, p10, p01)
        out ^= flip.astype(np.int64) << q
    return out


def expectation_from_counts(counts, pauli):
    """Estimate <Z..Z> on the wires of `pauli` (all Z) from counts, with its standard error."""
    if any(a != "Z" for a in pauli.values()):
        raise ValueError("counts measure Z; rotate X/Y wires into the Z basis before sampling")
    total = sum(counts.values())
    mean = sum((-1) ** sum(int(b[-1 - q]) for q in pauli) * c for b, c in counts.items()) / total
    return mean, math.sqrt(max(0.0, 1 - mean ** 2) / total)


def sample_compiled(compiled, shots=1024, *, values=None, seed=None, method="auto"):
    """Sample a circuit compiled onto a device, as measured on its logical qubits.

    Only the physical qubits the circuit touches are simulated, so a 20-qubit
    circuit on a 127-qubit chip is a 20-qubit simulation; bitstrings follow the
    logical qubits (wire 0 rightmost), read from where routing left each one,
    with the device's readout errors.
    """
    from dataclasses import replace
    from .api import Circuit
    impl = compiled.circuit._impl
    active = sorted({q for s in impl.stages for q in s.wires} | set(compiled.final_layout))
    local = {q: i for i, q in enumerate(active)}
    small = Circuit(len(active), compiled.circuit.n_parameters, compiled.circuit.max_terms)
    small._impl.stages = [replace(s, wires=tuple(local[q] for q in s.wires)) for s in impl.stages]
    readout = compiled.readout
    for q in compiled.final_layout:
        pair = readout.get(q) if isinstance(readout, dict) else readout
        if pair:
            small.readout(local[q], *pair)
    shifts = compiled.shifts(values) if compiled.parameters else None
    raw = sample(small, shots, shifts=shifts, seed=seed, method=method)
    width = len(active)
    counts = Counter()
    for bits, count in raw["counts"].items():
        logical = "".join(bits[width - 1 - local[compiled.final_layout[q]]]
                          for q in reversed(range(len(compiled.final_layout))))
        counts[logical] += count
    return {**raw, "counts": dict(counts.most_common()), "n_qubits": len(compiled.final_layout),
            "simulated_qubits": width}
