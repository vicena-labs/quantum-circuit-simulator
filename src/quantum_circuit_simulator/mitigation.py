"""Error mitigation on a known noise model: noise inversion (PNA), PEC and readout correction.

    from quantum_circuit_simulator import devices, mitigation as mt
    from quantum_circuit_simulator.compiler import compile_circuit
    compiled = compile_circuit(qasm, devices.get("ibm_fez"))
    mt.pna(compiled, "Z0*Z1")                  # noise absorbed exactly: the ideal value
    mt.pec_overhead(compiled)                   # sampling cost gamma and the shot multiplier gamma^2
    mt.pec(compiled, "Z0*Z1", samples=200, seed=1)

Conventions. A noise stage is a DEPOLARIZE, X/Y/Z_ERROR or T1_T2 stage of a
`Circuit` (readout is classical and handled separately). Every stage holds
its dual Pauli transfer matrix in label order I, X, Z, Y (two wires:
index = label(first wire) + 4 label(second wire)), so a Pauli channel has a
diagonal matrix of Pauli fidelities f_P. `scale` multiplies the learned
error rates (probabilities, and the relaxation duration, so 1/T1 and 1/T2):
scale = 1 is a perfectly learned model, anything else a mis-learned one.

`target` arguments take a `Circuit` (its own readout errors apply) or a
`Compiled` (observables use logical qubits; the device's readout applies);
`values` are the Compiled's parameter values, or a Circuit's shift vector.

- `invert_noise` follows every noise stage with its exact inverse, a
  non-physical transfer matrix only Pauli propagation can evaluate.
  `pna` evaluates an observable on it: propagated-noise absorption
  (qiskit-addon-pna) with the model known exactly.
- `pec` is probabilistic error cancellation for Pauli channels (van den Berg
  et al., arXiv:2201.09866): each inverse channel is a quasi-probability
  mixture of Paulis whose 1-norm gamma sets the sampling overhead.
- `readout_correct` and `mitigated_counts` invert the per-qubit confusion
  matrices exactly (TREX when readout is symmetrised).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ._core._reference import Stage, depolarizing_ptm, relaxation_ptm, transfer
from .api import Circuit
from .interop import parse_observable, readout_expand
from .statevector import NOISE, PAULI

PAULI_CHANNELS = {"DEPOLARIZE", "X_ERROR", "Y_ERROR", "Z_ERROR"}
LABELS = "IXZY"   # transfer-matrix label order: bit x + 2 bit z


# ---------------------------------------------------------------- noise inversion and PNA
def invert_noise(circuit, *, scale=1.0):
    """A copy of `circuit` with each noise stage followed by the inverse of its (scaled) model.

    The inverse stage is named INVERSE_<name> and holds the exact inverse
    transfer matrix of the stage's channel with its error rates multiplied by
    `scale`; with scale = 1 each pair is the identity map. The circuit keeps
    its readout errors. Only Pauli propagation evaluates it: the inverse maps
    are not quantum channels, so state simulators refuse them.
    """
    circuit = _circuit(circuit)
    stages = []
    for stage in circuit._impl.stages:
        stages.append(stage)
        if stage.name in NOISE:
            stages.append(_inverse_stage(stage, scale))
    return _copy(circuit, stages, dict(circuit._impl.readout))


def _circuit(target):
    """The `Circuit` of a Circuit or a Compiled."""
    circuit = getattr(target, "circuit", target)
    if not isinstance(circuit, Circuit):
        raise TypeError("expected a Circuit or a Compiled circuit")
    return circuit


def _inverse_stage(stage, scale):
    matrix = np.linalg.inv(_noise_ptm(stage, scale))
    return Stage(stage.wires, matrix, np.zeros_like(matrix), name="INVERSE_" + stage.name)


def _noise_ptm(stage, scale):
    """The transfer matrix of a noise stage with its error rates multiplied by scale."""
    if scale == 1.0:
        return stage.matrix
    if not (math.isfinite(scale) and scale >= 0):
        raise ValueError("scale must be a finite, nonnegative number")
    if stage.name == "DEPOLARIZE":
        return depolarizing_ptm(min(1.0, stage.data[0] * scale), len(stage.wires))
    if stage.name == "T1_T2":
        duration, t1, t2 = stage.data
        return relaxation_ptm(duration * scale, t1, t2)
    p = min(1.0, stage.data[0] * scale)
    return (1 - p) * np.eye(4) + p * transfer(PAULI[stage.name[0]])[0]


def _copy(circuit, stages, readout):
    out = Circuit(circuit.n_qubits, circuit.n_parameters, circuit.max_terms)
    out._impl.stages = list(stages)
    out._impl.readout = readout
    return out


def pna(target, observable, *, scale=1.0, readout=True, truncation=None, values=None):
    """Propagated-noise absorption: <observable> on `invert_noise(target, scale=scale)`.

    readout=True also inverts the readout model (with the same `scale`) on the
    observable, as `readout_correct` does, so scale = 1 returns the noise-free
    value; readout=False leaves readout error in both values. Returns value,
    unmitigated (the noisy circuit as it is), and with a `Truncation` the
    solver's error_bound and whether it is rigorous: it assumes every
    discarded term has |<P>| <= 1 on the partial circuit, which holds when
    each noise-and-inverse pair is a weaker physical channel, scale <= 1.
    """
    circuit, terms, readout_map = _resolve(target, observable)
    shifts = _shifts(target, values)
    mitigated_terms = _inverse_readout(terms, readout_map, scale) if readout else terms
    noisy = _evaluate(_copy(circuit, circuit._impl.stages, {}), readout_expand(terms, readout_map), shifts, None)
    stages = invert_noise(circuit, scale=scale)._impl.stages
    result = _evaluate(_copy(circuit, stages, {}), readout_expand(mitigated_terms, readout_map), shifts, truncation)
    out = {"value": result.value, "unmitigated": noisy.value, "scale": scale, "readout_mitigated": bool(readout),
           "noise_stages": sum(s.name in NOISE for s in circuit._impl.stages)}
    if truncation is not None:
        out.update(error_bound=result.error_bound, bound_rigorous=scale <= 1.0)
    return out


def _resolve(target, observable):
    """(Circuit, observable terms on its wires before readout, {wire: (p01, p10)} readout map)."""
    if isinstance(target, Circuit):
        return target, parse_observable(observable, target.n_qubits), dict(target._impl.readout)
    circuit = _circuit(target)
    layout = target.final_layout
    terms = [(c, {layout[q]: a for q, a in p.items()}) for c, p in parse_observable(observable, len(layout))]
    return circuit, terms, _readout_map(target.readout, {q for _, p in terms for q in p})


def _readout_map(readout, wires):
    """A {wire: (p01, p10)} map from a pair for every wire, a mapping or None."""
    if not readout:
        return {}
    if isinstance(readout, dict):
        return {q: tuple(readout[q]) for q in wires if readout.get(q)}
    return {q: tuple(readout) for q in wires}


def _shifts(target, values):
    if isinstance(target, Circuit):
        return values
    return target.shifts(values) if target.parameters else None


def _evaluate(circuit, terms, shifts, truncation):
    if not terms:
        from .api import Evaluation
        return Evaluation(0.0, (), "none", 0.0 if truncation else None)
    return circuit.prepare(terms, truncation=truncation).evaluate(shifts, gradients=False)


def _inverse_readout(terms, readout_map, scale=1.0):
    """Terms O' with readout(O') = O for the readout model scaled by `scale`.

    Readout maps a measured factor P to alpha I + beta P (alpha = p10 - p01,
    beta = 1 - p01 - p10), so its inverse maps P to (P - alpha) / beta.
    """
    merged = {}
    for coefficient, pauli in terms:
        expanded = [(coefficient, {})]
        for q, axis in sorted(pauli.items()):
            if q not in readout_map:
                expanded = [(c, {**p, q: axis}) for c, p in expanded]
                continue
            p01, p10 = (min(1.0, x * scale) for x in readout_map[q])
            alpha, beta = p10 - p01, 1 - p01 - p10
            expanded = ([(-c * alpha / beta, p) for c, p in expanded if alpha]
                        + [(c / beta, {**p, q: axis}) for c, p in expanded])
        for c, p in expanded:
            key = tuple(sorted(p.items()))
            merged[key] = merged.get(key, 0.0) + c
    return [(c, dict(k)) for k, c in merged.items() if c != 0.0]


# ---------------------------------------------------------------- probabilistic error cancellation
def pec_overhead(target):
    """The PEC sampling overhead of a circuit's noise: gamma and the shot multiplier gamma^2.

    gamma is the product over noise stages of the 1-norm of the inverse
    channel's quasi-probabilities. A Pauli channel with fidelities f_P has
    inverse fidelities 1/f_P, and its quasi-probability of Pauli Q is
    q_Q = 4^-k sum_P s(P, Q) / f_P, with s = +1 when P and Q commute and -1
    otherwise; for one-qubit depolarizing, gamma = (3/f - 1) / 2. Relaxation
    stages are Pauli-twirled first (off-diagonal transfer entries dropped);
    `twirled_stages` counts them. A gamma of e^lambda multiplies the shots
    needed for a given error by e^(2 lambda).
    """
    channels = [_pec_channel(s) for s in _circuit(target)._impl.stages if s.name in NOISE]
    log_gamma = math.fsum(math.log(np.abs(q).sum()) for _, q in channels)
    return {"gamma": math.exp(log_gamma), "shot_multiplier": math.exp(2 * log_gamma), "noise_stages": len(channels),
            "twirled_stages": sum(not _is_pauli_channel(s) for s in _circuit(target)._impl.stages if s.name in NOISE)}


def _pec_channel(stage):
    """(twirled Pauli fidelities, inverse quasi-probabilities) of a noise stage, both in label order."""
    fidelities = np.diag(stage.matrix).copy()
    if np.any(fidelities <= 0):
        raise ValueError(f"{stage.name} on {stage.wires} has a zero Pauli fidelity; it has no inverse")
    return fidelities, _commutation_signs(len(stage.wires)) @ (1 / fidelities) / len(fidelities)


def _is_pauli_channel(stage):
    return np.allclose(stage.matrix, np.diag(np.diag(stage.matrix)), atol=1e-15)


def _commutation_signs(k):
    """S[Q, P] = +1 if Paulis Q and P (label-order indices on k wires) commute, else -1."""
    n = 4 ** k
    signs = np.empty((n, n))
    for a in range(n):
        for b in range(n):
            anticommuting = 0
            for h in range(k):
                pa, pb = (a >> 2 * h) & 3, (b >> 2 * h) & 3
                anticommuting += ((pa & 1) * (pb >> 1) + (pa >> 1) * (pb & 1)) & 1
            signs[a, b] = -1.0 if anticommuting % 2 else 1.0
    return signs


def pec(target, observable, samples, seed=None, *, values=None, readout=True):
    """Probabilistic error cancellation of a circuit's Pauli noise, by Monte Carlo over corrections.

    Each sample draws, after every noise stage, a Pauli Q with probability
    |q_Q| / gamma_stage and sign sgn(q_Q), evaluates that circuit exactly by
    Pauli propagation and contributes gamma * (product of signs) * value. The
    mean is an unbiased estimate of the noise-free value of the twirled
    circuit; relaxation stages are replaced by their Pauli twirl (as on
    hardware that twirls its gates), counted in `twirled_stages`. readout=True
    inverts readout on the observable exactly. Returns estimate, standard
    error (sample standard deviation / sqrt(samples)), gamma and the
    unmitigated (twirled, noisy) value.
    """
    if samples < 2:
        raise ValueError("pec needs at least 2 samples for a standard error")
    circuit, terms, readout_map = _resolve(target, observable)
    shifts = _shifts(target, values)
    noisy_terms = readout_expand(terms, readout_map)
    terms = readout_expand(_inverse_readout(terms, readout_map) if readout else terms, readout_map)
    rng = np.random.default_rng(seed)
    stages = [replace(s, matrix=np.diag(np.diag(s.matrix))) if s.name in NOISE else s
              for s in circuit._impl.stages]
    noise = [i for i, s in enumerate(stages) if s.name in NOISE]
    channels = [_pec_channel(stages[i]) for i in noise]
    gamma = math.exp(math.fsum(math.log(np.abs(q).sum()) for _, q in channels))
    signs = {k: _commutation_signs(k) for k in (1, 2)}
    draws = []
    for _ in range(samples):
        sampled, sign = list(stages), 1.0
        for i, (_, q) in zip(noise, channels):
            pick = rng.choice(len(q), p=np.abs(q) / np.abs(q).sum())
            sign *= math.copysign(1.0, q[pick])
            # Following the channel with Pauli Q multiplies each fidelity by s(P, Q).
            sampled[i] = replace(stages[i], matrix=stages[i].matrix * signs[len(stages[i].wires)][pick])
        draws.append(gamma * sign * _evaluate(_copy(circuit, sampled, {}), terms, shifts, None).value)
    draws = np.array(draws)
    unmitigated = _evaluate(_copy(circuit, stages, {}), noisy_terms, shifts, None).value
    return {"estimate": float(draws.mean()), "standard_error": float(draws.std(ddof=1) / math.sqrt(samples)),
            "gamma": gamma, "samples": int(samples), "unmitigated": unmitigated,
            "twirled_stages": sum(not _is_pauli_channel(circuit._impl.stages[i]) for i in noise),
            "readout_mitigated": bool(readout)}


# ---------------------------------------------------------------- readout mitigation
def readout_correct(target, observable, measured):
    """Readout-corrected <observable> for a product of Z factors, from what was measured.

    target is a Compiled, a Circuit, or a {wire: (p01, p10)} map, where p01 =
    P(read 1 | state 0) and p10 = P(read 0 | state 1); wires are the logical
    qubits a counts dictionary reports. Per wire the measured value is
    z' = alpha + beta z with alpha = p10 - p01 and beta = 1 - p01 - p10, so the
    exact inverse is z = (z' - alpha) / beta. `measured` may be:

    - counts {bitstring: count} (wire 0 rightmost): the shot average of
      prod_q (z'_q - alpha_q) / beta_q, unbiased for any asymmetry; returns
      value and standard_error;
    - {tuple of wires: measured <prod Z>} for every subset of the observable's
      wires (the empty one is 1): the exact expansion
      prod_q (Z'_q - alpha_q) / beta_q, term by term;
    - a single measured value: valid only for symmetrised (twirled) readout,
      alpha = 0, as in TREX: value / prod_q beta_q, with each beta taken as
      1 - p01 - p10 of the device.
    """
    words = parse_observable(observable)
    if len(words) != 1 or any(a != "Z" for a in words[0][1].values()):
        raise ValueError("readout_correct takes one Pauli-Z product, such as 'Z0*Z1'")
    coefficient, pauli = words[0]
    wires = sorted(pauli)
    readout = _logical_readout(target, wires)
    if isinstance(measured, dict) and all(isinstance(k, str) for k in measured):
        value, error = _corrected_from_counts(measured, wires, readout)
        return {"value": coefficient * value, "standard_error": abs(coefficient) * error, "method": "counts"}
    if isinstance(measured, dict):
        terms = _inverse_readout([(1.0, {q: "Z" for q in wires})], readout)
        values = {tuple(sorted(k)): v for k, v in measured.items()}
        value = math.fsum(c * (values[tuple(sorted(p))] if p else 1.0) for c, p in terms)
        return {"value": coefficient * value, "method": "exact"}
    scale = math.prod(1 - p01 - p10 for p01, p10 in (readout.get(q, (0.0, 0.0)) for q in wires))
    return {"value": coefficient * float(measured) / scale, "method": "trex"}


def _logical_readout(target, wires):
    """{logical wire: (p01, p10)} for the measured wires."""
    if isinstance(target, dict):
        return {q: tuple(target[q]) for q in wires if q in target}
    if isinstance(target, Circuit):
        return {q: target._impl.readout[q] for q in wires if q in target._impl.readout}
    physical = _readout_map(target.readout, {target.final_layout[q] for q in wires})
    return {q: physical[target.final_layout[q]] for q in wires if target.final_layout[q] in physical}


def _corrected_from_counts(counts, wires, readout):
    total = sum(counts.values())
    mean = second = 0.0
    for bits, count in counts.items():
        shot = 1.0
        for q in wires:
            z = 1.0 - 2.0 * int(bits[-1 - q])
            p01, p10 = readout.get(q, (0.0, 0.0))
            shot *= (z - (p10 - p01)) / (1 - p01 - p10)
        mean += count * shot
        second += count * shot * shot
    mean /= total
    return mean, math.sqrt(max(0.0, second / total - mean ** 2) / total)


def mitigated_counts(counts, readout, *, max_qubits=20):
    """Readout-mitigated distribution by tensored confusion-matrix inversion.

    readout is a Compiled, a Circuit or a {wire: (p01, p10)} map (wires as in
    the bitstrings, wire 0 rightmost). The measured distribution is multiplied
    by each wire's inverse confusion matrix,
    A^-1 = [[1 - p10, -p10], [-p01, 1 - p01]] / (1 - p01 - p10), on the full
    2^n vector (n <= max_qubits). Returns `quasi_probabilities`, unbiased and
    possibly negative; `probabilities`, the same with negative entries clipped
    to zero and renormalised to sum 1 (slightly biased); and `negativity`, the
    total negative weight clipped.
    """
    n = len(next(iter(counts)))
    if n > max_qubits:
        raise ValueError(f"{n} qubits exceed max_qubits={max_qubits} for the dense inversion")
    total = sum(counts.values())
    vector = np.zeros((2,) * n)
    for bits, count in counts.items():
        vector[tuple(int(b) for b in bits)] += count / total   # axis 0 is the leftmost bit, wire n-1
    for q, (p01, p10) in _logical_readout(readout, range(n)).items():
        inverse = np.array([[1 - p10, -p10], [-p01, 1 - p01]]) / (1 - p01 - p10)
        axis = n - 1 - q
        vector = np.moveaxis(np.tensordot(inverse, vector, axes=([1], [axis])), 0, axis)
    flat = vector.reshape(-1)
    quasi = {format(i, f"0{n}b"): float(v) for i, v in enumerate(flat) if abs(v) > 1e-15}
    clipped = np.clip(flat, 0.0, None)
    probabilities = {format(i, f"0{n}b"): float(v / clipped.sum()) for i, v in enumerate(clipped) if v > 0}
    return {"quasi_probabilities": quasi, "probabilities": probabilities,
            "negativity": float(-flat[flat < 0].sum()), "shots": int(total)}
