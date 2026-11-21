"""Fault-tolerant resource estimation: logical counts, surface-code distance, physical qubits and runtime.

A transparent model in the spirit of Beverland et al., "Assessing requirements
to scale to practical quantum advantage", arXiv:2211.07629 (the Azure Quantum
Resource Estimator). It is not a copy of their code and does not claim to
reproduce their numbers; every step is stated below and in each estimate's
`assumptions`.

    from quantum_circuit_simulator import resources
    counts = resources.logical_counts(qasm_text)
    resources.estimate(counts, physical_error=1e-3, budget=1e-3)

The model:

1. Error budget. With arbitrary rotations the budget is split equally three
   ways (logical failures, T-state distillation, rotation synthesis), otherwise
   two ways; with no T states at all it goes wholly to logical failures.
2. T counting. T and Tdg are one T each, as is any rotation by an odd
   multiple of pi/4; multiples of pi/2 are Clifford; any other (or
   parameterised) rotation is synthesized with ceil(3 log2(1/eps_rot)) T gates
   (Ross and Selinger, arXiv:1403.2975), eps_rot = synthesis budget / rotations.
   This is the classification `compiler.tetron_cost` applies on tetron chips.
3. Layout. Q algorithm qubits occupy 2Q + ceil(sqrt(8Q)) + 1 surface-code
   patches, the routing overhead Beverland et al. use (after Litinski,
   arXiv:1808.02892).
4. Time. Cliffords are absorbed into Pauli-based computation; each T state and
   each measurement takes one logical cycle of d syndrome rounds (Litinski,
   arXiv:1808.02892), so the algorithm runs in C = T + M logical cycles.
5. Distance. p_L(d) = a (p / p_th)^((d+1)/2) per patch per logical cycle, with
   a = 0.03 and p_th = 0.01 as in Beverland et al.; d is the smallest odd
   distance with (patches x C x p_L(d)) within the logical budget.
6. Factories. A 15-to-1 block uses 11 patches for 11 logical cycles and
   outputs a T state with error 35 e_in^3 plus its own logical failures,
   121 p_L(d) (Bravyi and Kitaev, quant-ph/0403025; layout of Litinski,
   arXiv:1808.02892, Sec. 3.1). Levels are stacked (15 blocks of level l feed
   one of level l+1) until the output error meets the distillation budget per
   T state; injected states have error p, and the distance is raised if the
   factories' own failures exceed that budget. Enough factories run in parallel to
   supply one T state per logical cycle, so distillation never sets the
   runtime. Factory patches use the algorithm's distance.
7. Physical qubits: 2 d^2 per patch (data and measurement qubits of a rotated
   surface code), for the algorithm and the factories.
"""
from __future__ import annotations

import math

CITATIONS = [
    "Beverland et al., Assessing requirements to scale to practical quantum advantage, arXiv:2211.07629",
    "Ross and Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, arXiv:1403.2975",
    "Litinski, A Game of Surface Codes, arXiv:1808.02892 (Quantum 3, 128 (2019))",
    "Bravyi and Kitaev, Universal quantum computation with ideal Clifford gates and noisy ancillas, "
    "quant-ph/0403025",
    "Fowler, Mariantoni, Martinis and Cleland, Surface codes: towards practical large-scale quantum "
    "computation, arXiv:1208.0928",
]
# Each syndrome round of a measurement-based surface code on tetrons is taken as this many sequential
# parity-measurement steps (an assumption; set cycle_time_s from your own protocol to override).
TETRON_MEASUREMENTS_PER_ROUND = 4
FACTORY_PATCHES, FACTORY_CYCLES = 11, 11


# ---------------------------------------------------------------- logical counts
def logical_counts(circuit, *, budget=1e-3, values=None, rotation_error=None):
    """Logical resources of a circuit (OpenQASM, a Program, a Qiskit circuit or a Circuit).

    Counts are of the converter's native gates (CX is H, CZ, H), so `cliffords`
    counts native Clifford operations. `t_gates` are explicit T-type gates,
    `rotations` the arbitrary-angle ones, and `t_count` adds their synthesis
    cost at rotation_error, by default the synthesis share of `budget` split
    over the rotations. depth is the native-gate depth, measurements the
    explicit measure instructions.
    """
    from .compiler import _instructions, tetron_cost
    from .interop import GATE_END, _lower
    n, instructions, _ = _instructions(circuit, values)
    measurements = sum(len(q) for name, q, _ in instructions if name == "measure")
    ops, _ = _lower(instructions)
    t_gates = rotations = cliffords = 0
    level = {}
    for prim, wires, base, index, _ in ops:
        if prim == GATE_END:
            continue
        # tetron_cost: 0 for a Clifford angle, 1 for a T-type angle, more for one needing synthesis.
        cost = tetron_cost(prim, base, index, {"synthesis_precision": 1e-3}) if len(wires) == 1 else 0
        t_gates += cost == 1
        rotations += cost > 1
        cliffords += cost == 0
        depth = max(level.get(w, 0) for w in wires) + 1
        level.update({w: depth for w in wires})
    counts = {"logical_qubits": n, "t_gates": t_gates, "rotations": rotations, "cliffords": cliffords,
              "measurements": measurements, "depth": max(level.values(), default=0)}
    return {**counts, **_synthesis(counts, budget, rotation_error)}


def _synthesis(counts, budget, rotation_error=None):
    """The rotation precision, T gates per rotation and total T count."""
    rotations = counts["rotations"]
    if rotation_error is None:
        rotation_error = _split(budget, counts)["synthesis"] / rotations if rotations else None
    per_rotation = math.ceil(3 * math.log2(1 / rotation_error)) if rotations else 0
    return {"rotation_error": rotation_error, "t_per_rotation": per_rotation,
            "t_count": counts["t_gates"] + rotations * per_rotation}


def _split(budget, counts):
    """The error budget's shares: logical, distillation and synthesis."""
    if counts["rotations"]:
        return {"logical": budget / 3, "distillation": budget / 3, "synthesis": budget / 3}
    if counts["t_gates"]:
        return {"logical": budget / 2, "distillation": budget / 2, "synthesis": 0.0}
    return {"logical": budget, "distillation": 0.0, "synthesis": 0.0}


# ---------------------------------------------------------------- estimate
def estimate(circuit_or_counts, *, physical_error=None, budget=1e-3, code="surface", cycle_time_s=None,
             factory="15-to-1", device=None, prefactor=0.03, threshold=0.01, values=None):
    """Physical resources to run a circuit fault-tolerantly on a surface code, as a dict.

    circuit_or_counts is anything `logical_counts` accepts or a dict with at least
    logical_qubits, t_gates, rotations and measurements (the T count is
    recomputed for this budget).
    physical_error (default 1e-3) is the error per physical operation and
    cycle_time_s (default 1 us) one syndrome round. With a topological tetron
    `device` (devices.get("tetron-array"), "microsoft_majorana_2"), physical
    error is the larger of its Z and X measurement errors and a round is
    TETRON_MEASUREMENTS_PER_ROUND measurement times, unless given explicitly.
    prefactor and threshold are a and p_th of p_L(d). The module docstring
    states the model; the result lists its assumptions and citations.
    """
    if code != "surface" or factory != "15-to-1":
        raise ValueError("only code='surface' with factory='15-to-1' is modelled")
    assumptions = []
    physical_error, cycle_time_s = _hardware(device, physical_error, cycle_time_s, assumptions)
    if physical_error >= threshold:
        raise ValueError(f"physical error {physical_error:g} is not below the threshold {threshold:g}; "
                         "no code distance meets any budget")
    counts = circuit_or_counts if isinstance(circuit_or_counts, dict) else logical_counts(circuit_or_counts,
                                                                                            values=values)
    counts = {**counts, **_synthesis(counts, budget)}
    shares = _split(budget, counts)
    q = counts["logical_qubits"]
    patches = 2 * q + math.ceil(math.sqrt(8 * q)) + 1
    t_states = counts["t_count"]
    cycles = max(t_states + counts["measurements"], 1)

    def logical_error(d):
        return prefactor * (physical_error / threshold) ** ((d + 1) / 2)
    d = code_distance(shares["logical"] / (patches * cycles), physical_error, prefactor, threshold)
    levels, output_error, factories, factory_patches = 0, None, 0, 0
    if t_states:
        logical_d = d
        while (distilled := _distillation(physical_error, logical_error(d), shares["distillation"] / t_states)) is None:
            d += 2   # the factories' own logical failures need a larger distance
        levels, output_error = distilled
        if d > logical_d:
            assumptions.append(f"distance raised from {logical_d} to {d} so the factories' own logical "
                               "failures fit the distillation budget")
        factories = math.ceil(FACTORY_CYCLES * t_states / cycles)
        factory_patches = FACTORY_PATCHES * sum(15 ** level for level in range(levels))
    per_patch = 2 * d * d
    algorithm_qubits, factory_qubits = patches * per_patch, factories * factory_patches * per_patch
    assumptions += [
        f"surface code, p_L(d) = {prefactor} (p/{threshold})^((d+1)/2) per patch per logical cycle",
        f"{q} algorithm qubits in {patches} patches (2Q + ceil(sqrt(8Q)) + 1 routing layout)",
        "Cliffords absorbed into Pauli-based computation; one logical cycle (d rounds) per T state and measurement",
        f"error budget {budget:g} split {shares}",
        "15-to-1 blocks of 11 patches for 11 logical cycles, output error 35 e_in^3 + 121 p_L(d), "
        "injected states at the physical error, factory patches at the algorithm distance",
        "2 d^2 physical qubits per patch",
    ]
    return {"logical_qubits": patches, "algorithm_qubits": q, "code_distance": d,
            "logical_error_per_cycle": logical_error(d), "physical_qubits": algorithm_qubits + factory_qubits,
            "physical_qubits_algorithm": algorithm_qubits, "physical_qubits_factories": factory_qubits,
            "logical_cycles": cycles, "logical_cycle_time_s": d * cycle_time_s,
            "runtime_s": cycles * d * cycle_time_s, "t_states": t_states, "t_factories": factories,
            "factory_levels": levels, "factory_output_error": output_error, "counts": counts,
            "physical_error": physical_error, "cycle_time_s": cycle_time_s, "error_budget": shares,
            "assumptions": assumptions, "citations": CITATIONS}


def _hardware(device, physical_error, cycle_time_s, assumptions):
    """Physical error and syndrome-round time: given, read from a tetron chip, or the defaults."""
    if device is not None:
        topo = device.parameters.get("topological") if device.family == "topological" else None
        if topo is None:
            raise ValueError(f"{device.name} is not a topological chip; pass physical_error and cycle_time_s")
        if physical_error is None:
            physical_error = max(topo["measurement_error_z"], topo["measurement_error_x"])
        if cycle_time_s is None:
            cycle_time_s = TETRON_MEASUREMENTS_PER_ROUND * topo["measurement_time_s"]
        assumptions.append(
            f"Majorana tetron chip {device.name}: operations are parity measurements, so the physical error is "
            f"the larger measurement error ({physical_error:g}) and a syndrome round is "
            f"{TETRON_MEASUREMENTS_PER_ROUND} measurement times ({cycle_time_s:g} s, an assumption); the surface "
            "code is measured directly by parity measurements, and the chip's values are those in "
            "device.parameters['sources'], several of them assumptions or targets")
    return (1e-3 if physical_error is None else physical_error), (1e-6 if cycle_time_s is None else cycle_time_s)


def code_distance(target, physical_error, prefactor=0.03, threshold=0.01):
    """The smallest odd d with prefactor (p/p_th)^((d+1)/2) <= target."""
    if physical_error >= threshold:
        raise ValueError("the physical error must be below the threshold")
    d = 3
    while prefactor * (physical_error / threshold) ** ((d + 1) / 2) > target:
        d += 2
    return d


def _distillation(injected, logical_error, target):
    """(levels, output error) of the fewest 15-to-1 levels (at most 4) meeting the per-T-state target,
    or None when none does at this distance."""
    error = injected
    for levels in range(1, 5):
        error = 35 * error ** 3 + FACTORY_PATCHES * FACTORY_CYCLES * logical_error
        if error <= target:
            return levels, error
    return None
