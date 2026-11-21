"""Machine-readable descriptions, dry-run planning and error hints for agents.

The command line exposes these as `quantum-circuit-simulator capabilities`, `schema`,
`plan` and `guide`, and attaches `hint` to every error, so an agent working
in a shell can learn the tool, check feasibility and recover from failures
without reading the source.
"""
from __future__ import annotations

from importlib import resources

from . import __version__

EXIT_CODES = {"0": "success", "2": "invalid request or failed evaluation; see error.type and error.hint",
              "124": "worker exceeded --timeout and was terminated"}

BACKENDS = {
    "native": "Default. Exact sparse Pauli propagation in C++; truncation optional.",
    "graph": "Compiles the propagation once; fastest when one circuit is evaluated at many parameter vectors. "
             "Truncation optional. Falls back to native over its node or edge budget.",
    "dense": "Exact local density matrices for cones of at most 6 wires.",
    "python": "Slow reference implementation, for cross-checks.",
    "auto": "Native, falling back to dense when an exact cone exceeds the term cap.",
}


def capabilities():
    from .contracts import OUTPUTS
    from .interop import SUPPORTED_GATES
    return {
        "version": __version__,
        "computes": "Weighted sums of Pauli expectations <psi|O|psi> of noisy circuits from |0...0>, "
                    "with exact gradients in up to 4096 shared parameters (an angle may be k*p + b).",
        "inputs": {
            "json_request": "schema_version 1 request; see `quantum-circuit-simulator schema`",
            "openqasm2": "built in, no dependencies",
            "openqasm3": "needs `pip install qiskit qiskit-qasm3-import`",
            "qiskit": "quantum_circuit_simulator.interop.from_qiskit and quantum_circuit_simulator.estimator.Estimator (Qiskit Estimator V2)",
        },
        "native_gates": ["H", "SX", "X", "Y", "Z", "RX", "RY", "RZ", "CZ", "ECR", "FSIM", "UNITARY (any fixed 1- or 2-qubit matrix)"],
        "devices": "chip catalogue: `quantum-circuit-simulator devices`; describe one with `quantum-circuit-simulator device NAME`; "
                   "apply a chip's noise and check its connectivity with --device on run and plan; "
                   "route onto it and simulate gate errors, idle T1/T2 and readout with --compile, "
                   "or report SWAPs, depth and duration with `quantum-circuit-simulator compile`",
        "design": "`quantum-circuit-simulator design` builds chips (modules with link errors, defects, couplers, errors from "
                  "T1/T2 and gate times, per-qubit spread, crosstalk, growth to more qubits with the same "
                  "architecture) as device files; sweeps grow a chip per size with --device fit:NAME; `quantum-circuit-simulator compare` ranks chips on a benchmark family",
        "sweeps": "`quantum-circuit-simulator sweep` compiles and solves over chips, noise scales and sizes, with built-in "
                  "families ghz, hardware-efficient, qaoa-ring and ising-trotter; the solver is chosen per point "
                  "(exact native, exact dense, then truncated with a convergence check)",
        "algorithms": "`quantum-circuit-simulator hamiltonian` (models and loaders), `optimize` (VQE or QAOA with exact gradients, "
                      "Adam, gradient descent, SPSA or L-BFGS, full trace), `zne` (zero-noise extrapolation on a device), "
                      "`shot-noise` (standard error of a hardware run with N shots)",
        "real_chips": "65 IBM chips (Heron, Eagle, Falcon, ...) and 3 Google chips from calibration snapshots, "
                      "e.g. ibm_fez, ibm_torino, google_willow_pink; listed by `quantum-circuit-simulator devices`",
        "circuit_library": "`quantum-circuit-simulator circuits`: MQT Bench and Supermarq benchmarks up to 127 qubits, "
                           "and families including IBM's 127-qubit kicked-Ising utility experiment",
        "cad": "`quantum-circuit-simulator cad DEVICE --out chip.gds|.dxf|.svg`: a placement layout of any chip",
        "annealing": "`quantum-circuit-simulator anneal`: Ising/QUBO problems embedded on dwave-advantage, "
                     "dwave-advantage2, chimera, pegasus or zephyr; simulated quantum annealing, simulated "
                     "annealing, tabu or exact; needs the dwave extra",
        "states": "`quantum-circuit-simulator sample` (bitstrings with noise and readout; exact density matrix up to 10 qubits, "
                  "trajectories up to 28, Stim for larger Clifford circuits) and `statevector` (amplitudes)",
        "outputs": {"schema": "every result carries schema quantum-circuit-simulator-<kind>/1 and provenance",
                    "kinds": OUTPUTS,
                    "progress": "--progress streams JSON-line events on stderr",
                    "checkpoint": "sweep --checkpoint FILE resumes from completed points"},
        "converted_gates": SUPPORTED_GATES,
        "other_gates": "transpiled by Qiskit when it is installed",
        "noise": ["DEPOLARIZE (1 or 2 wires)", "PAULI_ERROR", "RELAX (T1/T2)", "readout (p01, p10)",
                  "uniform noise for converted circuits: --noise-1q, --noise-2q, --readout"],
        "observables": 'strings like "0.5*Z0*Z1 - X3" (wire-indexed) or "0.5*IXZ" (Qiskit order, rightmost is '
                       "wire 0), or JSON terms {coefficient, pauli: {wire: axis}}",
        "limits": {"logical_qubits": 16384, "wires_per_cone": 256, "shared_parameters": 4096,
                   "observable_terms": 65536, "parameter_vectors_per_request": 128,
                   "max_terms_per_cone": "1..1000000; exact mode refuses beyond it instead of approximating"},
        "backends": BACKENDS,
        "truncation": {"flags": "--min-coefficient F (fraction of the observable l1 norm), --max-weight N",
                       "result_fields": "error_bound and, with gradients, gradient_error_bound: rigorous "
                                        "bounds on |exact - returned|, often loose; check convergence by "
                                        "tightening F tenfold"},
        "not_supported": ["mid-circuit measurement", "reset", "classical control",
                          "non-Pauli observables", "initial states other than |0...0>"],
        "exit_codes": EXIT_CODES,
    }


def schema():
    """JSON Schema of the request, and the response fields."""
    term = {"type": "object", "required": ["coefficient", "pauli"], "additionalProperties": False,
            "properties": {"coefficient": {"type": "number"},
                           "pauli": {"type": "object", "additionalProperties": {"enum": ["I", "X", "Y", "Z"]},
                                     "description": "decimal wire index -> axis; empty means identity"}}}
    common = {"schema_version": {"const": 1},
              "max_terms": {"type": "integer", "minimum": 1, "maximum": 1000000, "default": 50000},
              "parameter_vectors": {"type": "array", "maxItems": 128, "items": {"type": "array",
                                    "items": {"type": "number"}},
                                    "description": "additive shifts per shared parameter (JSON form) or "
                                                   "parameter values in source order (OpenQASM 3 form)"},
              "gradients": {"type": "boolean", "default": True}}
    operation = {"type": "object", "required": ["op"],
                 "description": "H|SX|X|Y|Z|CZ|ECR {op, wires}; RX|RY|RZ {op, wires, theta, param?}; "
                                "FSIM {op, wires, theta, phi, param?}; DEPOLARIZE {op, wires, probability}; "
                                "PAULI_ERROR {op, wire, axis, probability}; RELAX {op, wire, duration, t1, t2}"}
    return {
        "request": {"oneOf": [
            {"title": "circuit", "type": "object", "additionalProperties": False,
             "required": ["schema_version", "n_qubits", "operations", "observables"],
             "properties": {**common, "n_qubits": {"type": "integer", "minimum": 1, "maximum": 16384},
                            "n_parameters": {"type": "integer", "minimum": 0, "maximum": 16},
                            "operations": {"type": "array", "maxItems": 10000, "items": operation},
                            "readout": {"type": "array", "items": {"type": "object",
                                        "required": ["wire", "p01", "p10"]}},
                            "observables": {"type": "array", "minItems": 1, "maxItems": 65536, "items": term}}},
            {"title": "openqasm", "type": "object", "additionalProperties": False,
             "required": ["schema_version", "qasm", "observables"],
             "properties": {**common, "qasm": {"type": "string"},
                            "observables": {"oneOf": [{"type": "string"},
                                                      {"type": "array", "items": term}]},
                            "noise": {"type": "object", "additionalProperties": False, "properties": {
                                "depolarizing_1q": {"type": "number"}, "depolarizing_2q": {"type": "number"},
                                "readout": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                            "maxItems": 2}}},
                            "compile": {"type": "object", "description": "with device: route onto the chip and "
                                        "simulate its noise", "properties": {
                                            "router": {"enum": ["greedy", "sabre"]},
                                            "layout": {"enum": ["greedy", "trivial"]},
                                            "noise_scale": {"type": "number", "minimum": 0},
                                            "idle_noise": {"type": "boolean"}, "noise": {"type": "boolean"}}},
                            "device": {"type": "object", "description": "a quantum-circuit-simulator-device/1 chip description "
                                       "(`quantum-circuit-simulator device NAME --edges`); its noise applies unless noise is "
                                       "given, and the response reports uncoupled two-qubit operations"}}},
        ]},
        "response": {"ok": "boolean",
                     "results": "one {value, gradient[, error_bound, gradient_error_bound]} per parameter vector",
                     "backend": "the method actually used (a fallback is reported in diagnostics.fallback_reason)",
                     "diagnostics": "cone sizes, groups, term counts", "numerics": "declared numerical policy",
                     "timings": "seconds", "error": "{type, message, hint} when ok is false"},
        "exit_codes": EXIT_CODES,
    }


def guide():
    return resources.files("quantum_circuit_simulator").joinpath("AGENT_GUIDE.md").read_text(encoding="utf-8")


def plan(circuit, observables):
    """Structural dry run: cone sizes, a rigorous term bound, and recommended next commands.

    The bound multiplies, over each observable's cone, the largest number of
    Pauli terms any stage can map one term to (1 for Clifford gates and Pauli
    noise, 2 for a rotation or relaxation) and caps it by 4^wires. When every
    cone's bound fits max_terms, exact evaluation cannot exceed the cap.
    """
    from ._core._graph import _support_record
    from ._core._indexed import ConeGroups
    from ._core._reference import ConeOverflow
    impl = circuit._impl
    facts = {"n_qubits": circuit.n_qubits, "n_parameters": circuit.n_parameters,
             "stages": len(impl.stages), "observable_terms": len(observables), "max_terms": impl.max_terms}
    try:
        cones = ConeGroups(impl, observables, max_cone_qubits=256)
    except ConeOverflow as exc:
        facts.update(feasible=False, reason=str(exc),
                     recommendations=["The observable's backward light cone exceeds 256 wires: shorten the "
                                      "circuit, or measure observables with smaller support."])
        return facts
    branching, cache = [], {}
    for stage in impl.stages:
        key = (stage.name, stage.phi) if stage.param >= 0 else stage.matrix.tobytes()
        if key not in cache:
            rows = _support_record(stage)[:16]
            cache[key] = max(bin(r).count("1") for r in rows)
        branching.append(cache[key])
    bounds = []
    for cone in cones.cones:
        bound = max(1, len(cone.terms))
        for index in cone.stages:
            bound *= branching[index]
            if bound > 4 ** len(cone.wires):
                break
        bounds.append(min(bound, 4 ** len(cone.wires)))
    widest = max((len(c.wires) for c in cones.cones), default=0)
    worst = max(bounds, default=1)
    guaranteed = worst <= impl.max_terms
    facts.update(feasible=True, widest_cone_wires=widest,
                 deepest_cone_stages=max((len(c.stages) for c in cones.cones), default=0),
                 cone_groups=len(cones.groups), largest_term_bound=worst if worst < 10 ** 18 else "above 1e18",
                 exact_guaranteed=guaranteed)
    recommendations = []
    if guaranteed:
        recommendations.append("Exact evaluation fits the term cap: `quantum-circuit-simulator run <input>`.")
    else:
        recommendations.append("Exact evaluation may exceed the term cap (the bound is a worst case; it often "
                               "succeeds). Try `quantum-circuit-simulator run <input>`; if it fails with ConeOverflow, use "
                               "`--min-coefficient 1e-4`, then `1e-5`, and accept the value once it stops "
                               "changing.")
    if widest <= 6:
        recommendations.append("Every cone has at most 6 wires: `--backend dense` is exact regardless of depth.")
    if circuit.n_parameters:
        recommendations.append("For many parameter vectors of one circuit, `--backend graph` compiles once and "
                               "evaluates each vector far faster.")
    if len(cones.groups) > 1:
        recommendations.append(f"{len(cones.groups)} independent cone groups: `--threads N` can help when "
                               "each group is large.")
    facts["recommendations"] = recommendations
    return facts


def hint_for(exc):
    """A next step for an exception raised while preparing or evaluating."""
    from ._core._reference import ConeOverflow
    hint = getattr(exc, "hint", None)
    if hint:
        return hint
    text = str(exc)
    if isinstance(exc, ConeOverflow):
        if "qubits" in text:
            return "The light cone is too wide for exact propagation: shorten the circuit or use observables " \
                   "with smaller support."
        return "Exact propagation exceeded the term cap. Rerun with --min-coefficient 1e-4 (approximate, with " \
               "a rigorous error_bound), then 1e-5 to check convergence; or --backend auto, which falls back " \
               "to dense for cones of at most 6 wires; or raise max_terms (at most 1000000)."
    if "truncation" in text:
        return "Truncation works with --backend native, graph or auto."
    if "unsupported" in text or "operation" in text:
        return "Run `quantum-circuit-simulator capabilities` for the supported gates and noise, or submit OpenQASM."
    if "parameter" in text:
        return "Each parameter vector needs exactly one value per shared parameter."
    return "Run `quantum-circuit-simulator schema` for the request format and `quantum-circuit-simulator guide` for usage."
