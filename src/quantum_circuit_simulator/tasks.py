"""JSON tasks for the command line: Hamiltonians, sampling, states, optimisation, ZNE and shot noise.

Each task takes a plain JSON specification and returns a JSON-ready body; the
worker wraps it in the result envelope (schema and provenance). A circuit is
OpenQASM text; a device is a description from `quantum-circuit-simulator device NAME --edges`;
a Hamiltonian is {"model": ..., "lattice": ..., "size": [...], "parameters": {...}},
{"terms": [...]} or a Pauli-sum string.
"""
from __future__ import annotations

import numpy as np

from .contracts import Progress

KINDS = ("hamiltonian", "sample", "statevector", "optimize", "zne", "shot-noise", "anneal", "circuits", "cad")
MAX_STATEVECTOR_OUTPUT_QUBITS = 16   # 2^16 amplitudes is about 2 MB of JSON


def run(spec, progress=False):
    """(kind, body) for a task specification {"kind": ..., ...}."""
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown task kind {kind!r}; choose from {', '.join(KINDS)}")
    handler = {"hamiltonian": _hamiltonian, "sample": _sample, "statevector": _statevector,
               "optimize": _optimize, "zne": _zne, "shot-noise": _shot_noise, "anneal": _anneal,
               "circuits": _circuits, "cad": _cad}[kind]
    return kind, handler(spec, progress)


def hamiltonian_of(spec):
    from . import hamiltonians as hm
    if isinstance(spec, dict) and "model" in spec:
        return hm.model(spec["model"], spec.get("lattice", "chain"), spec.get("size", [8]),
                        spec.get("periodic", False), spec.get("seed", 0), **spec.get("parameters", {}))
    return hm.load(spec)


def _device(spec):
    from .devices import Device
    return Device.from_dict(spec) if spec else None


def _hamiltonian(spec, progress):
    h = hamiltonian_of(spec["hamiltonian"])
    body = {"hamiltonian": h.to_dict(), "summary": h.summary()}
    if h.n_qubits <= spec.get("exact_up_to", 14):
        body["ground_state_energy"] = h.ground_state_energy()
    return body


def _sample(spec, progress):
    from .statevector import sample, sample_compiled
    shots, seed = int(spec.get("shots", 1024)), spec.get("seed")
    device = _device(spec.get("device"))
    if device is not None:
        from .compiler import compile_circuit
        compiled = compile_circuit(spec["circuit"], device, noise_scale=spec.get("noise_scale", 1.0),
                                   router=spec.get("router", "greedy"), values=spec.get("values"))
        out = sample_compiled(compiled, shots, seed=seed, method=spec.get("method", "auto"))
        out["compile"] = {k: compiled.report[k] for k in ("device", "swaps", "two_qubit_gates", "depth", "duration_s")}
        return out
    return sample(spec["circuit"], shots, values=spec.get("values"), seed=seed, method=spec.get("method", "auto"))


def _statevector(spec, progress):
    from .statevector import statevector
    psi = statevector(spec["circuit"], values=spec.get("values"))
    n = int(np.log2(psi.size))
    if n > MAX_STATEVECTOR_OUTPUT_QUBITS:
        raise ValueError(f"{n} qubits is too many amplitudes to print; the limit is {MAX_STATEVECTOR_OUTPUT_QUBITS}",)
    return {"n_qubits": n, "amplitudes": [[float(a.real), float(a.imag)] for a in psi],
            "probabilities_top": _top(psi, spec.get("top", 16))}


def _top(psi, k):
    p = np.abs(psi) ** 2
    order = np.argsort(-p)[:k]
    n = int(np.log2(psi.size))
    return [{"bitstring": format(int(i), f"0{n}b"), "probability": float(p[i])} for i in order if p[i] > 0]


def _optimize(spec, progress):
    from . import algorithms as al
    h = hamiltonian_of(spec["hamiltonian"])
    device = _device(spec.get("device"))
    algorithm = spec.get("algorithm", "vqe")
    steps = int(spec.get("steps", 200))
    report = Progress("optimize", steps, enabled=progress)
    options = dict(device=device, noise_scale=spec.get("noise_scale", 1.0), method=spec.get("method", "adam"),
                   steps=steps, learning_rate=spec.get("learning_rate", 0.05), seed=spec.get("seed", 0),
                   callback=lambda row: report(iteration=row["iteration"], value=row["value"]),
                   keep_parameters=spec.get("keep_parameters", False))
    if algorithm == "vqe":
        a = spec.get("ansatz", {})
        ansatz = al.hardware_efficient(h.n_qubits, layers=a.get("layers", 2), edges=a.get("edges"),
                                       rotations=tuple(a.get("rotations", ("ry", "rz"))),
                                       entangler=a.get("entangler", "cz"))
        result = al.vqe(h, ansatz, **options)
    elif algorithm == "qaoa":
        result = al.qaoa(h, spec.get("depth", 1), shots=spec.get("shots", 1024), mixer=spec.get("mixer", "x"),
                         **options)
    else:
        raise ValueError("algorithm must be vqe or qaoa")
    return {"algorithm": algorithm, **result.to_dict()}


def _zne(spec, progress):
    from .algorithms import zne
    return zne(spec["circuit"], spec["observable"], _device(spec["device"]),
               scales=tuple(spec.get("scales", (1.0, 1.5, 2.0, 3.0))), fit=spec.get("fit", "richardson"),
               values=spec.get("values"), router=spec.get("router", "greedy"))


def _shot_noise(spec, progress):
    from .algorithms import shot_noise
    return shot_noise(spec["circuit"], hamiltonian_of(spec["observable"]), int(spec.get("shots", 10_000)),
                      values=spec.get("values"), allocation=spec.get("allocation", "weighted"), seed=spec.get("seed"))


def _anneal(spec, progress):
    from . import annealing
    from .devices import get
    if "problem" in spec:
        problem = annealing.Problem.from_dict(spec["problem"])
    else:
        problem = annealing.Problem.from_hamiltonian(hamiltonian_of(spec["hamiltonian"]))
    device = get(spec["device_name"]) if spec.get("device_name") else _device(spec.get("device"))
    return annealing.anneal(problem, device, sampler=spec.get("sampler", "sqa"), num_reads=spec.get("num_reads", 100),
                            chain_strength=spec.get("chain_strength"), control_noise=spec.get("control_noise", 0.0),
                            seed=spec.get("seed"), num_sweeps=spec.get("num_sweeps"))


def _circuits(spec, progress):
    from .circuits import FAMILIES, library, load
    if spec.get("name"):
        qasm = load(spec["name"], spec.get("size"))
        return {"name": spec["name"], "qasm": qasm}
    return {"families": {k: v[1] for k, v in FAMILIES.items()},
            "library": library(spec.get("suite"), spec.get("search"))}


def _cad(spec, progress):
    from . import cad
    from .devices import get
    device = get(spec["device_name"]) if spec.get("device_name") else _device(spec["device"])
    return cad.write(device, spec["out"], fmt=spec.get("format"), pitch=spec.get("pitch", 800.0))
