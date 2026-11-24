"""Circuits: generated families for scaling studies and a library of standard benchmarks.

Each family returns (qasm, observable): dependency-free OpenQASM 2 text and a
natural observable string for it, so sweeps over size need no other tools.
The library bundles benchmark circuits from MQT Bench (MIT) and Supermarq
(Apache-2.0) at several sizes, as OpenQASM 2 in this package's gate set.

    from quantum_circuit_simulator.circuits import FAMILIES, family, library, load
    qasm, observable = family("qaoa-ring", n=12, p=2)
    [c["name"] for c in library(suite="MQT Bench")]
    qasm = load("mqt/qft", 16)
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from importlib import resources

import numpy as np

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n'


def ghz(n=8):
    """GHZ state by a CNOT chain; <Z0 Z(n-1)> is 1 without noise."""
    body = "h q[0];\n" + "".join(f"cx q[{i}],q[{i + 1}];\n" for i in range(n - 1))
    return HEADER.format(n=n) + body, f"Z0*Z{n - 1}"


def hardware_efficient(n=8, layers=2, seed=0):
    """RY-RZ layers with a CZ brickwork; the observable is the mean Z."""
    rng = np.random.default_rng(seed)
    body = []
    for layer in range(layers):
        for q in range(n):
            body.append(f"ry({rng.uniform(-1, 1):.17g}) q[{q}];\nrz({rng.uniform(-1, 1):.17g}) q[{q}];\n")
        body += [f"cz q[{q}],q[{q + 1}];\n" for q in range(layer % 2, n - 1, 2)]
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.17g}*Z{q}" for q in range(n))


def qaoa_ring(n=8, p=1, gamma=0.4, beta=0.3):
    """QAOA for MaxCut on a ring; the observable is the mean ZZ over ring edges
    (the expected cut fraction is (1 - <mean ZZ>) / 2)."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    body = ["h q;\n"]
    for _ in range(p):
        body += [f"rzz({2 * gamma:.17g}) q[{a}],q[{b}];\n" for a, b in edges]
        body.append(f"rx({2 * beta:.17g}) q;\n")
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.17g}*Z{a}*Z{b}" for a, b in edges)


def ising_trotter(n=8, steps=4, dt=0.1, coupling=1.0, field=1.0):
    """First-order Trotter evolution of the transverse-field Ising chain from |0...0>;
    the observable is the mean Z magnetisation."""
    body = []
    for _ in range(steps):
        body += [f"rzz({-2 * coupling * dt:.17g}) q[{i}],q[{i + 1}];\n" for i in range(n - 1)]
        body.append(f"rx({-2 * field * dt:.17g}) q;\n")
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.17g}*Z{q}" for q in range(n))


def kicked_ising(n=127, steps=5, theta_h=0.5, theta_j=-np.pi / 2):
    """IBM's utility experiment (Kim et al., Nature 618, 500, 2023): Trotterised kicked Ising
    on the 127-qubit heavy-hex chip, RX(theta_h) on every qubit then RZZ(theta_j) on every
    coupler, per step. n=127 uses the Eagle heavy-hex couplers; other sizes use a ring.
    The observable is the mean Z magnetisation."""
    if n == 127:
        from .devices import _heavy_hex_edges
        _, edges, _ = _heavy_hex_edges(7, 15)
    else:
        edges = [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    layers = _edge_layers(edges)
    body = []
    for _ in range(steps):
        body.append(f"rx({theta_h:.17g}) q;\n")
        for layer in layers:
            body += [f"rzz({theta_j:.17g}) q[{a}],q[{b}];\n" for a, b in layer]
    return HEADER.format(n=n) + "".join(body), " + ".join(f"{1 / n:.17g}*Z{q}" for q in range(n))


def _edge_layers(edges):
    """Greedy partition of couplers into layers of disjoint pairs (three for heavy-hex)."""
    layers, remaining = [], list(edges)
    while remaining:
        busy, layer, rest = set(), [], []
        for a, b in remaining:
            if a in busy or b in busy:
                rest.append((a, b))
            else:
                layer.append((a, b))
                busy.update((a, b))
        layers.append(layer)
        remaining = rest
    return layers


FAMILIES = {
    "ghz": (ghz, "GHZ state by a CNOT chain; observable Z0*Z(n-1), ideally 1"),
    "hardware-efficient": (hardware_efficient, "RY-RZ layers with CZ brickwork; observable mean Z"),
    "qaoa-ring": (qaoa_ring, "QAOA MaxCut on a ring; observable mean ZZ over edges"),
    "ising-trotter": (ising_trotter, "Trotterized transverse-field Ising chain; observable mean Z"),
    "kicked-ising": (kicked_ising, "IBM's 127-qubit utility experiment (heavy-hex kicked Ising); observable mean Z"),
}


def family(name, **parameters):
    """(qasm, observable) for a named family; parameters as in its function."""
    if name not in FAMILIES:
        raise KeyError(f"unknown circuit family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name][0](**parameters)


# ---------------------------------------------------------------- library
@lru_cache(maxsize=1)
def _bundle():
    path = resources.files("quantum_circuit_simulator") / "data" / "circuits.json.gz"
    if not path.is_file():
        return []
    with path.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        return json.load(f)["circuits"]


def library(suite=None, search=None):
    """Benchmark circuits in the bundled library, without their text: name, suite, qubits, gates and source."""
    out = []
    for c in _bundle():
        if suite and c["suite"].lower() != suite.lower():
            continue
        if search and search.lower() not in c["name"].lower():
            continue
        out.append({k: v for k, v in c.items() if k != "qasm"})
    return out


def load(name, n_qubits=None):
    """OpenQASM 2 of a library circuit: the given size, or the smallest one with at least n_qubits.

    Without n_qubits, the smallest size. Raises KeyError listing the sizes available.
    """
    sizes = sorted((c for c in _bundle() if c["name"] == name), key=lambda c: c["n_qubits"])
    if not sizes:
        names = sorted({c["name"] for c in _bundle()})
        raise KeyError(f"unknown library circuit {name!r}; {len(names)} are available, e.g. {names[:8]}; "
                       "see `quantum-circuit-simulator circuits`")
    if n_qubits is None:
        return sizes[0]["qasm"]
    for c in sizes:
        if c["n_qubits"] >= n_qubits:
            return c["qasm"]
    raise KeyError(f"{name} has sizes {[c['n_qubits'] for c in sizes]}; none has {n_qubits} or more qubits")


def library_names():
    return sorted({c["name"] for c in _bundle()})
