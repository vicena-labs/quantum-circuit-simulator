"""Hamiltonians as Pauli sums: standard models, loaders, and circuits that evolve them.

A Hamiltonian is a list of (coefficient, {wire: axis}) terms, the observable
form every solver accepts, with an empty mapping for the identity. Models
cover spin chains and lattices, Max-Cut costs and the Fermi-Hubbard model
under the Jordan-Wigner transformation. `trotter` and `qaoa` build Programs
whose angles are shared, weighted parameters, so one prepared plan serves
every time or angle value and gradients come for free.

    from quantum_circuit_simulator import hamiltonians as hm
    h = hm.transverse_field_ising(hm.grid(4, 4), j=1.0, h=0.5)
    circuit = hm.trotter(h, time="t", steps=4)     # time as a parameter
    h.expectation(circuit, values={"t": 0.8})
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .interop import Angle, ConversionError, Program, parse_observable

AXES = ("X", "Y", "Z")


@dataclass
class Hamiltonian:
    """A real-weighted Pauli sum on n_qubits wires, with like terms merged."""
    n_qubits: int
    terms: list = field(default_factory=list)
    name: str = "hamiltonian"

    def __post_init__(self):
        merged = {}
        for coefficient, pauli in self.terms:
            word = tuple(sorted((int(q), a) for q, a in pauli.items() if a != "I"))
            if any(a not in AXES for _, a in word) or any(not 0 <= q < self.n_qubits for q, _ in word):
                raise ConversionError(f"term {pauli} is not a Pauli word on {self.n_qubits} wires")
            merged[word] = merged.get(word, 0.0) + float(coefficient)
        self.terms = [(c, dict(w)) for w, c in merged.items() if c != 0.0]

    def __add__(self, other):
        return Hamiltonian(max(self.n_qubits, other.n_qubits), self.terms + other.terms, self.name)

    def __mul__(self, k):
        return Hamiltonian(self.n_qubits, [(c * float(k), p) for c, p in self.terms], self.name)

    __rmul__ = __mul__

    @property
    def offset(self):
        """The identity coefficient."""
        return sum(c for c, p in self.terms if not p)

    @property
    def norm(self):
        """l1 norm of the coefficients, a bound on |<H>|."""
        return math.fsum(abs(c) for c, _ in self.terms)

    def summary(self):
        weights = [len(p) for _, p in self.terms]
        return {"name": self.name, "n_qubits": self.n_qubits, "terms": len(self.terms),
                "max_weight": max(weights, default=0), "l1_norm": self.norm, "offset": self.offset,
                "qubitwise_commuting_groups": len(self.groups())}

    def groups(self):
        """Qubit-wise commuting groups, measurable together in one basis (greedy, largest first).

        The number of groups sets how many circuit variants a shot-based
        estimate of <H> needs on hardware.
        """
        groups = []
        for c, p in sorted(self.terms, key=lambda t: -abs(t[0])):
            if not p:
                continue
            for basis, members in groups:
                if all(basis.get(q, a) == a for q, a in p.items()):
                    basis.update(p)
                    members.append((c, p))
                    break
            else:
                groups.append((dict(p), [(c, p)]))
        return [members for _, members in groups]

    def matrix(self):
        """The dense 2^n x 2^n matrix; wire 0 is the least significant bit. At most 12 qubits."""
        if self.n_qubits > 12:
            raise ValueError("dense matrices are limited to 12 qubits; use the state-vector backend")
        from .statevector import pauli_matrix
        dim = 2 ** self.n_qubits
        out = np.zeros((dim, dim), complex)
        for c, p in self.terms:
            out += c * pauli_matrix(p, self.n_qubits)
        return out

    def ground_state_energy(self):
        """Exact lowest eigenvalue: dense up to 12 qubits, sparse Lanczos (SciPy) up to 22."""
        if self.n_qubits <= 12:
            return float(np.linalg.eigvalsh(self.matrix())[0])
        if self.n_qubits > 22:
            raise ValueError("exact diagonalisation is limited to 22 qubits")
        try:
            from scipy.sparse.linalg import LinearOperator, eigsh
        except ImportError as exc:
            raise ImportError("more than 12 qubits needs SciPy: pip install scipy") from exc
        from .statevector import apply_pauli_sum
        dim = 2 ** self.n_qubits
        op = LinearOperator((dim, dim), matvec=lambda v: apply_pauli_sum(self.terms, v.astype(complex), self.n_qubits),
                            dtype=complex)
        return float(eigsh(op, k=1, which="SA", tol=1e-10)[0][0].real)

    def expectation(self, circuit, *, values=None, gradients=False, **options):
        """<H> on a circuit (a Program, OpenQASM, a Qiskit circuit or a Circuit), by the router."""
        from .experiments import solve
        from .interop import convert
        converted = convert(circuit, values=values)
        shifts = converted.shifts(values) if converted.parameters else None
        return solve(converted.circuit, self.terms, shifts, gradients=gradients, **options)

    def to_dict(self):
        return {"schema": "quantum-circuit-simulator-hamiltonian/1", "name": self.name, "n_qubits": self.n_qubits,
                "terms": [{"coefficient": c, "pauli": {str(q): a for q, a in sorted(p.items())}}
                          for c, p in self.terms]}

    def to_json(self, path=None):
        text = json.dumps(self.to_dict(), indent=1)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_string(self):
        return " + ".join(f"{c!r}*" + ("*".join(f"{a}{q}" for q, a in sorted(p.items())) or "I")
                          for c, p in self.terms).replace("+ -", "- ")


def load(source, n_qubits=None, name=None):
    """A Hamiltonian from a JSON file or dict, a Pauli-sum string, a Qiskit SparsePauliOp,
    an OpenFermion QubitOperator, or a list of (coefficient, {wire: axis}) terms."""
    if isinstance(source, Hamiltonian):
        return source
    if isinstance(source, (str, Path)) and str(source).endswith(".json") and Path(source).exists():
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(source, dict) and "terms" in source:
        terms = [(t["coefficient"], {int(q): a for q, a in t["pauli"].items()}) for t in source["terms"]]
        return Hamiltonian(n_qubits or source.get("n_qubits") or _width(terms), terms,
                           name or source.get("name", "hamiltonian"))
    if hasattr(source, "terms") and isinstance(source.terms, dict):   # OpenFermion QubitOperator
        terms = []
        for word, c in source.terms.items():
            c = complex(c)
            if abs(c.imag) > 1e-12:
                raise ConversionError("QubitOperator has an imaginary coefficient; it is not Hermitian")
            terms.append((c.real, {q: a for q, a in word}))
        return Hamiltonian(n_qubits or _width(terms), terms, name or "hamiltonian")
    terms = parse_observable(source)
    width = n_qubits or _width(terms)
    if hasattr(source, "num_qubits"):
        width = n_qubits or source.num_qubits
    return Hamiltonian(width, terms, name or "hamiltonian")


def _width(terms):
    return 1 + max((q for _, p in terms for q in p), default=0)


# ---------------------------------------------------------------- lattices
def chain(n, periodic=False):
    """Edges of a line of n sites, closed into a ring when periodic."""
    edges = [(i, i + 1) for i in range(n - 1)]
    return edges + [(n - 1, 0)] if periodic and n > 2 else edges


def grid(rows, columns, periodic=False):
    """Edges of a rows x columns square lattice, sites numbered row by row."""
    edges = []
    for r in range(rows):
        for c in range(columns):
            i = r * columns + c
            if c + 1 < columns or (periodic and columns > 2):
                edges.append((i, r * columns + (c + 1) % columns))
            if r + 1 < rows or (periodic and rows > 2):
                edges.append((i, ((r + 1) % rows) * columns + c))
    return edges


def random_regular(n, degree=3, seed=0):
    """Edges of a random regular graph (a standard Max-Cut benchmark), by retrying pairings."""
    if n * degree % 2:
        raise ValueError("n * degree must be even")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        stubs = list(rng.permutation(np.repeat(np.arange(n), degree)))
        edges = set()
        ok = True
        while stubs:
            a, b = stubs.pop(), stubs.pop()
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((int(min(a, b)), int(max(a, b))))
        if ok:
            return sorted(edges)
    raise RuntimeError("could not draw a simple regular graph; try another seed")


def _sites(edges):
    return 1 + max((max(e) for e in edges), default=-1)


# ---------------------------------------------------------------- models
def transverse_field_ising(edges, j=1.0, h=1.0, n_qubits=None):
    """H = -J sum_<ij> Z_i Z_j - h sum_i X_i."""
    n = n_qubits or _sites(edges)
    terms = [(-j, {a: "Z", b: "Z"}) for a, b in edges] + [(-h, {i: "X"}) for i in range(n)]
    return Hamiltonian(n, terms, "transverse-field-ising")


def heisenberg(edges, j=(1.0, 1.0, 1.0), h=0.0, n_qubits=None):
    """H = sum_<ij> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) + h sum_i Z_i (XXZ when Jx = Jy)."""
    n = n_qubits or _sites(edges)
    jx, jy, jz = (j, j, j) if isinstance(j, (int, float)) else j
    terms = [(k, {a: axis, b: axis}) for a, b in edges for k, axis in ((jx, "X"), (jy, "Y"), (jz, "Z")) if k]
    terms += [(h, {i: "Z"}) for i in range(n)] if h else []
    return Hamiltonian(n, terms, "heisenberg")


def maxcut(edges, weights=None, n_qubits=None):
    """H = sum_<ij> w_ij (Z_i Z_j - 1) / 2, so <H> = -(expected cut weight); minimise it."""
    n = n_qubits or _sites(edges)
    weights = [1.0] * len(edges) if weights is None else list(weights)
    terms = [(w / 2, {a: "Z", b: "Z"}) for (a, b), w in zip(edges, weights)]
    terms.append((-sum(weights) / 2, {}))
    return Hamiltonian(n, terms, "maxcut")


def fermi_hubbard(edges, t=1.0, u=4.0, mu=0.0, n_sites=None):
    """Spinful Fermi-Hubbard model under Jordan-Wigner.

    H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_i,up n_i,down - mu sum_i,s n_is.
    Spin-up orbitals are wires 0..N-1 and spin-down N..2N-1, each in site order,
    so a hop between sites i < j carries Z on the orbitals between them.
    """
    n = n_sites or _sites(edges)
    terms = []
    for spin in (0, n):
        for a, b in edges:
            i, k = sorted((a + spin, b + spin))
            string = {q: "Z" for q in range(i + 1, k)}
            # c+_i c_k + h.c. = (X_i Z.. X_k + Y_i Z.. Y_k) / 2
            terms += [(-t / 2, {i: "X", **string, k: "X"}), (-t / 2, {i: "Y", **string, k: "Y"})]
    for i in range(n):
        up, down = i, i + n
        # n = (1 - Z) / 2, so U n_up n_down = U/4 (1 - Z_up - Z_down + Z_up Z_down).
        terms += [(u / 4, {}), (-u / 4, {up: "Z"}), (-u / 4, {down: "Z"}), (u / 4, {up: "Z", down: "Z"})]
        terms += [(-mu / 2, {}), (mu / 2, {up: "Z"}), (-mu / 2, {}), (mu / 2, {down: "Z"})] if mu else []
    return Hamiltonian(2 * n, terms, "fermi-hubbard")


MODELS = {
    "transverse-field-ising": transverse_field_ising,
    "heisenberg": heisenberg,
    "maxcut": maxcut,
    "fermi-hubbard": fermi_hubbard,
}


def model(name, lattice="chain", size=(8,), periodic=False, seed=0, **parameters):
    """A model by name on a named lattice: chain (n), grid (rows, columns) or regular (n, degree)."""
    if name not in MODELS:
        raise ConversionError(f"unknown model {name!r}", "choose from " + ", ".join(MODELS))
    size = (size,) if isinstance(size, int) else tuple(size)
    if lattice == "chain":
        edges = chain(size[0], periodic)
    elif lattice == "grid":
        edges = grid(size[0], size[1] if len(size) > 1 else size[0], periodic)
    elif lattice == "regular":
        edges = random_regular(size[0], size[1] if len(size) > 1 else 3, seed)
    else:
        raise ConversionError(f"unknown lattice {lattice!r}", "choose chain, grid or regular")
    return MODELS[name](edges, **parameters)


# ---------------------------------------------------------------- circuits
def pauli_rotation(pauli, angle):
    """Instructions for exp(-i angle/2 P): a basis change, a CX ladder and one RZ.

    angle may be a number or an Angle, so the rotation can carry a shared parameter.
    """
    wires = sorted(pauli)
    if len(wires) == 1:
        (q,) = wires
        return [(f"r{pauli[q].lower()}", [q], [angle])]
    before, after = [], []
    for q in wires:
        if pauli[q] == "X":
            before.append(("h", [q], []))
            after.append(("h", [q], []))
        elif pauli[q] == "Y":
            # RX(pi/2)^dagger Y RX(pi/2) = Z.
            before.append(("rx", [q], [math.pi / 2]))
            after.append(("rx", [q], [-math.pi / 2]))
    ladder = [("cx", [a, b], []) for a, b in zip(wires, wires[1:])]
    return before + ladder + [("rz", [wires[-1]], [angle])] + ladder[::-1] + after[::-1]


def evolution_step(hamiltonian, dt, order=1):
    """Instructions for one product-formula step approximating exp(-i H dt).

    order 1 is Lie-Trotter, order 2 the symmetric Strang splitting, and any
    higher even order Suzuki's recursion (J. Math. Phys. 32, 400 (1991)):
    S_2k(dt) = S_2k-2(p dt)^2 S_2k-2((1 - 4p) dt) S_2k-2(p dt)^2 with
    p = 1 / (4 - 4^(1/(2k-1))). dt may be a number or an Angle. Identity terms
    only add a global phase and are dropped.
    """
    terms = [(c, p) for c, p in hamiltonian.terms if p]
    if order == 1:
        return [g for c, p in terms for g in pauli_rotation(p, dt * (2 * c))]
    if order == 2:
        half = [g for c, p in terms[:-1] for g in pauli_rotation(p, dt * c)]
        middle = pauli_rotation(terms[-1][1], dt * (2 * terms[-1][0])) if terms else []
        mirrored = [g for c, p in reversed(terms[:-1]) for g in pauli_rotation(p, dt * c)]
        return half + middle + mirrored
    if order > 2 and order % 2 == 0:
        p = 1 / (4 - 4 ** (1 / (order - 1)))
        outer = evolution_step(hamiltonian, dt * p, order - 2)
        return 2 * outer + evolution_step(hamiltonian, dt * (1 - 4 * p), order - 2) + 2 * outer
    raise ValueError("order must be 1 or an even number")


def trotter(hamiltonian, time, steps=1, order=1, initial=None):
    """A Program approximating exp(-i H time) |initial> with `steps` steps of a product formula
    of the given order (1, 2, or a higher even order; see `evolution_step`).

    time is a number, or a parameter name (str) so that one prepared plan
    serves every time and the gradient is d<O>/dt. initial is None for |0..0>,
    "plus" for |+..+>, or a list of instructions to prepare first.
    """
    dt = Angle(time, 1.0 / steps) if isinstance(time, str) else float(time) / steps
    prep = ([("h", [q], []) for q in range(hamiltonian.n_qubits)] if initial == "plus"
            else list(initial or []))
    body = [g for _ in range(steps) for g in evolution_step(hamiltonian, dt, order)]
    return Program(hamiltonian.n_qubits, prep + body, [time] if isinstance(time, str) else [])


def qaoa(cost, depth=1, mixer="x"):
    """A depth-p QAOA Program: |+..+>, then exp(-i gamma_k C) exp(-i beta_k B) for k = 1..p.

    Parameters are gamma_1, beta_1, ..., gamma_p, beta_p, each shared across
    every term it multiplies (weights enter as parameter scales). mixer "x" is
    the transverse field B = sum X_i; "xy" is the ring XY mixer that conserves
    Hamming weight.
    """
    n = cost.n_qubits
    instructions = [("h", [q], []) for q in range(n)]
    names = []
    for k in range(1, depth + 1):
        gamma, beta = f"gamma_{k}", f"beta_{k}"
        names += [gamma, beta]
        instructions += evolution_step(cost, Angle(gamma))
        if mixer == "x":
            instructions += [("rx", [q], [Angle(beta, 2.0)]) for q in range(n)]
        elif mixer == "xy":
            ring = Hamiltonian(n, [(0.5, {a: ax, b: ax}) for a, b in chain(n, periodic=True) for ax in ("X", "Y")])
            instructions += evolution_step(ring, Angle(beta))
        else:
            raise ValueError("mixer must be 'x' or 'xy'")
    return Program(n, instructions, names)
