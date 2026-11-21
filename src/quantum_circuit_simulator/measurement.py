"""Measurement planning: commuting groups, exact shot budgets and classical shadows.

    from quantum_circuit_simulator import hamiltonians as hm, measurement as ms
    h = hm.heisenberg(hm.chain(6))
    ms.group(h, "commuting")                      # fewer groups than qubit-wise commuting ones
    ms.shot_budget(circuit, h, precision=1e-3)   # exact variances and optimal shots per group
    ms.diagonalizing_clifford(group)              # the measurement circuit, as OpenQASM
    ms.shadows(circuit, ["Z0*Z1", "X2"], snapshots=2000, seed=1)

Conventions. Observables are lists of (coefficient, {wire: axis}) terms or
anything `hamiltonians.load` reads; the identity term is a constant and is
never measured. A group's single-shot estimator is sum_P c_P x_P, where
x_P = +-1 is P's outcome in that shot; its variance is
sum_{P,Q} c_P c_Q (<PQ> - <P><Q>), exact from the circuit's state. Bitstrings
print wire 0 rightmost, as in `statevector.sample`.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .api import Circuit

# Single-qubit products a * b = phase * c, for the sign of a commuting product.
_PRODUCT = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
            ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}


# ---------------------------------------------------------------- grouping
def group(hamiltonian, method="qwc"):
    """Terms in groups measurable together, by sorted insertion (Crawford et al., arXiv:1908.06942).

    Terms are taken by decreasing |coefficient| and each joins the first group
    whose every member it commutes with: qubit-wise ("qwc", one product
    basis per group) or fully ("commuting", Gokhale et al., arXiv:1907.13623,
    which needs an entangling Clifford per group; see `diagonalizing_clifford`).
    Returns a list of groups, each a list of (coefficient, {wire: axis}).
    """
    if method not in ("qwc", "commuting"):
        raise ValueError('method must be "qwc" or "commuting"')
    commute = qubitwise_commute if method == "qwc" else commute_fully
    groups = []
    for term in sorted(_terms(hamiltonian), key=lambda t: -abs(t[0])):
        if not term[1]:
            continue
        for members in groups:
            if all(commute(term[1], p) for _, p in members):
                members.append(term)
                break
        else:
            groups.append([term])
    return groups


def _terms(hamiltonian):
    from .hamiltonians import load
    return load(hamiltonian).terms


def qubitwise_commute(p, q):
    """True when the words agree on every wire where both act."""
    return all(q.get(w, a) == a for w, a in p.items())


def commute_fully(p, q):
    """True when the words differ on an even number of wires where both act."""
    return sum(1 for w, a in p.items() if w in q and q[w] != a) % 2 == 0


# ---------------------------------------------------------------- shot budgets
def shot_budget(circuit, hamiltonian, precision, method="commuting", *, values=None):
    """Total shots to estimate <H> to standard error `precision`, with exact group variances.

    circuit is a Circuit, a Compiled (logical observables; readout error is
    not part of the variance), OpenQASM, a Qiskit circuit or a Program. Each
    group's variance comes from exact expectations of its terms and their
    products (in a commuting group PQ is a Hermitian Pauli up to a sign).
    Shots are allotted N_i proportional to sqrt(Var_i), which minimises the
    total for a given error: N = (sum_i sqrt(Var_i))^2 / precision^2.
    Returns the plan for `method` and, for comparison, the qwc plan's total.
    """
    expect = _expectations(circuit, values)
    plan = _plan(group(hamiltonian, method), expect, precision)
    out = {"method": method, "precision": precision, **plan}
    if method != "qwc":
        out["qwc_shots"] = _plan(group(hamiltonian, "qwc"), expect, precision)["shots"]
        out["saving_vs_qwc"] = out["qwc_shots"] / out["shots"] if out["shots"] else 1.0
    return out


def _expectations(circuit, values):
    """A function giving the exact <P> of a Pauli word on the circuit's state, cached."""
    from .experiments import solve
    from .interop import convert
    if hasattr(circuit, "final_layout"):
        layout, shifts, target = circuit.final_layout, (circuit.shifts(values) if circuit.parameters else None), circuit.circuit
    else:
        converted = convert(circuit, values=values)
        layout, shifts, target = None, (converted.shifts(values) if converted.parameters else None), converted.circuit
    cache = {}

    def expect(word):
        if not word:
            return 1.0
        if word not in cache:
            pauli = {layout[q] if layout else q: a for q, a in word}
            cache[word] = solve(target, [(1.0, pauli)], shifts)["value"]
        return cache[word]
    return expect


def _plan(groups, expect, precision):
    rows = [{"terms": len(members), "variance": group_variance(members, expect)} for members in groups]
    roots = [math.sqrt(r["variance"]) for r in rows]
    total = sum(roots) ** 2 / precision ** 2
    for row, root in zip(rows, roots):
        row["shots"] = math.ceil(total * root / sum(roots)) if root else 0
    return {"groups": len(rows), "shots": sum(r["shots"] for r in rows), "optimal_shots": total, "per_group": rows}


def group_variance(members, expect):
    """sum_{P,Q} c_P c_Q (<PQ> - <P><Q>) for one shot of a commuting group; expect(word) gives <word>."""
    words = [_key(p) for _, p in members]
    means = [expect(w) for w in words]
    second = 0.0
    for (c1, p1), w1 in zip(members, words):
        for (c2, p2), w2 in zip(members, words):
            phase, product = pauli_product(p1, p2)
            second += c1 * c2 * phase.real * expect(_key(product))
    first = sum(c * m for (c, _), m in zip(members, means))
    return max(0.0, second - first ** 2)


def _key(pauli):
    return tuple(sorted(pauli.items()))


def pauli_product(p, q):
    """(phase, word) with P Q = phase * word; the phase is +-1 when P and Q commute."""
    phase, out = 1 + 0j, dict(p)
    for w, b in q.items():
        a = out.get(w)
        if a is None:
            out[w] = b
        elif a == b:
            del out[w]
        else:
            factor, out[w] = _PRODUCT[(a, b)]
            phase *= factor
    return phase, out


# ---------------------------------------------------------------- diagonalizing Clifford
def diagonalizing_clifford(members, n_qubits=None):
    """A Clifford circuit C with C P C^dagger = sign * (Z word) for every word P of a commuting group.

    Built by symplectic Gaussian elimination on the group's generators
    (van den Berg and Temme, arXiv:2003.13599): Hadamards make the X block full
    rank, CNOTs reduce it to the identity on pivot wires, S and CZ clear the
    (symmetric) Z block there, and Hadamards on the pivots turn every
    generator into a Z word. Returns `gates` [(name, wires)] in order (h, s,
    cx with control first, cz), `qasm` (OpenQASM 2.0) and `measure`: for each
    term, its sign and the wires whose Z parity gives its outcome.
    """
    words = [p for _, p in members]
    n = n_qubits or 1 + max((w for p in words for w in p), default=0)
    rows = _independent([_symplectic(p, n) for p in words], n)
    gates = []

    def apply(name, *wires):
        gates.append((name, wires))
        for row in rows:
            _conjugate_row(row, name, wires, n)

    pivots = _make_x_full_rank(rows, n, apply)   # X block now in reduced echelon form on `pivots`
    for i, p in enumerate(pivots):
        for c in range(n):
            if c not in pivots and rows[i][c]:
                apply("cx", p, c)
    for i, p in enumerate(pivots):
        if rows[i][n + p]:
            apply("s", p)
        for j in range(i + 1, len(pivots)):
            if rows[i][n + pivots[j]]:
                apply("cz", p, pivots[j])
    for p in pivots:
        apply("h", p)
    measure = []
    for coefficient, pauli in members:
        sign, image = _conjugate_word(pauli, gates)
        if any(a != "Z" for a in image.values()):
            raise ArithmeticError("the group is not mutually commuting; no Clifford diagonalises it")
        measure.append({"term": pauli, "coefficient": coefficient, "sign": sign, "wires": sorted(image)})
    qasm = "\n".join([f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];']
                     + [f"{name} " + ",".join(f"q[{w}]" for w in wires) + ";" for name, wires in gates])
    return {"gates": gates, "qasm": qasm, "measure": measure, "n_qubits": n}


def _symplectic(pauli, n):
    """[x_0..x_{n-1}, z_0..z_{n-1}] over GF(2) as a list of 0/1."""
    row = [0] * (2 * n)
    for w, a in pauli.items():
        row[w] = int(a in "XY")
        row[n + w] = int(a in "YZ")
    return row


def _independent(rows, n):
    """A basis of the rows' GF(2) span (Gaussian elimination over all 2n columns)."""
    basis = [r[:] for r in rows]
    rank = _reduce(basis, 2 * n, list(range(2 * n)))
    return basis[:rank]


def _reduce(rows, width, columns):
    """Reduced row echelon form over GF(2) on the given columns, in place; returns the rank."""
    rank = 0
    for c in columns:
        pivot = next((i for i in range(rank, len(rows)) if rows[i][c]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _make_x_full_rank(rows, n, apply):
    """Apply Hadamards until the X block has full row rank; returns the X pivot wires in row order.

    Rows without an X pivot are pure Z after elimination; their Z parts are
    independent on the non-pivot wires (they commute with the pivot rows), so
    a Hadamard on each of their Z pivots moves it into the X block.
    """
    rank = _reduce(rows, n, list(range(n)))
    x_pivots = [next(c for c in range(n) if rows[i][c]) for i in range(rank)]
    rest = rows[rank:]
    free = [c for c in range(n) if c not in x_pivots]
    z_rank = _reduce(rest, 2 * n, [n + c for c in free])
    rows[rank:] = rest
    for i in range(rank, rank + z_rank):
        apply("h", next(c for c in free if rows[i][n + c]))
    _reduce(rows, n, list(range(n)))
    return [next(c for c in range(n) if row[c]) for row in rows]


def _conjugate_row(row, name, wires, n):
    """Update a symplectic row for P -> g P g^dagger (signs are tracked separately)."""
    if name == "h":
        (q,) = wires
        row[q], row[n + q] = row[n + q], row[q]
    elif name == "s":
        (q,) = wires
        row[n + q] ^= row[q]
    elif name == "cz":
        a, b = wires
        row[n + a] ^= row[b]
        row[n + b] ^= row[a]
    elif name == "cx":
        c, t = wires
        row[t] ^= row[c]
        row[n + c] ^= row[n + t]


def _conjugate_word(pauli, gates):
    """(sign, word) of C P C^dagger for the gate sequence, using exact local matrices."""
    from .statevector import PAULI
    sign, word = 1, dict(pauli)
    for name, wires in gates:
        local = [word.get(w, "I") for w in wires]
        if all(a == "I" for a in local):
            continue
        u = _GATES[name]
        op = PAULI[local[0]] if len(wires) == 1 else np.kron(PAULI[local[1]], PAULI[local[0]])
        image = u @ op @ u.conj().T
        s, new = _as_pauli(image, len(wires))
        sign *= s
        for w, a in zip(wires, new):
            if a == "I":
                word.pop(w, None)
            else:
                word[w] = a
    return sign, word


def _as_pauli(matrix, k):
    """(+-1, labels per wire) for a matrix equal to +- a Pauli word (first wire least significant)."""
    from .statevector import PAULI
    for labels in ([(a,) for a in "IXYZ"] if k == 1 else [(a, b) for a in "IXYZ" for b in "IXYZ"]):
        p = PAULI[labels[0]] if k == 1 else np.kron(PAULI[labels[1]], PAULI[labels[0]])
        overlap = np.trace(p @ matrix) / 2 ** k
        if abs(abs(overlap) - 1) < 1e-9:
            return int(round(overlap.real)), labels
    raise ArithmeticError("not a Pauli word")


_H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
# Two-wire matrices index bit(first wire) + 2 bit(second wire); cx's control is the first wire.
_GATES = {"h": _H, "s": np.diag([1, 1j]),
          "cz": np.diag([1, 1, 1, -1]).astype(complex),
          "cx": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], complex)}


def append_clifford(circuit, gates):
    """A copy of `circuit` followed by the Clifford's gates (ideal, noise-free), for sampling.

    s is RZ(pi/2) and cx is H CZ H on the target, equal up to global phase.
    """
    out = Circuit(circuit.n_qubits, circuit.n_parameters, circuit.max_terms)
    out._impl.stages = list(circuit._impl.stages)
    out._impl.readout = dict(circuit._impl.readout)
    for name, wires in gates:
        if name == "h":
            out.gate("H", wires)
        elif name == "s":
            out.gate("RZ", wires, theta=math.pi / 2)
        elif name == "cz":
            out.gate("CZ", wires)
        else:
            out.gate("H", wires[1:]).gate("CZ", wires).gate("H", wires[1:])
    return out


# ---------------------------------------------------------------- classical shadows
def shadows(circuit, observables, snapshots, *, median_of_means=1, seed=None, values=None):
    """Classical-shadow estimates of Pauli observables (Huang, Kueng and Preskill, arXiv:2002.08953).

    Each snapshot measures every wire in a uniformly random Pauli basis: one
    shot of the circuit followed by an ideal basis change (H for X, S^dagger
    then H for Y), sampled with `statevector.sample` (noise and readout
    included; noisy circuits through the density matrix or trajectories).
    A word P is estimated from a snapshot as prod_{q in P} 3 b_q [basis_q = P_q],
    with b_q = +-1 the outcome, so 3^|P| times the outcome parity when every
    basis matches and 0 otherwise. Observables are specs `parse_observable`
    reads (sums allowed). With median_of_means = K > 1 the snapshots are split
    into K batches and the median of the batch means is returned.
    Returns per observable the estimate and the empirical standard error
    (sample standard deviation of the per-snapshot estimates / sqrt(snapshots)).
    """
    from .interop import parse_observable
    circuit, shifts, wires = _sampling_target(circuit, values)
    n = len(wires)
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 3, size=(snapshots, n))     # 0 = X, 1 = Y, 2 = Z per logical wire
    outcomes = np.empty((snapshots, n), dtype=np.int8)
    unique, inverse = np.unique(bases, axis=0, return_inverse=True)
    for index, basis in enumerate(unique):
        rows = np.flatnonzero(inverse.reshape(-1) == index)
        bits = _sample_basis(circuit, shifts, wires, basis, len(rows), int(rng.integers(2 ** 62)))
        rng.shuffle(bits)
        outcomes[rows] = 1 - 2 * bits
    results = []
    for spec in observables:
        terms = parse_observable(spec, n)
        per_snapshot = sum(c * _shadow_word(p, bases, outcomes) for c, p in terms)
        per_snapshot = np.broadcast_to(per_snapshot, (snapshots,)).astype(float)
        batches = np.array_split(per_snapshot, median_of_means)
        estimate = float(np.median([b.mean() for b in batches])) if median_of_means > 1 else float(per_snapshot.mean())
        results.append({"observable": spec, "estimate": estimate,
                        "standard_error": float(per_snapshot.std(ddof=1) / math.sqrt(snapshots))})
    return {"snapshots": int(snapshots), "median_of_means": int(median_of_means), "results": results}


def _sampling_target(circuit, values):
    """(Circuit or Compiled, shifts, wires that bitstrings report) for sampling."""
    if hasattr(circuit, "final_layout"):
        return circuit, values, list(range(len(circuit.final_layout)))
    from .interop import convert
    converted = convert(circuit, values=values)
    shifts = converted.shifts(values) if converted.parameters else None
    return converted.circuit, shifts, list(range(converted.circuit.n_qubits))


def _sample_basis(circuit, shifts, wires, basis, shots, seed):
    """(shots, wires) array of 0/1 outcomes after rotating each wire into its basis."""
    from .statevector import sample, sample_compiled
    compiled = hasattr(circuit, "final_layout")
    base = circuit.circuit if compiled else circuit
    physical = [circuit.final_layout[q] for q in wires] if compiled else wires
    gates = []
    for q, b in zip(physical, basis):
        if b == 1:
            gates.append(("s_dagger", (q,)))
        if b in (0, 1):
            gates.append(("h", (q,)))
    rotated = Circuit(base.n_qubits, base.n_parameters, base.max_terms)
    rotated._impl.stages = list(base._impl.stages)
    rotated._impl.readout = dict(base._impl.readout)
    for name, (q,) in gates:
        rotated.gate("RZ", [q], theta=-math.pi / 2) if name == "s_dagger" else rotated.gate("H", [q])
    if compiled:
        raw = sample_compiled(replace(circuit, circuit=rotated), shots, values=shifts, seed=seed)
    else:
        raw = sample(rotated, shots, shifts=shifts, seed=seed)
    width = len(wires)
    bits = [[int(b[width - 1 - q]) for q in wires] for b, count in raw["counts"].items() for _ in range(count)]
    return np.array(bits, dtype=np.int8)


def _shadow_word(pauli, bases, outcomes):
    """Per-snapshot estimate of a Pauli word: prod over its wires of 3 b_q [basis_q = axis]."""
    code = {"X": 0, "Y": 1, "Z": 2}
    value = np.ones(len(bases))
    for q, axis in pauli.items():
        value *= 3.0 * outcomes[:, q] * (bases[:, q] == code[axis])
    return value
