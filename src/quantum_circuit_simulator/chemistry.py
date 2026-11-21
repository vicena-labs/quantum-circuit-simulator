"""Molecular Hamiltonians for VQE, QPE and sampling studies: PySCF to tapered qubit operators.

`molecule` runs restricted Hartree-Fock in PySCF, takes an optional active
space, writes the second-quantized Hamiltonian over spin orbitals, maps it to
qubits (Jordan-Wigner, Bravyi-Kitaev or parity) and, on request, removes the
qubits fixed by Z2 symmetries. `hartree_fock` and `uccsd` build the reference
state and the standard ansatz as Programs that `Hamiltonian.expectation`,
`algorithms.vqe` and `hamiltonians.trotter(initial=...)` accept.

    from quantum_circuit_simulator import chemistry
    h2 = chemistry.molecule("H2")
    ansatz = chemistry.uccsd(h2.n_qubits, h2.n_electrons)
    h2.hamiltonian.expectation(ansatz, values={p: 0.0 for p in ansatz.parameters})  # = h2.energies["hf"]

Conventions. Spin orbitals interleave spin: mode 2p is spatial orbital p
with spin alpha and mode 2p+1 the same orbital with spin beta, orbitals in
increasing energy, so Hartree-Fock fills the lowest modes. Mode j is wire j
before mapping. A fermionic operator is a dict {((mode, dagger), ...):
coefficient}, the ladder operators multiplied left to right, dagger 1 for a
creation and 0 for an annihilation operator. Bitstrings list wire 0 first.
Lengths are in Angstrom and energies in Hartree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hamiltonians import Hamiltonian, pauli_rotation
from .interop import Angle, ConversionError, Program

MAPPINGS = ("jordan-wigner", "bravyi-kitaev", "parity")

# Experimental equilibrium geometries (NIST CCCBDB), in Angstrom.
MOLECULES = {
    "H2": "H 0 0 0; H 0 0 0.7414",
    "LiH": "Li 0 0 0; H 0 0 1.5949",
    "H2O": "O 0 0 0; H 0 0.75695 0.58588; H 0 -0.75695 0.58588",   # r = 0.9572, angle 104.52 degrees
    "BeH2": "Be 0 0 0; H 0 0 1.3264; H 0 0 -1.3264",
    "N2": "N 0 0 0; N 0 0 1.0977",
}


@dataclass
class Tapering:
    """Qubits removed by Z2 symmetries (Bravyi, Gambetta, Mezzacapo and Temme, arXiv:1701.08213).

    generators are the Z-type Pauli symmetries {wire: "Z"} of the untapered
    Hamiltonian; generator k is rotated onto X on wire removed[k] and that
    X replaced by its eigenvalue sector[k] (+1 or -1). kept lists the
    untapered wires that become wires 0, 1, ... of the tapered Hamiltonian.
    discarded is the l1 weight of terms below the tolerance, dropped because
    they are numerical noise that would otherwise break the symmetries.
    """
    hamiltonian: Hamiltonian
    generators: list
    removed: list
    sector: list
    kept: list
    discarded: float = 0.0

    def reduce(self, bits):
        """A bitstring of the untapered wires, restricted to the kept wires.

        The Clifford rotation turns a computational basis state into an X
        eigenstate on the removed wires only, so a reference bitstring in the
        chosen sector stays a bitstring on the kept wires.
        """
        return "".join(bits[q] for q in self.kept)


@dataclass
class Molecule:
    """A molecular Hamiltonian ready for circuits, with its references and provenance.

    hamiltonian is circuit-ready: tapered when tapering is set, else equal to
    untapered. Its identity term carries the nuclear repulsion and the frozen
    core energy. hf_occupation gives the Hartree-Fock occupation of each spin
    orbital; hf_bitstring the qubit values that prepare it in this mapping
    (after tapering, if any), wire 0 first. energies holds "hf" and, when
    PySCF finds it cheap, "fci" (full space) or "casci" (active space).
    """
    hamiltonian: Hamiltonian
    untapered: Hamiltonian
    n_electrons: int
    n_alpha: int
    n_beta: int
    mapping: str
    hf_occupation: tuple
    hf_bitstring: str
    energies: dict
    tapering: Tapering | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def n_qubits(self):
        return self.hamiltonian.n_qubits

    def hf_program(self):
        """The Program preparing the Hartree-Fock state on the circuit-ready Hamiltonian's wires."""
        return _x_program(self.hf_bitstring)

    def exact_energy(self):
        """The lowest eigenvalue with this molecule's electron numbers, by dense diagonalisation."""
        if self.tapering is not None:
            return self.hamiltonian.ground_state_energy()
        return sector_ground_energy(self.untapered, self.n_alpha, self.n_beta, self.mapping)


def molecule(geometry, basis="sto-3g", charge=0, spin=0, active=None, mapping="jordan-wigner", taper=False):
    """A molecule's qubit Hamiltonian from PySCF, with references for checking a solver.

    geometry is a name in MOLECULES, a PySCF atom string ("H 0 0 0; H 0 0
    0.74") or a list of (symbol, (x, y, z)), in Angstrom. spin is
    n_alpha - n_beta; open shells use ROHF orbitals. active is (n_electrons,
    n_orbitals): the lowest remaining orbitals are frozen doubly occupied and
    their energy folds into the constant, as in CASCI. taper=True removes the
    qubits fixed by Z-type Z2 symmetries, in the sector of the Hartree-Fock state.
    """
    try:
        import pyscf
        from pyscf import ao2mo, gto, mcscf, scf
    except ImportError as exc:
        raise ImportError('molecular Hamiltonians need PySCF: pip install "quantum-circuit-simulator[chemistry]"') \
            from exc
    if mapping not in MAPPINGS:
        raise ConversionError(f"unknown mapping {mapping!r}", "choose from " + ", ".join(MAPPINGS))
    atoms = MOLECULES.get(geometry, geometry) if isinstance(geometry, str) else [
        (symbol, tuple(float(x) for x in xyz)) for symbol, xyz in geometry]
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, unit="Angstrom", verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run()
    n_electrons, n_orbitals = active or (mol.nelectron, mf.mo_coeff.shape[1])
    n_alpha, n_beta = (n_electrons + spin) // 2, (n_electrons - spin) // 2
    cas = mcscf.CASCI(mf, n_orbitals, (n_alpha, n_beta))
    one_body, constant = cas.get_h1eff()
    two_body = ao2mo.restore(1, cas.get_h2eff(), n_orbitals)
    full = molecular_hamiltonian(constant, one_body, two_body, mapping)
    occupation = hf_occupation(2 * n_orbitals, n_alpha, n_beta)
    bits = "".join(str(b) for b in encode(occupation, mapping))
    energies = {"hf": float(mf.e_tot)}
    if n_orbitals <= 10:
        energies["fci" if active is None else "casci"] = float(cas.kernel()[0])
    tapering = taper_z2(full, bits) if taper else None
    return Molecule(
        hamiltonian=tapering.hamiltonian if tapering else full, untapered=full, n_electrons=n_electrons,
        n_alpha=n_alpha, n_beta=n_beta, mapping=mapping, hf_occupation=occupation,
        hf_bitstring=tapering.reduce(bits) if tapering else bits, energies=energies, tapering=tapering,
        provenance={"geometry": [(mol.atom_symbol(i), tuple(float(x) for x in mol.atom_coord(i, unit="Angstrom")))
                                 for i in range(mol.natm)],
                    "basis": basis, "charge": charge, "spin": spin, "active": active, "mapping": mapping,
                    "frozen_core_orbitals": cas.ncore, "method": "RHF" if spin == 0 else "ROHF",
                    "pyscf": pyscf.__version__})


def molecular_hamiltonian(constant, one_body, two_body, mapping="jordan-wigner"):
    """The qubit Hamiltonian of spatial-orbital integrals.

    H = constant + sum_pq,s h_pq a+_ps a_qs + 1/2 sum_pqrs,st (pq|rs) a+_ps a+_rt a_st a_qs,
    with one_body h and two_body (pq|rs) in chemists' notation, as PySCF returns them.
    """
    return to_qubits(molecular_operator(constant, one_body, two_body), 2 * len(one_body), mapping, "molecular")


def molecular_operator(constant, one_body, two_body):
    """The fermionic operator of spatial-orbital integrals over interleaved spin orbitals, normal ordered."""
    op = {(): complex(constant)}
    for p, q in zip(*np.nonzero(one_body)):
        for s in (0, 1):
            _accumulate(op, ((2 * p + s, 1), (2 * q + s, 0)), one_body[p, q])
    for p, q, r, t in zip(*np.nonzero(two_body)):
        for s in (0, 1):
            for u in (0, 1):
                if (p, s) != (r, u) and (q, s) != (t, u):
                    _accumulate(op, ((2 * p + s, 1), (2 * r + u, 1), (2 * t + u, 0), (2 * q + s, 0)),
                                0.5 * two_body[p, q, r, t])
    return normal_ordered(op)


def normal_ordered(op):
    """An equal operator with every term normal ordered: creations first, each group in descending mode.

    Reordering uses a_i a+_j = delta_ij - a+_j a_i and the anticommutation of
    like operators; a term repeating a ladder operator vanishes.
    """
    out = {}
    stack = [(tuple(word), complex(c)) for word, c in op.items()]
    while stack:
        word, c = stack.pop()
        for k in range(len(word) - 1):
            (i, di), (j, dj) = word[k], word[k + 1]
            if (di, i) < (dj, j):          # out of order: creation after annihilation, or ascending mode
                swapped = word[:k] + (word[k + 1], word[k]) + word[k + 2:]
                stack.append((swapped, -c))
                if i == j and di != dj:
                    stack.append((word[:k] + word[k + 2:], c))
                break
            if (i, di) == (j, dj):
                break                       # a+_i a+_i = a_i a_i = 0
        else:
            _accumulate(out, word, c)
    return {w: c for w, c in out.items() if abs(c) > 1e-14}


def _accumulate(op, word, c):
    op[word] = op.get(word, 0.0) + complex(c)


# ---------------------------------------------------------------- mappings
def to_qubits(op, n_modes, mapping="jordan-wigner", name="fermionic", tol=1e-12):
    """The Hamiltonian of a Hermitian fermionic operator under a linear encoding.

    Every mapping stores a linear function of the occupations on each qubit,
    b = A n (mod 2): Jordan-Wigner the occupations, parity their prefix sums,
    and Bravyi-Kitaev the Fenwick-tree partial sums of Seeley, Richard and Love
    (arXiv:1208.5986). Arithmetic is exact and complex; the result must be
    real, and terms below tol are dropped.
    """
    ladders = _ladders(n_modes, mapping)
    total = {}
    for word, c in op.items():
        product = {(0, 0): complex(c)}
        for mode, dagger in word:
            product = _multiply(product, ladders[mode][dagger])
        for key, value in product.items():
            total[key] = total.get(key, 0.0) + value
    terms = []
    for (x, z), c in total.items():
        c *= (-1j) ** bin(x & z).count("1")       # X Z = -i Y on each wire holding both
        if abs(c) <= tol:
            continue
        if abs(c.imag) > 1e-10:
            raise ConversionError(f"the mapped operator has an imaginary coefficient {c}; it is not Hermitian")
        pauli = {q: "Y" if x >> q & z >> q & 1 else "X" if x >> q & 1 else "Z"
                 for q in range(n_modes) if (x | z) >> q & 1}
        terms.append((c.real, pauli))
    return Hamiltonian(n_modes, terms, name)


def _ladders(n_modes, mapping):
    """Each mode's (annihilation, creation) operator as {(x, z): coefficient}, words X^x Z^z.

    a+_j flips every qubit storing n_j (its update set and j), keeps states
    with n_j = 0, and takes the sign of the occupations below j:
    a+_j = X_{U+j} Z_P (1 + Z_j Z_F) / 2, where P holds the qubits whose sum
    is the parity of modes < j and F the other qubits that n_j depends on.
    """
    a = _encoding(n_modes, mapping)
    inverse = _gf2_inverse(a)
    out = []
    for j in range(n_modes):
        flips = _mask(a[:, j])
        parity = _mask(inverse[:j].sum(axis=0) % 2)
        occupation = _mask(inverse[j])          # includes j: Z_j Z_F
        out.append(({(flips, parity): 0.5, (flips, parity ^ occupation): -0.5},
                    {(flips, parity): 0.5, (flips, parity ^ occupation): 0.5}))
    return out


def _encoding(n_modes, mapping):
    """The binary matrix A with qubit values b = A n (mod 2), unit lower triangular for every mapping."""
    if mapping == "jordan-wigner":
        return np.eye(n_modes, dtype=np.int64)
    if mapping == "parity":
        return np.tril(np.ones((n_modes, n_modes), dtype=np.int64))
    if mapping == "bravyi-kitaev":
        a = np.zeros((n_modes, n_modes), dtype=np.int64)
        for i in range(n_modes):
            low = (i + 1) & -(i + 1)            # qubit i stores modes i - low + 1 .. i
            a[i, i - low + 1:i + 1] = 1
        return a
    raise ConversionError(f"unknown mapping {mapping!r}", "choose from " + ", ".join(MAPPINGS))


def _gf2_inverse(a):
    n = len(a)
    work = np.concatenate([a % 2, np.eye(n, dtype=np.int64)], axis=1)
    for col in range(n):
        pivot = col + int(np.argmax(work[col:, col]))
        work[[col, pivot]] = work[[pivot, col]]
        for row in range(n):
            if row != col and work[row, col]:
                work[row] ^= work[col]
    return work[:, n:]


def _mask(bits):
    return sum(1 << q for q, b in enumerate(bits) if b)


def _multiply(left, right):
    """Product of two operators in the X^x Z^z basis: Z^b X^c = (-1)^|b & c| X^c Z^b."""
    out = {}
    for (x1, z1), c1 in left.items():
        for (x2, z2), c2 in right.items():
            key = (x1 ^ x2, z1 ^ z2)
            sign = -1 if bin(z1 & x2).count("1") % 2 else 1
            out[key] = out.get(key, 0.0) + sign * c1 * c2
    return out


def encode(occupation, mapping="jordan-wigner"):
    """The qubit values b = A n (mod 2) of a mode occupation in a mapping."""
    return tuple(int(b) for b in _encoding(len(occupation), mapping) @ np.array(occupation, dtype=np.int64) % 2)


def hf_occupation(n_modes, n_alpha, n_beta):
    """Occupation of each interleaved spin orbital in the Hartree-Fock determinant."""
    return tuple(int(m // 2 < (n_alpha if m % 2 == 0 else n_beta)) for m in range(n_modes))


def sector_ground_energy(hamiltonian, n_alpha, n_beta, mapping="jordan-wigner"):
    """The lowest eigenvalue among states with n_alpha even-mode and n_beta odd-mode electrons.

    Every mapping takes occupation states to computational basis states, so
    the sector is a set of basis states, decoded through the inverse encoding.
    Wire 0 is the least significant bit of a basis index; at most 12 qubits.
    """
    n = hamiltonian.n_qubits
    inverse = _gf2_inverse(_encoding(n, mapping))
    indices = np.arange(2 ** n)
    bits = (indices[:, None] >> np.arange(n)) & 1
    occupations = bits @ inverse.T % 2
    keep = (occupations[:, 0::2].sum(axis=1) == n_alpha) & (occupations[:, 1::2].sum(axis=1) == n_beta)
    block = hamiltonian.matrix()[np.ix_(keep, keep)]
    return float(np.linalg.eigvalsh(block)[0])


# ---------------------------------------------------------------- tapering
def taper_z2(hamiltonian, reference, tol=1e-10):
    """Remove one qubit per independent Z-type Z2 symmetry, in the sector of a reference bitstring.

    A Z-type Pauli Z_v commutes with every term exactly when v is orthogonal
    (mod 2) to each term's X part, so the generators are a basis of that
    null space. Reduced row echelon form gives each generator a free wire
    q_k that no other generator touches; U_k = (X_{q_k} + Z_{v_k}) / sqrt 2
    rotates Z_{v_k} onto X_{q_k}, and a term P anticommuting with X_{q_k}
    becomes X_{q_k} P Z_{v_k}. The sector is the reference's eigenvalue
    (-1)^(v_k . b). reference lists wire 0 first, like Molecule.hf_bitstring.
    """
    n = hamiltonian.n_qubits
    kept_terms = [(c, p) for c, p in hamiltonian.terms if abs(c) > tol]
    discarded = float(sum(abs(c) for c, p in hamiltonian.terms if abs(c) <= tol))
    x_parts = np.array([[int(p.get(q) in ("X", "Y")) for q in range(n)] for _, p in kept_terms] or
                       [[0] * n], dtype=np.int64)
    generators, removed = _gf2_null_space(x_parts)
    bits = [int(b) for b in reference]
    sector = [(-1) ** (int(v @ bits) % 2) for v in generators]
    terms = kept_terms
    for v, q in zip(generators, removed):
        z = {w: "Z" for w in range(n) if v[w]}
        terms = [_rotate(c, p, q, z) for c, p in terms]
    kept = [w for w in range(n) if w not in removed]
    position = {w: i for i, w in enumerate(kept)}
    tapered = []
    for c, p in terms:
        for q, s in zip(removed, sector):
            if q in p:
                assert p[q] == "X", "a rotated term is not diagonal on a removed wire"
                c *= s
        tapered.append((c, {position[w]: a for w, a in p.items() if w in position}))
    return Tapering(Hamiltonian(len(kept), tapered, f"{hamiltonian.name}-tapered"),
                    [{w: "Z" for w in range(n) if v[w]} for v in generators], removed, sector, kept, discarded)


def _gf2_null_space(matrix):
    """A basis of {v : matrix v = 0 mod 2}, each vector with its own free column set to 1."""
    work = matrix.copy() % 2
    rows, cols = work.shape
    pivots, r = [], 0
    for col in range(cols):
        hits = np.nonzero(work[r:, col])[0] if r < rows else []
        if len(hits) == 0:
            continue
        work[[r, r + hits[0]]] = work[[r + hits[0], r]]
        for row in range(rows):
            if row != r and work[row, col]:
                work[row] ^= work[r]
        pivots.append(col)
        r += 1
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.int64)
        v[f] = 1
        for row, p in enumerate(pivots):
            v[p] = work[row, f]
        basis.append(v)
    return basis, free


def _rotate(c, pauli, q, z):
    """U P U for U = (X_q + Z_v) / sqrt 2, where P commutes with Z_v: P, or X_q P Z_v if P anticommutes with X_q."""
    if pauli.get(q, "X") == "X":
        return c, pauli
    phase, product = _pauli_product({q: "X"}, pauli)
    phase2, product = _pauli_product(product, z)
    total = phase * phase2
    assert abs(total.imag) < 1e-12
    return c * total.real, product


_PRODUCT = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
            ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}


def _pauli_product(a, b):
    """(phase, word) with a b = phase * word."""
    phase, out = 1, dict(a)
    for q, axis in b.items():
        if q not in out:
            out[q] = axis
        elif out[q] == axis:
            del out[q]
        else:
            factor, out[q] = _PRODUCT[(out[q], axis)]
            phase *= factor
    return phase, out


# ---------------------------------------------------------------- circuits
def hartree_fock(n_qubits, n_electrons, mapping="jordan-wigner", spin=0):
    """The Program preparing the Hartree-Fock determinant: X on each qubit whose encoded value is 1."""
    occupation = hf_occupation(n_qubits, (n_electrons + spin) // 2, (n_electrons - spin) // 2)
    return _x_program("".join(str(b) for b in encode(occupation, mapping)))


def _x_program(bits):
    return Program(len(bits), [("x", [q], []) for q, b in enumerate(bits) if b == "1"])


def uccsd(n_qubits, n_electrons, mapping="jordan-wigner", spin=0, reference=True):
    """Trotterized UCCSD: the Hartree-Fock state, then exp(theta_k (T_k - T_k+)) for every excitation.

    Excitations conserve spin: singles i -> a and doubles (i, j) -> (a, b)
    from occupied to virtual spin orbitals. Each has one parameter, named
    s_i_a or d_i_j_a_b by mode, shared by the rotations of every Pauli string
    of its generator G_k = i (T_k - T_k+), since exp(theta (T - T+)) =
    exp(-i theta G). One excitation's strings commute, so each factor is
    exact; only the product over excitations is Trotterized.
    reference=False leaves out the Hartree-Fock gates.
    """
    occupation = hf_occupation(n_qubits, (n_electrons + spin) // 2, (n_electrons - spin) // 2)
    occupied = [m for m in range(n_qubits) if occupation[m]]
    virtual = [m for m in range(n_qubits) if not occupation[m]]
    excitations = [(f"s_{i}_{a}", ((a, 1), (i, 0))) for i in occupied for a in virtual if i % 2 == a % 2]
    excitations += [(f"d_{i}_{j}_{a}_{b}", ((a, 1), (b, 1), (j, 0), (i, 0)))
                    for i in occupied for j in occupied if i < j
                    for a in virtual for b in virtual if a < b and (i % 2 + j % 2) == (a % 2 + b % 2)]
    instructions = list(hartree_fock(n_qubits, n_electrons, mapping, spin).instructions) if reference else []
    for name, word in excitations:
        dagger = tuple((m, 1 - d) for m, d in reversed(word))
        generator = to_qubits({word: 1j, dagger: -1j}, n_qubits, mapping)
        for g, pauli in generator.terms:
            instructions += pauli_rotation(pauli, Angle(name, 2 * g))
    return Program(n_qubits, instructions, [name for name, _ in excitations])
