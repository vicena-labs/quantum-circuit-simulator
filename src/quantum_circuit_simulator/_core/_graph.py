"""Bounded, reusable Pauli arithmetic graphs with reverse differentiation.

This backend compiles structural paths, not answers at one angle, and does not
prune small intermediate coefficients. Preparation can consequently exceed a
budget even when one particular numeric binding is cheap; callers may then use
the ordinary bounded propagation backend explicitly.

Observables are grouped into backward cones exactly as for the native solver.
The C++ compiler records each group's propagation as edges between scalar
nodes; evaluation is one pass over the edges and reverse differentiation one
pass back, against the same bound stage matrices the native solver uses.

Optional truncation chooses the discarded terms from coefficients at the angles
bound when the graph is compiled, and keeps each discarded term as a node.
Every evaluation bounds the error from those terms' coefficients at its own
angles, so the reported bound holds at any binding; it grows when the
parameters move to where the discarded terms matter. When it grows past
`reprepare_factor` times its value where the graph was compiled, the graph is
compiled again at the current angles and the evaluation repeated.
"""
from __future__ import annotations

import ctypes as ct
import math
import numbers
import weakref
import numpy as np

from ._indexed import ConeGroups
from ._native import CGraphStats, CGraphSupport, library
from ._reference import ConeOverflow, P, P2


class GraphLimitExceeded(ConeOverflow):
    """Structural graph preparation exceeded a declared resource budget."""


def _fourier(stage):
    """Exact analytic support in [1, cos(t), sin(t), cos(2t), sin(2t)].

    The support is obtained from coefficient matrices, never sampled angles.
    Tiny nonzero Fourier coefficients are retained conservatively.
    """
    if stage.name in ("RX", "RY", "RZ"):
        axis = {"RX": 1, "RY": 3, "RZ": 2}[stage.name]
        result = np.zeros((5, 4, 4))
        result[0, 0, 0] = result[0, axis, axis] = 1.0
        generator = P[axis]
        for row in range(1, 4):
            if row == axis:
                continue
            result[1, row, row] = 1.0
            # d/dt [U(t)^dagger P U(t)] at t=0 = i [G,P] / 2.
            tangent = .5j * (generator @ P[row] - P[row] @ generator)
            for column in range(4):
                result[2, row, column] = np.trace(P[column] @ tangent).real / 2
        return result
    if stage.name == "RZZ":
        # Like a rotation on 16 two-qubit Paulis: those commuting with Z.Z stay,
        # the others turn by the angle towards i[ZZ, P]/2.
        generator = np.kron(P[2], P[2])
        result = np.zeros((5, 16, 16))
        for row, observable in enumerate(P2):
            commutator = generator @ observable - observable @ generator
            if np.allclose(commutator, 0):
                result[0, row, row] = 1.0
                continue
            result[1, row, row] = 1.0
            for column, pauli in enumerate(P2):
                result[2, row, column] = np.trace(pauli @ (.5j * commutator)).real / 4
        return result
    if stage.name != "FSIM":
        raise ValueError("graph rebinding supports RX, RY, RZ, RZZ and FSIM")

    # U(t) = A + cos(t) B + sin(t) C. Form the five Fourier
    # coefficient matrices directly; this remains valid at zero angles and
    # for every fixed phi, including special values which cancel paths.
    a = np.diag([1, 0, 0, np.exp(-1j * stage.phi)])
    b = np.diag([0, 1, 1, 0]).astype(complex)
    c = np.zeros((4, 4), complex)
    c[1, 2] = c[2, 1] = -1j
    result = np.zeros((5, 16, 16))
    for row, observable in enumerate(P2):
        aa = a.conj().T @ observable @ a
        bb = b.conj().T @ observable @ b
        cc = c.conj().T @ observable @ c
        ab = a.conj().T @ observable @ b + b.conj().T @ observable @ a
        ac = a.conj().T @ observable @ c + c.conj().T @ observable @ a
        bc = b.conj().T @ observable @ c + c.conj().T @ observable @ b
        operators = (aa + (bb + cc) / 2, ab, ac, (bb - cc) / 2, bc / 2)
        for column, pauli in enumerate(P2):
            for frequency, operator in enumerate(operators):
                result[frequency, row, column] = np.trace(pauli @ operator).real / 4
    return result


def _budget(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _support(stages):
    """Structural support and constant-identity rows of every stage.

    Parameterised stages take support from their exact Fourier coefficients,
    so a path absent at one angle is kept for the others. Fixed stages keep the
    established 5e-15 transfer-entry roundoff filter. Stages sharing a gate
    family or a matrix share one packed record.
    """
    records, known = [], {}
    which = np.empty(len(stages), dtype=np.intp)
    for index, stage in enumerate(stages):
        if stage.param >= 0:
            key = (stage.name, stage.phi if stage.name == "FSIM" else None)
        else:
            key = np.asarray(stage.matrix, dtype=float).tobytes()
        if key not in known:
            known[key] = len(records)
            records.append(_support_record(stage))
        which[index] = known[key]
    packed = (CGraphSupport * max(1, len(stages)))()
    if len(stages):
        rows = np.array(records, dtype=np.uint16)[which]
        ct.memmove(packed, rows.ctypes.data, rows.nbytes)
    return packed


def _support_record(stage):
    """One CGraphSupport as 20 uint16: 16 row masks, the identity mask, padding."""
    if stage.param >= 0:
        coefficients = _fourier(stage)
        support = np.any(coefficients != 0, axis=0)
        constant = np.zeros_like(coefficients)
        constant[0] = np.eye(len(support))
        identity = [np.array_equal(coefficients[:, r], constant[:, r]) for r in range(len(support))]
    else:
        matrix = np.asarray(stage.matrix, dtype=float)
        support = np.abs(matrix) > 5e-15
        identity = [support[r].sum() == 1 and matrix[r, r] == 1. for r in range(len(support))]
    record = [0] * 20
    for row, columns in enumerate(support):
        record[row] = int(np.dot(columns, 1 << np.arange(len(columns))))
    record[16] = sum(1 << row for row, flag in enumerate(identity) if flag)
    return record


_FAILURES = {-1: "structural cone exceeds max_terms={terms}",
             -7: "structural graph exceeds max_nodes={nodes}",
             -8: "structural graph exceeds max_edges={edges}"}


class GraphPrepared(ConeGroups):
    """Compile grouped observable cones into reusable scalar arithmetic.

    Parameters are additive shifts from base angles, matching NativePrepared.
    This internal class expects an owned, validated NoisyPauliCircuit snapshot.
    The public Prepared wrapper supplies snapshot and concurrency protection.
    """

    def __init__(self, circuit, observables, *, max_nodes=200_000, max_edges=1_000_000,
                 initial="zero", threads=1, max_cone_qubits=256, stage_budgets=None,
                 truncate_below=0., max_weight=-1, reprepare_factor=None):
        if initial not in ("zero", "plus"):
            raise ValueError("graph supports zero/plus product input")
        self.initial = initial
        self.max_nodes = _budget(max_nodes, "max_nodes")
        self.max_edges = _budget(max_edges, "max_edges")
        try:
            super().__init__(circuit, observables, max_cone_qubits=max_cone_qubits,
                             stage_budgets=stage_budgets, threads=threads,
                             truncate_below=truncate_below, max_weight=max_weight)
        except GraphLimitExceeded:
            raise
        except ConeOverflow as exc:
            raise GraphLimitExceeded(f"graph cone preparation exceeded its budget: {exc}") from exc
        self._support = _support(circuit.stages)
        parameterised = [i for i, s in enumerate(circuit.stages) if s.param >= 0]
        self._params = np.array([circuit.stages[i].param for i in parameterised], dtype=int)
        self._angles = np.array([circuit.stages[i].angle for i in parameterised], dtype=float)
        # Only a coefficient threshold depends on the angles; the terms a
        # weight limit discards are the same at every binding.
        self.reprepare_factor = reprepare_factor if self.truncate_below > 0 else None
        self.reprepared = 0
        self.reprepare_failure = None
        self._compile()
        self.bind([0.0] * circuit.nparams)

    def _initial_terms(self, coefficient, axes):
        # No coefficient cutoff: the graph must stay valid at every binding.
        # One- and two-wire Z words expand readout directly, dropping only
        # exact zeros; the compiler then contracts idle wires for either input.
        if 1 <= len(axes) <= 2 and all(axis == "Z" for axis in axes.values()):
            return super()._initial_terms(coefficient, axes)
        try:
            terms = self.c.initial_terms([(coefficient, axes)], initial=self.initial,
                                         gradients=False, coefficient_cutoff=0.0)
        except ConeOverflow as exc:
            raise GraphLimitExceeded(
                f"graph observable initialization exceeded its term budget: {exc}") from exc
        if len(terms) > self.c.max_terms:
            raise GraphLimitExceeded("graph observable initialization exceeds max_terms")
        return {key: float(value[0]) for key, value in terms.items()}

    def _compile(self):
        lib = library()
        while True:
            handle, stats = ct.c_void_p(), CGraphStats()
            codes = (ct.c_int32 * max(1, len(self.groups)))()
            code = lib.pauli_graph_compile(
                self.c.nparams, self.full.stages, len(self.full.stages), self._support,
                len(self.groups), self._groups, 0 if self.initial == "zero" else 1,
                self.c.max_terms, self.max_nodes, self.max_edges, self.truncate_below,
                self.max_weight, self.lipschitz, self.threads, ct.byref(handle), ct.byref(stats),
                codes)
            # Joint groups that outgrew the cap: split them all into single cones.
            if code == -1 and self._split_overflows(codes[:len(self.groups)]):
                continue
            break
        if code in _FAILURES:
            raise GraphLimitExceeded(_FAILURES[code].format(
                terms=self.c.max_terms, nodes=self.max_nodes, edges=self.max_edges))
        if code == -3:
            raise MemoryError("graph compilation could not allocate its arrays")
        if code == -6:
            raise RuntimeError("graph did not reach the product input boundary")
        if code:
            raise RuntimeError(f"graph compilation failed with code {code}")
        if getattr(self, "_finalizer", None):
            self._finalizer()  # free the graph this one replaces
        self._handle = handle
        self._finalizer = weakref.finalize(self, lib.pauli_graph_free, handle)
        self._counts = stats
        # The value bound where the graph was compiled, against which later
        # bounds are compared.
        self._prepared_bound = self._run(False)[2][0] if self.truncating else 0.

    @property
    def stats(self):
        return {
            "mode": "structural_graph_reverse_ad",
            "observables": len(self.observables),
            "cone_groups": len(self.groups),
            "nodes": int(self._counts.nodes),
            "edges": int(self._counts.edges),
            "max_single_cone_terms": int(self._counts.peak_terms),
            "array_bytes": int(self._counts.bytes),
            "array_bytes_excludes_python_metadata": True,
            "coefficient_pruning": self.truncating,
            **({"graph_reprepared": self.reprepared, "reprepare_failed": self.reprepare_failure}
               if self.reprepare_factor else {}),
        }

    def bind(self, shifts):
        shifts = tuple(shifts)
        if len(shifts) != self.c.nparams or not all(math.isfinite(x) for x in shifts):
            raise ValueError("one finite shift per shared parameter required")
        # Finite base angles plus finite shifts overflow only for shifts near
        # the float limit, so the vector check runs only then.
        if (len(self._params) and max(map(abs, shifts), default=0.) > 1e300
                and not np.all(np.isfinite(self._angles + np.asarray(shifts)[self._params]))):
            raise ValueError("bound angle must be finite")
        self.full.bind(shifts)
        return self

    def value(self, shifts):
        """The value at validated shifts, for an exact graph."""
        self.bind(shifts)
        return self._run(False)[0]

    def evaluate(self, *, with_gradients=True, include_stats=False):
        value, gradient, bounds = self._run(with_gradients)
        if self._outgrown(bounds[0]):
            try:
                self._compile()  # at the angles bound now
            except GraphLimitExceeded as exc:
                # The current graph stays valid and its bound honest; stop
                # retrying a compilation that cannot fit.
                self.reprepare_failure = str(exc)
            else:
                self.reprepared += 1
                value, gradient, bounds = self._run(with_gradients)
        if not include_stats:
            return value, gradient
        facts = self.stats
        if self.truncating:
            facts.update(self._bound_facts(value, gradient, bounds[0],
                                           bounds[1:] if with_gradients else None,
                                           self._counts.truncated_terms))
        return value, gradient, facts

    def _outgrown(self, bound):
        """Whether the value bound has grown past reprepare_factor times its
        value at compilation; bounds below one threshold's worth never count."""
        return (self.reprepare_factor is not None and self.reprepare_failure is None
                and bound > self.reprepare_factor * max(self._prepared_bound, self.truncate_below))

    def _run(self, with_gradients):
        width = self.c.nparams + 1 if with_gradients else 1
        # Output buffers are reused: allocating them was a visible share of
        # a small circuit's evaluation.
        if getattr(self, "_buffers", None) is None:
            self._buffers = (ct.c_double * (self.c.nparams + 1))(), (ct.c_double * (self.c.nparams + 1))()
        out, bounds = self._buffers
        code = library().pauli_graph_evaluate(
            self._handle, self.c.nparams, self.full.stages, len(self.full.stages),
            int(with_gradients), self.threads, out, bounds)
        if code == -5:
            raise ArithmeticError("nonfinite graph result")
        if code:
            raise RuntimeError(f"graph evaluation failed with code {code}")
        return float(out[0]), np.array(out[1:width]), list(bounds)
