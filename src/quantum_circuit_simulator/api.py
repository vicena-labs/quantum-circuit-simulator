"""Validated public API. Prepared circuits own snapshots, not caller-owned buffers.

All parameter vectors are additive shifts from the circuit's base angles.
Only a zero product initial state is exposed in this first public API.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import numbers
import threading
from typing import Mapping, Sequence
import numpy as np
from ._core._reference import DEFAULT_MAX_TERMS, MAX_PARAMETERS, MAX_TERMS, NoisyPauliCircuit, ConeOverflow


# Hamiltonians of small molecules reach thousands of terms; each term costs one
# bit per stage in the cone pass, so the count is bounded.
MAX_OBSERVABLE_TERMS = 65536


def _integer(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class Truncation:
    """Approximate mode: discard Pauli terms while propagating.

    min_coefficient discards terms whose coefficient (or, for gradient
    evaluations, any derivative coefficient) is below this fraction of the
    observable's l1 norm. max_weight discards terms acting non-trivially on
    more than this many wires that are still open. Every evaluation reports a
    rigorous bound on the discarded contribution: a term c*P discarded at any
    point contributes c*Tr(rho P) to the exact value, and |Tr(rho P)| <= 1.

    reprepare_factor applies to the graph backend, which chooses discarded
    terms when it compiles: when an evaluation's value bound exceeds this
    multiple of the bound where the graph was compiled, the graph is compiled
    again at the current angles. None keeps the first graph.
    """
    min_coefficient: float = 0.0
    max_weight: int | None = None
    reprepare_factor: float | None = 2.0

    def __post_init__(self):
        coefficient = _real(self.min_coefficient, "min_coefficient")
        if not 0 <= coefficient < 1:
            raise ValueError("min_coefficient must be in [0, 1)")
        if self.max_weight is not None:
            _integer(self.max_weight, "max_weight", 0, 256)
        if self.reprepare_factor is not None and not _real(self.reprepare_factor, "reprepare_factor") > 1:
            raise ValueError("reprepare_factor must be greater than 1, or None")


@dataclass(frozen=True)
class Evaluation:
    """value and gradient; truncated plans also bound |exact - returned|.

    error_bound and gradient_error_bound are None for exact plans, which are
    exact up to floating-point roundoff.
    """
    value: float
    gradient: tuple[float, ...]
    backend: str
    error_bound: float | None = None
    gradient_error_bound: tuple[float, ...] | None = None
    # False when error_bound is no tighter than the trivial bound ||O||_1 + |value|:
    # the value is then supported only by convergence across truncations.
    bound_informative: bool | None = None

    def as_dict(self):
        out = {"value": self.value, "gradient": list(self.gradient), "backend": self.backend}
        if self.error_bound is not None:
            out["error_bound"] = self.error_bound
            out["bound_informative"] = self.bound_informative
        if self.gradient_error_bound is not None:
            out["gradient_error_bound"] = list(self.gradient_error_bound)
        return out


class Circuit:
    """Circuit builder for 1..16384 logical wires and 0..4096 shared parameters.

    A parameterised gate's angle is theta + scale * shift[param], so one shared
    parameter can drive many gates with different weights (QAOA's gamma * w_ij,
    a Trotter step's 2 h dt). The native backend computes gradients 16
    parameters per pass; the graph backend computes all of them in one reverse
    pass. max_terms limits each observable cone, not all process memory. This
    is a trusted local numerical library, not an isolation boundary for hostile input.
    """
    def __init__(self, n_qubits: int, n_parameters: int = 0, max_terms: int = DEFAULT_MAX_TERMS):
        self._impl = NoisyPauliCircuit(
            _integer(n_qubits, "n_qubits", 1, 16384),
            _integer(n_parameters, "n_parameters", 0, MAX_PARAMETERS),
            _integer(max_terms, "max_terms", 1, MAX_TERMS),
        )

    @property
    def n_qubits(self): return self._impl.nqubits
    @property
    def n_parameters(self): return self._impl.nparams
    @property
    def max_terms(self): return self._impl.max_terms

    def _wires(self, wires, arity=None):
        if isinstance(wires, (str, bytes)):
            raise ValueError("wires must be a sequence of wire indices")
        ws = tuple(_integer(q, "wire", 0, self.n_qubits-1) for q in wires)
        if len(ws) not in (1, 2) or len(set(ws)) != len(ws) or (arity is not None and len(ws) != arity):
            raise ValueError("wrong number of wires or duplicate wires")
        return ws

    def _param(self, param):
        return _integer(param, "param", -1, self.n_parameters-1)

    def gate(self, name: str, wires: Sequence[int], theta: float = 0.0, param: int = -1, scale: float = 1.0):
        if not isinstance(name, str): raise ValueError("gate name must be text")
        name = name.upper()
        one = {"H", "SX", "X", "Y", "Z", "RX", "RY", "RZ"}
        if name not in one | {"CZ", "ECR", "RZZ"}:
            raise ValueError("unsupported gate; use fsim() for fSim and unitary() for any other matrix")
        ws = self._wires(wires, 1 if name in one else 2)
        angle = _real(theta, "theta"); p = self._param(param); k = self._scale(scale, p)
        if name not in {"RX", "RY", "RZ", "RZZ"} and (p != -1 or angle != 0):
            raise ValueError("fixed gates do not accept an angle or parameter")
        self._impl.gate(name, ws, angle, p, k)
        return self

    def fsim(self, wires: Sequence[int], theta: float, phi: float, param: int = -1, scale: float = 1.0):
        p = self._param(param)
        self._impl.fsim(self._wires(wires, 2), _real(theta, "theta"), _real(phi, "phi"), p,
                        scale=self._scale(scale, p))
        return self

    def unitary(self, wires: Sequence[int], matrix):
        """Any fixed one- or two-qubit unitary (2x2 or 4x4).

        The first wire is the least significant bit of the matrix index, as for
        the native ECR; Qiskit's little-endian Operator.data uses the same order.
        """
        ws = self._wires(wires)
        m = np.asarray(matrix, dtype=complex)
        if m.shape != (2 ** len(ws),) * 2 or not np.all(np.isfinite(m)):
            raise ValueError(f"a {len(ws)}-qubit unitary needs a finite {2 ** len(ws)}x{2 ** len(ws)} matrix")
        self._impl.unitary(ws, m)
        return self

    def _scale(self, scale, param):
        k = _real(scale, "scale")
        if param != -1 and k == 0:
            raise ValueError("a parameter's scale must be nonzero")
        return k

    def pauli_error(self, wire: int, axis: str, probability: float):
        wire = self._wires((wire,), 1)[0]
        if not isinstance(axis, str): raise ValueError("axis must be X, Y or Z")
        self._impl.channel_pauli(wire, axis, _real(probability, "probability"))
        return self

    def depolarize(self, wires: Sequence[int], probability: float):
        """Total probability of uniformly selected NONIDENTITY Pauli errors.

        This is not Aer lambda: lambda = p * 4**k / (4**k - 1).
        """
        self._impl.channel_depolarize(self._wires(wires), _real(probability, "probability"))
        return self

    def relax(self, wire: int, duration: float, t1: float, t2: float):
        """Zero-temperature relaxation; duration, t1 and t2 use the SAME unit."""
        self._impl.channel_relax(self._wires((wire,), 1)[0], _real(duration, "duration"),
                                 _real(t1, "t1"), _real(t2, "t2"))
        return self

    def readout(self, wire: int, p01: float, p10: float):
        self._impl.set_readout(self._wires((wire,), 1)[0], _real(p01, "p01"), _real(p10, "p10"))
        return self

    def prepare(self, observables, *, backend: str = "native",
                graph_max_nodes: int = 2_000_000, graph_max_edges: int = 10_000_000,
                dense_max_qubits: int = 6, dense_max_bytes: int = 64*1024*1024,
                max_cone_qubits: int = 256, truncation: Truncation | None = None,
                threads: int = 1) -> Prepared:
        return Prepared(self, observables, backend=backend,
                        graph_max_nodes=graph_max_nodes, graph_max_edges=graph_max_edges,
                        dense_max_qubits=dense_max_qubits, dense_max_bytes=dense_max_bytes,
                        max_cone_qubits=max_cone_qubits, truncation=truncation, threads=threads)


class Prepared:
    """An owned prepared circuit. It can be rebound but not used concurrently.

    The caller may safely modify the original Circuit after this snapshot.
    truncation (native, graph and auto backends) selects the approximate mode;
    threads lets the native and graph backends evaluate independent cones in
    parallel, with results that do not depend on the thread count.
    A truncated graph chooses its discarded terms at the angles bound at
    preparation; its reported bound is computed at every evaluation's angles.
    """
    def __init__(self, circuit: Circuit, observables, *, backend: str = "native",
                 graph_max_nodes: int = 2_000_000, graph_max_edges: int = 10_000_000,
                 dense_max_qubits: int = 6, dense_max_bytes: int = 64*1024*1024,
                 max_cone_qubits: int = 256, truncation: Truncation | None = None,
                 threads: int = 1):
        if not isinstance(circuit, Circuit): raise TypeError("expected Circuit")
        from .backends import available_backends, get_backend_factory
        if backend not in available_backends():
            raise ValueError(f"unknown backend {backend!r}; see available_backends()")
        graph_max_nodes = _integer(graph_max_nodes, 'graph_max_nodes', 1, 10_000_000)
        graph_max_edges = _integer(graph_max_edges, 'graph_max_edges', 1, 50_000_000)
        dense_max_qubits = _integer(dense_max_qubits, 'dense_max_qubits', 1, 10)
        dense_max_bytes = _integer(dense_max_bytes, 'dense_max_bytes', 1, 2**31)
        max_cone_qubits = _integer(max_cone_qubits, 'max_cone_qubits', 1, 256)
        threads = _integer(threads, 'threads', 1, 256)
        if truncation is not None and not isinstance(truncation, Truncation):
            raise TypeError('truncation must be a Truncation or None')
        if truncation is not None and backend not in ('native', 'graph', 'auto'):
            raise ValueError('truncation is supported by the native, graph and auto backends')
        items = list(observables)
        if not 1 <= len(items) <= MAX_OBSERVABLE_TERMS: raise ValueError(f"provide 1..{MAX_OBSERVABLE_TERMS} observable terms")
        checked = []
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2: raise ValueError("observable is (coefficient, wire-to-Pauli mapping)")
            coef, ops = item
            coef = _real(coef, "coefficient")
            if not isinstance(ops, Mapping): raise ValueError("Pauli operators must be a mapping")
            mapping = {}
            for wire, axis in ops.items():
                q = _integer(wire, "observable wire", 0, circuit.n_qubits-1)
                if axis not in ("I", "X", "Y", "Z"): raise ValueError("Pauli axes must be I, X, Y or Z")
                if q in circuit._impl.readout and axis in ("X", "Y"):
                    raise ValueError("classical Z-readout errors do not define an X/Y readout model")
                if axis != "I": mapping[q] = axis
            checked.append((coef, mapping))
        # Canonicalize by physical Pauli word, independent of mapping order.
        # fsum prevents avoidable cancellation error when combining weights.
        grouped = {}
        for coefficient, mapping in checked:
            grouped.setdefault(tuple(sorted(mapping.items())), []).append(coefficient)
        canonical = []
        for word, coefficients in grouped.items():
            try:
                coefficient = math.fsum(coefficients)
            except OverflowError as exc:
                raise ValueError('combined observable coefficient overflows float64') from exc
            if coefficient != 0.:
                canonical.append((coefficient, dict(word)))
        self.requested_backend = backend
        self.backend = 'native' if backend == 'auto' else backend
        self._circuit = circuit._impl.snapshot()
        self._observables = tuple(canonical)
        self._lock = threading.Lock()
        self._plan = None
        self._truncation = truncation
        # Thresholds are relative to the whole observable, as for roundoff.
        norm = math.fsum(abs(c) for c, _ in canonical)
        self._native_options = dict(
            truncate_below=truncation.min_coefficient * norm if truncation else 0.,
            max_weight=truncation.max_weight if truncation and truncation.max_weight is not None else -1,
            threads=threads)
        self._dense_limits = dict(max_qubits=dense_max_qubits, max_bytes=dense_max_bytes)
        self._diagnostics = {
            'requested_backend': backend,
            'backend': self.backend,
            'input_observable_terms': len(items),
            'unique_observable_terms': len(self._observables),
            'fallback_reason': None,
            'certified_global_error_bound': False,
        }
        factory = get_backend_factory(backend)
        if self._observables and factory is not None:
            # Plugin receives owned data; never mutate the caller's Circuit.
            self._plan = factory(self._circuit, self._observables,
                                 options=dict(graph_max_nodes=graph_max_nodes,
                                              graph_max_edges=graph_max_edges,
                                              dense_max_qubits=dense_max_qubits,
                                              dense_max_bytes=dense_max_bytes))
            if not callable(getattr(self._plan, 'bind', None)) or not callable(getattr(self._plan, 'evaluate', None)):
                raise TypeError('registered backend plan must implement bind() and evaluate()')
            return
        if self._observables and circuit.n_qubits <= 256 and max_cone_qubits < circuit.n_qubits:
            widest = _widest_cone(self._circuit, self._observables)
            if widest > max_cone_qubits:
                raise ConeOverflow(f'backward cone needs {widest} qubits; max_cone_qubits={max_cone_qubits}')
        wide = circuit.n_qubits > 256
        # Wide registers bound the cones' stage selections before any work.
        budgets = dict(max_cone_qubits=max_cone_qubits, stage_budgets=(100_000, 500_000)) if wide else {}
        if self._observables and backend == 'graph':
            from ._core._graph import GraphPrepared, GraphLimitExceeded
            try:
                self._plan = GraphPrepared(self._circuit, self._observables,
                                           max_nodes=graph_max_nodes, max_edges=graph_max_edges,
                                           reprepare_factor=truncation.reprepare_factor if truncation else None,
                                           **budgets, **self._native_options)
                self.backend = 'wide-graph' if wide else 'graph'
                self._diagnostics.update(self._plan.stats, backend=self.backend)
            except GraphLimitExceeded as exc:
                self.backend = 'native'
                self._diagnostics.update(backend='native', fallback_reason=str(exc))
        if self._observables and wide and self._plan is None and (backend in ('native', 'graph') or truncation):
            # One shared bound circuit; each cone group is renumbered to fit.
            from ._core._indexed import IndexedBatchPrepared
            self.backend = 'wide-native'
            self._diagnostics['backend'] = self.backend
            self._plan = IndexedBatchPrepared(self._circuit, self._observables,
                                              **budgets, **self._native_options)
        if self._observables and wide and self._plan is not None:
            self._diagnostics.update(logical_qubits=circuit.n_qubits,
                                     max_cone_qubits=max((len(c.wires) for c in self._plan.cones), default=0),
                                     unique_cone_groups=len(self._plan.groups),
                                     total_prepared_cone_stages=sum(len(c.stages) for c in self._plan.cones))
            return
        if self._observables and wide:
            from ._core._wide import WidePrepared
            self.backend = 'wide-' + ('native' if backend == 'auto' else backend)
            self._plan = WidePrepared(self._circuit, self._observables,
                                      backend=backend, graph_max_nodes=graph_max_nodes,
                                      graph_max_edges=graph_max_edges,
                                      dense_max_qubits=dense_max_qubits, dense_max_bytes=dense_max_bytes,
                                      max_cone_qubits=max_cone_qubits)
            self._diagnostics.update(self._plan.stats, backend=self.backend)
            return
        if self._observables and backend == 'dense':
            self._make_dense()
        if self._observables and self.backend == "native":
            from ._core._indexed import IndexedBatchPrepared
            try:
                self._plan = IndexedBatchPrepared(self._circuit, self._observables,
                                                  **self._native_options)
            except ConeOverflow as exc:
                if backend != 'auto':
                    raise
                self._make_dense()
                self._diagnostics['fallback_reason'] = str(exc)

    @property
    def n_parameters(self): return self._circuit.nparams

    @property
    def diagnostics(self):
        """A copy of preparation/evaluation facts, including any backend fallback."""
        return copy.deepcopy(self._diagnostics)

    @property
    def numerics(self):
        """Declared numerical policy of the currently selected representation."""
        from .backends import get_backend_factory
        if get_backend_factory(self.requested_backend) is not None:
            return {'precision': 'backend-defined', 'certified_global_error_bound': False,
                    'max_terms_per_observable': None}
        if self.backend == 'wide-mixed':
            return {'precision': 'mixed-backend', 'certified_global_error_bound': False,
                    'max_terms_per_observable': self._circuit.max_terms}
        sparse = self.backend in ('native', 'python', 'wide-native', 'wide-python')
        truncated = self._truncation is not None and self.backend in ('native', 'wide-native', 'graph', 'wide-graph')
        return {
            'precision': 'complex128' if self.backend in ('dense', 'wide-dense') else 'float64',
            'transfer_entry_cutoff': 0. if self.backend in ('dense', 'wide-dense') else 5e-15,
            'parameterized_transfer_cutoff': 5e-15 if sparse else 0.,
            'coefficient_cutoff': 1e-16 if sparse else 0.,
            'coefficient_cutoff_relative_to': 'observable_l1_norm' if sparse else None,
            'certified_global_error_bound': False,
            'max_terms_per_observable': self._circuit.max_terms if self.backend not in ('dense', 'wide-dense') else None,
            'truncation': {'min_coefficient': self._truncation.min_coefficient,
                           'max_weight': self._truncation.max_weight,
                           'min_coefficient_relative_to': 'observable_l1_norm',
                           'discarded_terms_chosen': ('when the graph is compiled; again when the bound '
                                                      'exceeds reprepare_factor times its compiled value'
                                                      if self.backend.endswith('graph') else 'at every evaluation'),
                           'reprepare_factor': self._truncation.reprepare_factor if self.backend.endswith('graph') else None,
                           'error_bound': 'rigorous, reported per evaluation'} if truncated else None,
        }

    def _make_dense(self):
        from ._core._dense import DensePrepared
        self._plan = DensePrepared(self._circuit, self._observables, **self._dense_limits)
        self.backend = 'dense'
        self._diagnostics.update(self._plan.stats, backend='dense')

    def _shifts(self, shifts):
        values = [0.0]*self.n_parameters if shifts is None else list(shifts)
        if len(values) != self.n_parameters: raise ValueError("one finite shift per shared parameter is required")
        # Plain finite floats, the common case in an optimiser loop, need no
        # per-element conversion.
        if all(isinstance(x, float) and math.isfinite(x) for x in values):
            return tuple(map(float, values))
        return tuple(_real(x, "parameter shift") for x in values)

    def _evaluate_locked(self, shifts, gradients):
        bounded = self._truncation is not None
        if not self._observables:
            zero = (0.,)*self.n_parameters if gradients else ()
            return Evaluation(0., zero, self.backend, 0. if bounded else None,
                              zero if bounded and gradients else None, True if bounded else None)
        if self._plan is not None:
            self._plan.bind(shifts)
            try:
                value, gradient, stats = self._plan.evaluate(
                    with_gradients=gradients, include_stats=True)
            except ConeOverflow as exc:
                if self.requested_backend != 'auto' or self.backend != 'native':
                    raise
                self._make_dense()  # May fail closed on its own explicit limits.
                self._diagnostics['fallback_reason'] = str(exc)
                self._plan.bind(shifts)
                value, gradient, stats = self._plan.evaluate(
                    with_gradients=gradients, include_stats=True)
            self._diagnostics.update(stats)
            if self.backend.startswith('wide-') and 'backend' in stats:
                self.backend = stats['backend']
            if bounded:
                # A dense fallback is exact, so its bound is zero.
                error_bound = stats.get('error_bound', 0.)
                informative = stats.get('bound_informative', True)
                gradient_bound = stats.get('gradient_error_bound') or (0.,)*self.n_parameters

        else:
            bound = self._circuit.shifted(shifts)
            total = np.zeros(self.n_parameters+1 if gradients else 1)
            for term in self._observables:
                v, g = bound.evaluate([term], gradients=gradients)
                total += np.array([v, *g])
            value, gradient = total[0], total[1:]
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ArithmeticError("nonfinite result; no successful output returned")
        if not bounded:
            return Evaluation(float(value), tuple(float(x) for x in gradient), self.backend)
        return Evaluation(float(value), tuple(float(x) for x in gradient), self.backend,
                          float(error_bound),
                          tuple(float(x) for x in gradient_bound) if gradients else None,
                          bool(informative))

    def evaluate(self, shifts: Sequence[float] | None = None, *, gradients: bool = True) -> Evaluation:
        if not isinstance(gradients, bool): raise ValueError('gradients must be a boolean')
        values = self._shifts(shifts)
        if not self._lock.acquire(blocking=False): raise RuntimeError("a prepared circuit cannot be evaluated concurrently")
        try: return self._evaluate_locked(values, gradients)
        finally: self._lock.release()

    def evaluate_many(self, parameter_vectors, *, gradients: bool = True) -> list[Evaluation]:
        if not isinstance(gradients, bool): raise ValueError('gradients must be a boolean')
        rows = list(parameter_vectors)
        if not 1 <= len(rows) <= 128: raise ValueError("a batch must contain 1..128 vectors")
        values = [self._shifts(row) for row in rows]
        if not self._lock.acquire(blocking=False): raise RuntimeError("a prepared circuit cannot be evaluated concurrently")
        try: return [self._evaluate_locked(v, gradients) for v in values]
        finally: self._lock.release()

    def evaluate_value(self, shifts: Sequence[float] | None = None) -> float:
        """Evaluate a rebound parameter vector without propagating derivatives."""
        return self.evaluate(shifts, gradients=False).value

    def evaluate_values(self, parameter_vectors) -> list[float]:
        return [result.value for result in self.evaluate_many(parameter_vectors, gradients=False)]


def _widest_cone(circuit, observables):
    """Widest backward light cone, in wires, over all observables.

    One reverse pass with one bit per observable, as in the wide planner: after
    it, bit i of a wire's mask is set exactly when that wire is in cone i.
    """
    wire_queries = [0] * circuit.nqubits
    for index, (_, axes) in enumerate(observables):
        for wire in axes:
            wire_queries[wire] |= 1 << index
    for stage in reversed(circuit.stages):
        active = 0
        for wire in stage.wires:
            active |= wire_queries[wire]
        for wire in stage.wires:
            wire_queries[wire] = active
    widths = [0] * len(observables)
    for mask in wire_queries:
        while mask:
            bit = mask & -mask
            widths[bit.bit_length() - 1] += 1
            mask ^= bit
    return max(widths, default=0)
