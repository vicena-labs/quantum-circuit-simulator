"""Backward cones grouped for the native solver and the graph compiler.

Preparation finds every observable's backward cone in one reverse bitset pass,
then groups observables whose combined cone fits the key width the widest
single cone needs anyway. A group is propagated jointly: the circuit is linear
in the observable, so this gives the same result while merging Pauli terms the
observables share. If a group exceeds the term cap, it is split back into one
cone per observable, so a joint group never fails where single cones succeed.

Each group is renumbered to its own wires, so any register width works as long
as each group fits 256 wires. All groups share one bound copy of the circuit's
stages. The native solver evaluates all groups in one call; optional
truncation discards small or heavy terms and reports a rigorous bound on the
discarded contribution.
"""
from __future__ import annotations
import ctypes as ct
from dataclasses import dataclass
import numpy as np
from ._native import NativePrepared, CGroup, CTerm, CBatchStats, library
from ._reference import ConeOverflow

# Generator norm bound 2*||G|| of each parameterised stage, used to bound the
# gradient of truncated terms: RX/RY/RZ = exp(-i t P/2), fSim's iSWAP block.
_LIPSCHITZ = {'RX': 1., 'RY': 1., 'RZ': 1., 'RZZ': 1., 'FSIM': 2.}


@dataclass(slots=True)
class _Cone:
    terms: dict      # (x, z) global masks -> value coefficient
    wires: list      # sorted global wires of the backward cone
    stages: list     # positions in the circuit, forward order


class ConeGroups:
    """Observables' backward cones, grouped and packed for one native call."""

    def __init__(self, circuit, observables, *, max_cone_qubits=256, stage_budgets=None, threads=1,
                 truncate_below=0., max_weight=-1):
        self.c = circuit
        self.observables = tuple((float(coef), dict(ops)) for coef, ops in observables)
        self.full = NativePrepared(circuit)  # one bound copy shared by every cone
        self.threads = int(threads)
        self.truncate_below = float(truncate_below)
        self.max_weight = int(max_weight)
        lipschitz = [_LIPSCHITZ.get(s.name, 0.) * abs(s.scale) if s.param >= 0 else 0. for s in circuit.stages]
        self.lipschitz = (ct.c_double * max(1, len(lipschitz)))(*lipschitz)
        # |exact value| <= observable l1 norm (readout only shrinks it), and
        # |d exact / d theta_j| <= norm * sum of that parameter's stage bounds.
        self.norm = sum(abs(c) for c, _ in self.observables)
        self.slope = [self.norm * sum(l for l, s in zip(lipschitz, circuit.stages) if s.param == j)
                      for j in range(circuit.nparams)]
        self.cones = self._cones(max_cone_qubits, stage_budgets)
        self._set_groups(self._cluster())

    @property
    def truncating(self):
        return self.truncate_below > 0 or self.max_weight >= 0

    def _bound_facts(self, value, gradient, value_bound, gradient_bound, truncated_terms):
        """Reported bounds: the accumulated bound is rigorous but can exceed the
        trivial one, |exact - returned| <= max|exact| + |returned|; report the smaller,
        and say whether it says anything beyond the trivial bound."""
        trivial = self.norm + abs(value)
        return dict(truncated_terms=int(truncated_terms),
                    error_bound=min(float(value_bound), trivial),
                    bound_informative=bool(value_bound < trivial),
                    gradient_error_bound=tuple(
                        min(float(gradient_bound[j]), self.slope[j] + abs(gradient[j]))
                        for j in range(self.c.nparams)) if gradient_bound is not None else None)

    def _cones(self, max_cone_qubits, stage_budgets):
        circuit = self.c
        initials = [self._initial_terms(coefficient, axes) for coefficient, axes in self.observables]
        # Dependency mask of all observables depending on each physical wire.
        # At each reverse gate, union its input-wire masks, exactly mirroring
        # active |= gate.wires for each independent observable.
        wire_queries = [0] * circuit.nqubits
        for query, terms in enumerate(initials):
            support = 0
            for x, z in terms:
                support |= x | z
            marker = 1 << query
            while support:
                bit = support & -support
                wire_queries[bit.bit_length() - 1] |= marker
                support ^= bit
        membership = [0] * len(circuit.stages)
        for index in range(len(circuit.stages) - 1, -1, -1):
            wires = circuit.stages[index].wires
            active = wire_queries[wires[0]]
            if len(wires) == 2:
                active |= wire_queries[wires[1]]
                wire_queries[wires[1]] = active
            wire_queries[wires[0]] = active
            membership[index] = active
        cone_wires = [[] for _ in initials]
        for wire, mask in enumerate(wire_queries):
            while mask:
                bit = mask & -mask
                cone_wires[bit.bit_length() - 1].append(wire)
                mask ^= bit
        widest = max(map(len, cone_wires), default=0)
        if widest > max_cone_qubits:
            raise ConeOverflow(f'backward cone needs {widest} qubits; max_cone_qubits={max_cone_qubits}')
        cone_stages = [[] for _ in initials]
        for index, queries in enumerate(membership):
            while queries:
                bit = queries & -queries
                cone_stages[bit.bit_length() - 1].append(index)
                queries ^= bit
        if stage_budgets is not None:
            per_cone, total = stage_budgets
            if max(map(len, cone_stages), default=0) > per_cone:
                raise ConeOverflow('backward cone stage budget exceeded')
            if sum(map(len, cone_stages)) > total:
                raise ConeOverflow('wide preparation total selected stages exceeded')
        return [_Cone(t, w, s) for t, w, s in zip(initials, cone_wires, cone_stages)]

    def _initial_terms(self, coefficient, axes):
        """Readout-expanded value coefficients keyed by global (x, z) masks.

        Single and two-wire Z observables expand readout here directly; the
        circuit's general routine handles the rest.
        """
        circuit = self.c
        if 1 <= len(axes) <= 2 and all(axis == 'Z' for axis in axes.values()):
            expanded = {0: coefficient}
            for q in sorted(axes):
                p01, p10 = circuit.readout.get(q, (0., 0.))
                alpha, beta = p10 - p01, 1 - p01 - p10
                following = {}
                for mask, value in expanded.items():
                    following[mask] = following.get(mask, 0.) + value * alpha
                    following[mask | (1 << q)] = following.get(mask | (1 << q), 0.) + value * beta
                expanded = following
            return {(0, mask): value for mask, value in expanded.items() if value != 0.}
        terms = circuit.initial_terms([(coefficient, axes)], initial='zero', gradients=False)
        return {key: float(value[0]) for key, value in terms.items()}

    def _cluster(self):
        """Greedy groups, in observable order, whose wire union fits one key width."""
        limit = 64 * max(1, -(-max((len(c.wires) for c in self.cones), default=1) // 64))
        groups, current, union = [], [], set()
        for index, cone in enumerate(self.cones):
            merged = union.union(cone.wires)
            if current and len(merged) > limit:
                groups.append(current)
                current, merged = [], set(cone.wires)
            current.append(index)
            union = merged
        if current:
            groups.append(current)
        return groups

    def _set_groups(self, groups):
        self.groups = groups
        self._keep = []
        packed = (CGroup * len(groups))()
        for slot, members in enumerate(groups):
            wires = sorted(set().union(*(self.cones[i].wires for i in members)))
            local = {wire: position for position, wire in enumerate(wires)}
            stages = sorted(set().union(*(self.cones[i].stages for i in members)))
            indices = (ct.c_int32 * max(1, len(stages)))(*stages)
            pairs = []
            for index in stages:
                stage_wires = self.c.stages[index].wires
                pairs += [local[stage_wires[0]], local[stage_wires[1]] if len(stage_wires) == 2 else -1]
            local_wires = (ct.c_int32 * max(1, len(pairs)))(*pairs)
            items = [(key, value) for i in members for key, value in self.cones[i].terms.items()]
            terms = (CTerm * max(1, len(items)))()
            for j, ((x, z), value) in enumerate(items):
                lx = _renumber(x, local)
                lz = _renumber(z, local)
                for k in range(4):
                    terms[j].x[k] = (lx >> (64 * k)) & ((1 << 64) - 1)
                    terms[j].z[k] = (lz >> (64 * k)) & ((1 << 64) - 1)
                terms[j].coef = value
            group = packed[slot]
            group.nqubits = max(1, len(wires))
            group.nstages = len(stages)
            group.nterms = len(items)
            group.indices = ct.cast(indices, ct.POINTER(ct.c_int32))
            group.local_wires = ct.cast(local_wires, ct.POINTER(ct.c_int32))
            group.terms = ct.cast(terms, ct.POINTER(CTerm))
            self._keep += [indices, local_wires, terms]
        self._groups = packed

    def bind(self, deltas):
        # No PTM copies: every cone shares `self.full.stages`.
        self.full.bind(deltas)
        return self

    def _split_overflows(self, codes):
        """Split every joint group over the term cap into single cones.

        Returns False when no joint group overflowed, so the caller stops.
        """
        overflowed = [g for g, code in zip(self.groups, codes) if code == -1 and len(g) > 1]
        if not overflowed:
            return False
        groups = []
        for g, code in zip(self.groups, codes):
            groups += [[i] for i in g] if code == -1 and len(g) > 1 else [g]
        self._set_groups(groups)
        return True


class IndexedBatchPrepared(ConeGroups):
    """Native sparse propagation of every cone group in one call."""

    def __init__(self, circuit, observables, *, max_cone_qubits=256, truncate_below=0.,
                 max_weight=-1, threads=1, stage_budgets=None):
        super().__init__(circuit, observables, max_cone_qubits=max_cone_qubits,
                         stage_budgets=stage_budgets, threads=threads,
                         truncate_below=truncate_below, max_weight=max_weight)
        self.preparation_mode = 'single_reverse_bitset_pass_grouped_cones'

    def evaluate(self, *, include_stats=False, with_gradients=True):
        out, passes = self.full.in_chunks(lambda n: self._run(n, with_gradients), with_gradients)
        value, gradient = float(out[0]), np.array(out[1:] if with_gradients else [])
        if not include_stats:
            return value, gradient
        first = passes[0]
        facts = {'max_single_cone_terms': max(int(s.peak_terms) for s in passes), 'observables': len(self.cones),
                 'cone_groups': len(self.groups),
                 'max_cone_qubits': max((len(c.wires) for c in self.cones), default=0)}
        if len(passes) > 1:
            facts['gradient_passes'] = len(passes)
        if self.truncating:
            bounds = [b for s in passes for b in list(s.gradient_bound)[:16]][:self.c.nparams]
            facts.update(self._bound_facts(value, gradient, first.value_bound,
                                           bounds if with_gradients else None, first.truncated_terms))
        return value, gradient, facts

    def _run(self, nparams, with_gradients):
        """One native call over every group with `nparams` derivatives."""
        width = nparams + 1 if with_gradients else 1
        out = (ct.c_double * width)()
        stats = CBatchStats()
        while True:
            codes = (ct.c_int32 * max(1, len(self.groups)))()
            code = library().run_pauli_batch(
                nparams, self.full.stages, len(self.full.stages), self.lipschitz,
                len(self.groups), self._groups, 0, self.c.max_terms,
                self.truncate_below, self.max_weight, int(with_gradients), self.threads,
                out, ct.byref(stats), codes)
            failed = self.groups[stats.failed_group] if code and stats.failed_group >= 0 else None
            # Joint groups that outgrew the cap: split them all into single cones.
            if code == -1 and self._split_overflows(codes[:len(self.groups)]):
                continue
            break
        if code == -1:
            kind = 'truncated' if self.truncating else 'exact'
            raise ConeOverflow(f'{kind} cone of observable index {failed[0]} exceeds cap {self.c.max_terms}')
        if code:
            raise RuntimeError(f'exact cone {failed[0] if failed else "?"} status {code}')
        return list(out), stats


def _renumber(mask, local):
    out = 0
    while mask:
        bit = mask & -mask
        out |= 1 << local[bit.bit_length() - 1]
        mask ^= bit
    return out
