"""Compile circuits onto a chip: layout, routing, scheduling and device noise.

    from quantum_circuit_simulator import devices
    from quantum_circuit_simulator.compiler import compile_circuit
    compiled = compile_circuit(qasm_or_qiskit_circuit, devices.get("heavy-hex-127"))
    compiled.report                     # SWAPs, two-qubit gates, depth, duration, layouts
    compiled.expectation("Z0*Z1")       # observables use the circuit's own (logical) qubits

Steps:
1. Layout places logical qubits on physical ones. "greedy" first looks for
   an exact placement with every interacting pair on a coupler, so no SWAPs
   are needed (a chain finds a path, a ring a cycle); failing that, it
   places qubits near their partners from several starting qubits and keeps
   the placement needing the fewest SWAPs. "trivial" keeps the numbering;
   a list gives the physical qubit of each logical one. The report's
   `layout` says which method was used.
2. Routing inserts SWAPs so every two-qubit gate acts on coupled qubits.
   "greedy" moves one qubit along a shortest path; "sabre" uses Qiskit's
   SABRE router, usually with fewer SWAPs, when Qiskit is installed.
3. Scheduling starts each gate as soon as its qubits are free, using the
   chip's gate durations (RZ is virtual and takes no time).
4. Noise follows each physical gate: the chip's one-qubit error after
   every one-qubit gate except RZ, its two-qubit error after every
   two-qubit gate (a SWAP is three), T1/T2 relaxation for every idle
   interval of a qubit that has started, and readout error on every
   measured Pauli. `noise_scale` multiplies every error and decoherence
   rate, so 0 is noise-free.

The result is exact for the routed circuit with the chosen noise model.
Routing is heuristic: the SWAP count is not minimal.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from ._core._reference import DEFAULT_MAX_TERMS
from .api import Circuit
from .interop import (GATE_END, ConversionError, Converted, _DECOMPOSE, _emit, _lower, parse_observable, readout_expand,
                      source_instructions)

TWO_QUBIT = {"CZ", "ECR"}
Z_TYPE = {"RZ", "Z", "CZ"}   # diagonal gates, which commute with ZZ crosstalk


@dataclass
class Compiled:
    """A circuit compiled onto a device, with its report and layouts."""
    circuit: Circuit
    device: object
    initial_layout: list
    final_layout: list
    parameters: list = field(default_factory=list)
    source_order: list | None = None
    report: dict = field(default_factory=dict)
    readout: tuple | None = None
    schedule: list = field(default_factory=list)   # gates on physical qubits with start times and durations

    def observable(self, spec):
        """Observable terms on the final physical qubits, with the device's readout error applied."""
        n_logical = len(self.final_layout)
        moved = [(c, {self.final_layout[q]: a for q, a in p.items()}) for c, p in parse_observable(spec, n_logical)]
        return readout_expand(moved, self.readout)

    def shifts(self, values):
        return Converted(self.circuit, self.parameters, self.source_order).shifts(values)

    def expectation(self, observable, *, values=None, gradients=False, backend="native", truncation=None,
                    threads=1):
        plan = self.circuit.prepare(self.observable(observable), backend=backend, truncation=truncation,
                                    threads=threads)
        return plan.evaluate(self.shifts(values) if self.parameters else None, gradients=gradients)


def compile_circuit(circuit, device, *, layout="greedy", router="greedy", noise=True, idle_noise=True,
                    noise_scale=1.0, values=None, max_terms=DEFAULT_MAX_TERMS, gate_model="thermal"):
    """Compile OpenQASM text, a Qiskit circuit or a gates-only `Circuit` onto `device`.

    gate_model "thermal" (default) gives each gate T1/T2 relaxation for its duration and a
    depolarizing error for the rest of its measured infidelity, as hardware errors arise and
    as Qiskit Aer models them; "depolarizing" puts the whole infidelity into a depolarizing
    channel with the same average fidelity.
    """
    if gate_model not in ("thermal", "depolarizing"):
        raise ValueError('gate_model must be "thermal" or "depolarizing"')
    if device.family == "annealer":
        raise ConversionError(f"{device.name} is an annealer; circuits cannot be compiled onto it",
                              "choose a gate-model device from `quantum-circuit-simulator devices`")
    if router == "sabre":
        n, physical_instructions, source_order, initial, final, swaps = _sabre(circuit, device, layout, values)
        layout_method = "sabre" if layout == "greedy" else (layout if isinstance(layout, str) else "given")
        ops, params = _lower(physical_instructions)
        routed_ops = ops
    elif router == "greedy":
        n, instructions, source_order = _instructions(circuit, values)
        if n > device.n_qubits:
            raise ConversionError(f"the circuit has {n} qubits; {device.name} has {device.n_qubits}",
                                  "choose a larger device")
        ops, params = _lower(instructions)
        initial, layout_method = _layout(layout, n, ops, device)
        routed_ops, final, swaps = _route(ops, initial, device)
    else:
        raise ValueError('router must be "greedy" or "sabre"')
    gate_noise = device.noise_model(noise_scale) if noise else None
    scheduled, facts = _schedule(routed_ops, device, gate_noise, bool(idle_noise and noise and noise_scale > 0),
                                 noise_scale if noise else 0.0, thermal=gate_model == "thermal")
    readout = device.readout_map(noise_scale) if noise else None  # per qubit where the chip says so
    converted = _emit(device.n_qubits, scheduled, params, None, max_terms)  # readout acts on observables
    schedule = facts.pop("schedule")
    budget = facts.pop("noise_budget")
    _add_readout(budget, readout, final)
    report = {"device": device.name, "router": router, "layout": layout_method,
              "logical_qubits": n, "swaps": swaps, **facts, "noise": bool(noise), "idle_noise": bool(idle_noise and noise),
              "noise_scale": noise_scale, "initial_layout": initial, "final_layout": final, "noise_budget": budget}
    return Compiled(converted.circuit, device, initial, final, params, source_order, report, readout, schedule)


def _add_readout(budget, readout, measured):
    """Each measured qubit's readout error, (p01 + p10) / 2, in the budget and its totals."""
    for q in measured:
        pair = readout.get(q) if isinstance(readout, dict) else readout
        if pair:
            budget["qubits"].setdefault(str(q), {})["readout_error"] = (pair[0] + pair[1]) / 2
            budget["totals"]["readout"] += (pair[0] + pair[1]) / 2
    totals = budget["totals"]
    budget["estimated_fidelity"] = math.exp(-sum(v for k, v in totals.items() if k != "zz_phase_rad"))


def _instructions(circuit, values):
    if isinstance(circuit, Circuit):
        instructions = []
        for stage in circuit._impl.stages:
            if stage.name not in {"H", "SX", "X", "Y", "Z", "RX", "RY", "RZ", "CZ", "ECR"}:
                raise ConversionError(f"cannot compile a {stage.name} stage", "compile a gates-only circuit; "
                                      "the device adds noise")
            if stage.param >= 0:
                raise ConversionError("compiling a `Circuit` with shared parameters is not supported",
                                      "compile the OpenQASM or Qiskit form instead")
            angles = [stage.angle] if stage.name in {"RX", "RY", "RZ"} else []
            instructions.append((stage.name.lower(), list(stage.wires), angles))
        return circuit.n_qubits, instructions, None
    return source_instructions(circuit, values)


# ---------------------------------------------------------------- layout
def _layout(layout, n, ops, device):
    """(initial layout, method): "given", "trivial", "exact" or "greedy".

    "greedy" first searches for an exact placement, where every interacting
    pair of qubits sits on a coupler and routing needs no SWAPs (a chain finds
    a path, a ring a cycle). When none exists or the search budget runs out,
    it places qubits greedily from several well-spread starting qubits and
    keeps the placement that routes with the fewest SWAPs.
    """
    if isinstance(layout, (list, tuple)):
        if len(layout) != n or len(set(layout)) != n or not all(0 <= p < device.n_qubits for p in layout):
            raise ValueError(f"layout must give {n} distinct physical qubits of {device.name}")
        return list(layout), "given"
    if layout == "trivial":
        return list(range(n)), "trivial"
    if layout != "greedy":
        raise ValueError('layout must be "greedy", "trivial" or a list of physical qubits')
    weights = Counter(tuple(sorted(w)) for prim, w, *_ in ops if prim in TWO_QUBIT)
    adjacency = device._adjacency()
    exact = _embed(n, weights, adjacency)
    if exact is not None:
        return exact, "exact"
    best = None
    for start in _spread(adjacency, 8):
        candidate = _greedy(n, weights, adjacency, start)
        swaps = _route(ops, candidate, device)[2]
        if best is None or swaps < best[0]:
            best = (swaps, candidate)
        if swaps == 0:
            break
    return best[1], "greedy"


class _Budget(Exception):
    pass


def _embed(n, weights, adjacency, budget=100_000):
    """A placement putting every interacting pair on a coupler, or None.

    Backtracking subgraph matching: logical qubits are placed in breadth-first
    order, each on a free physical qubit coupled to all of its placed partners
    and with enough free neighbours left for its unplaced ones. The search
    stops after `budget` placements; qubits without interactions go last,
    next to the placed ones.
    """
    partners = {q: set() for q in range(n)}
    for a, b in weights:
        partners[a].add(b)
        partners[b].add(a)
    interacting = [q for q in range(n) if partners[q]]
    if not interacting or len(interacting) > 500:
        return None
    if max(len(partners[q]) for q in interacting) > max(len(a) for a in adjacency):
        return None
    neighbours = [set(a) for a in adjacency]
    order, seen = [], set()
    for root in sorted(interacting, key=lambda q: (-len(partners[q]), q)):
        if root in seen:
            continue
        queue, seen = deque([root]), seen | {root}
        while queue:
            q = queue.popleft()
            order.append(q)
            for m in sorted(partners[q], key=lambda m: (-len(partners[m]), m)):
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
    by_degree = sorted(range(len(adjacency)), key=lambda p: (-len(adjacency[p]), p))
    position, used, steps = {}, set(), [0]

    def candidates(q):
        placed = [position[m] for m in partners[q] if m in position]
        if placed:
            return sorted(set.intersection(*(neighbours[p] for p in placed)) - used,
                          key=lambda p: (-len(adjacency[p]), p))
        return [p for p in by_degree if p not in used][:64]  # the start of a component

    def extend(i):
        if i == len(order):
            return True
        q = order[i]
        unplaced = sum(1 for m in partners[q] if m not in position)
        for p in candidates(q):
            if len(neighbours[p] - used) < unplaced:
                continue
            steps[0] += 1
            if steps[0] > budget:
                raise _Budget
            position[q] = p
            used.add(p)
            if extend(i + 1):
                return True
            del position[q]
            used.discard(p)
        return False

    try:
        if not extend(0):
            return None
    except (_Budget, RecursionError):
        return None
    for q in range(n):  # qubits without interactions: next to the placed ones
        if q not in position:
            near = _bfs_multi(adjacency, list(used))
            p = min((p for p in range(len(adjacency)) if p not in used), key=lambda p: (near[p], p))
            position[q] = p
            used.add(p)
    return [position[q] for q in range(n)]


def _spread(adjacency, k):
    """Up to k starting qubits: the best-connected one, then each farthest from those chosen."""
    first = max(range(len(adjacency)), key=lambda p: (len(adjacency[p]), -p))
    chosen = [first]
    while len(chosen) < min(k, len(adjacency)):
        far = _bfs_multi(adjacency, chosen)
        nxt = max((p for p in range(len(adjacency)) if p not in chosen), key=lambda p: (far[p], len(adjacency[p]), -p))
        chosen.append(nxt)
    return chosen


def _greedy(n, weights, adjacency, hub):
    """Place logical qubits in order of interaction, each where its partners are nearest."""
    total = Counter()
    for (a, b), w in weights.items():
        total[a] += w
        total[b] += w
    order = sorted(range(n), key=lambda q: (-total[q], q))
    placed, used, distance_cache = {}, set(), {}

    def distances(p):
        if p not in distance_cache:
            distance_cache[p] = _bfs(adjacency, p)
        return distance_cache[p]

    for q in order:
        partners = [(placed[m], w) for (a, b), w in weights.items() for m in (a, b)
                    if q in (a, b) and m != q and m in placed]
        if not placed:
            choice = hub
        elif partners:
            free = [p for p in range(len(adjacency)) if p not in used]
            choice = min(free, key=lambda p: (sum(w * distances(pp)[p] for pp, w in partners), p))
        else:  # no placed partner: the free qubit nearest the placed ones
            near = _bfs_multi(adjacency, list(used))
            choice = min((p for p in range(len(adjacency)) if p not in used), key=lambda p: (near[p], p))
        placed[q] = choice
        used.add(choice)
    return [placed[q] for q in range(n)]


def _bfs(adjacency, source):
    return _bfs_multi(adjacency, [source])


def _bfs_multi(adjacency, sources):
    far = len(adjacency) + 1
    dist = [far] * len(adjacency)
    queue = deque(sources)
    for s in sources:
        dist[s] = 0
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if dist[b] == far:
                dist[b] = dist[a] + 1
                queue.append(b)
    return dist


# ---------------------------------------------------------------- routing
def _route(ops, layout, device):
    """Map ops to physical qubits, inserting SWAPs before two-qubit gates on uncoupled qubits."""
    adjacency = device._adjacency()
    coupled = set(device.edges)
    l2p = list(layout)
    p2l = {p: q for q, p in enumerate(l2p)}
    routed, swaps = [], 0
    for prim, wires, base, index, weight in ops:
        physical = [l2p[q] for q in wires]
        if prim in TWO_QUBIT and tuple(sorted(physical)) not in coupled:
            path = _path(adjacency, physical[0], physical[1])
            if path is None:
                raise ConversionError(f"qubits {physical} of {device.name} are not connected",
                                      "choose a connected device or another layout")
            for a, b in zip(path, path[1:-1]):  # walk the first qubit to beside the second
                for c, t in ((a, b), (b, a), (a, b)):   # a SWAP is three CX, each a gate with its own error
                    routed += [(sw_prim, list(sw_wires), None, -1, 1) for sw_prim, sw_wires, _ in _DECOMPOSE["cx"]([c, t], [])]
                    routed.append((GATE_END, [c, t], None, -1, 1))
                swaps += 1
                qa, qb = p2l.get(a), p2l.get(b)
                if qa is not None:
                    l2p[qa] = b
                if qb is not None:
                    l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
            physical = [l2p[q] for q in wires]
        routed.append((prim, physical, base, index, weight))
    return routed, l2p, swaps


def _path(adjacency, source, target):
    parent = {source: None}
    queue = deque([source])
    while queue:
        a = queue.popleft()
        if a == target:
            path = []
            while a is not None:
                path.append(a)
                a = parent[a]
            return path[::-1]
        for b in adjacency[a]:
            if b not in parent:
                parent[b] = a
                queue.append(b)
    return None


def _sabre(circuit, device, layout, values):
    try:
        from qiskit import QuantumCircuit, qasm2, transpile
    except ImportError as exc:
        raise ConversionError("router='sabre' needs Qiskit", "pip install qiskit, or use router='greedy'") from exc
    if isinstance(circuit, Circuit):
        raise ConversionError("router='sabre' compiles OpenQASM or Qiskit circuits",
                              "pass the circuit as OpenQASM or Qiskit, or use router='greedy'")
    if isinstance(circuit, str):
        circuit = qasm2.loads(circuit, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS) \
            if "OPENQASM 3" not in circuit else __import__("qiskit").qasm3.loads(circuit)
    if values is not None:
        params = list(circuit.parameters)
        circuit = circuit.assign_parameters(dict(zip(params, values)) if not isinstance(values, dict)
                                            else {p: values[p.name] for p in params})
    if circuit.num_qubits > device.n_qubits:
        raise ConversionError(f"the circuit has {circuit.num_qubits} qubits; {device.name} has {device.n_qubits}",
                              "choose a larger device")
    from qiskit.transpiler import CouplingMap
    coupling = CouplingMap([list(e) for e in device.edges] + [[b, a] for a, b in device.edges])
    initial = None
    if isinstance(layout, (list, tuple)):
        initial = list(layout)
    elif layout == "trivial":
        initial = list(range(circuit.num_qubits))
    out = transpile(circuit, coupling_map=coupling, initial_layout=initial, routing_method="sabre",
                    layout_method=None if initial else "sabre", optimization_level=1, seed_transpiler=7,
                    basis_gates=["rz", "sx", "x", "rx", "ry", "h", "cz", "swap"])
    swaps = sum(1 for item in out.data if item.operation.name == "swap")
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(circuit.num_qubits))
    initial_layout = list(out.layout.initial_index_layout()[:circuit.num_qubits]) if out.layout is not None \
        else list(range(circuit.num_qubits))
    n, instructions, _ = source_instructions(out)
    order = list(circuit.parameters)
    return n, instructions, order, initial_layout, final[:circuit.num_qubits], swaps


# ---------------------------------------------------------------- schedule and noise
def _schedule(ops, device, noise, idle, scale=1.0, thermal=False):
    """Insert gate errors, crosstalk and idle relaxation; return ops and timing facts.

    Errors follow each qubit's and coupler's own values where the chip gives
    them. After every two-qubit gate, each started neighbour of its qubits (a
    spectator) takes the chip's depolarizing spectator error. An always-on ZZ
    coupling zeta acts on every coupled pair of started qubits for as long as
    both are active, from the later one's first gate to the end of the
    circuit: the pair accumulates a conditional phase 2 pi zeta t, applied as
    RZZ just before the next gate on either qubit that does not commute with
    it (X- or Y-type: diagonal gates commute) and at the end. A coupler's
    own two-qubit gate replaces its ZZ during the gate (it is calibrated into
    the gate); spectator pairs keep accumulating through it.
    `scale` multiplies every error, ZZ rate and decoherence rate (T1 and T2 are divided by it).
    """
    if noise is None:
        scale = 0.0
    durations = {1: device.time_1q or 0.0, 2: device.time_2q or 0.0}
    adjacency = device._adjacency()
    spectator = device.crosstalk.get("spectator_error", 0.) * scale
    zz_rate = 2 * math.pi * device.crosstalk.get("zz_hz", 0.) * scale   # conditional phase per second
    zz_clock = {}   # coupled pair of started qubits -> time up to which its ZZ phase has been applied
    coherence = {}
    # The noise budget: what each qubit and coupler contributes, for reports and chip heat maps.
    qubits, couplers = {}, {}
    totals = {"gate_1q": 0.0, "gate_2q": 0.0, "idle": 0.0, "spectator": 0.0, "readout": 0.0, "zz_phase_rad": 0.0}

    def flush_pair(pair, until):
        """Apply a pair's ZZ phase accumulated up to `until`."""
        elapsed = until - zz_clock[pair]
        if elapsed > 0:
            out.append(("RZZ", list(pair), zz_rate * elapsed, -1, 1))  # coherent error, not a gate
            zz_clock[pair] = until
            entry = couplers.setdefault(f"{pair[0]}-{pair[1]}", {})
            entry["zz_phase_rad"] = entry.get("zz_phase_rad", 0.0) + zz_rate * elapsed
            totals["zz_phase_rad"] += zz_rate * elapsed

    def flush_zz(qubit, until):
        """Apply each pair's ZZ phase accumulated up to `until`, for pairs containing `qubit`."""
        for other in adjacency[qubit]:
            pair = (min(qubit, other), max(qubit, other))
            if pair in zz_clock:
                flush_pair(pair, until)

    def relax_times(w):
        if w not in coherence:
            t1, t2 = device.qubit(w, "t1"), device.qubit(w, "t2")
            if t1 is None and t2 is None:
                coherence[w] = None
            else:
                t1 = t1 if t1 is not None else 1e6 * t2   # ions: effectively no T1 decay
                t2 = min(t2 if t2 is not None else 2 * t1, 2 * t1)
                coherence[w] = (t1 / scale, t2 / scale)
        return coherence[w]

    def gate_error(wires, error, duration):
        """A gate's infidelity: relaxation during it when thermal, depolarizing for the remainder."""
        d = 2 ** len(wires)
        if thermal and duration > 0 and all(relax_times(w) for w in wires):
            fidelity = 1.0   # process fidelity of the relaxation, Tr R / d^2, a product over wires
            for w in wires:
                t1, t2 = relax_times(w)
                out.append(("RELAX", [w], (duration, t1, t2), -1, 1))
                fidelity *= (1 + math.exp(-duration / t1) + 2 * math.exp(-duration / t2)) / 4
            error = max(0.0, error - d * (1 - fidelity) / (d + 1))   # average infidelity r = d (1 - F) / (d + 1)
        if error:
            out.append(("DEPOL", wires, min(1 - 1 / d ** 2, error * (d + 1) / d), -1, 1))

    pending = {}   # coupler -> (wires, infidelity, duration) of a two-qubit gate whose source gate has not ended

    def flush(wires):
        """Apply the pending two-qubit errors on couplers that share a wire with `wires`."""
        for pair in [p for p in pending if set(p) & set(wires)]:
            gate_error(*pending.pop(pair))

    busy, started, level = {}, set(), {}
    out, one, two = [], 0, 0
    schedule = []

    def charge(q, key, amount, total=None):
        entry = qubits.setdefault(str(q), {})
        entry[key] = entry.get(key, 0) + amount
        if total:
            totals[total] += amount

    def relax(w, gap):
        t1, t2 = relax_times(w)
        out.append(("RELAX", [w], (gap, t1, t2), -1, 1))
        charge(w, "idle_s", gap)
        # Average infidelity of the relaxation channel: (3 - e^{-t/T1} - 2 e^{-t/T2}) / 6.
        charge(w, "idle_error", (3 - math.exp(-gap / t1) - 2 * math.exp(-gap / t2)) / 6, "idle")

    for prim, wires, base, index, weight in ops:
        if prim == GATE_END:
            flush(wires)
            continue
        if len(wires) == 2:
            flush(wires)   # a new two-qubit gate on these wires: the previous one has certainly ended
        virtual = prim == "RZ"
        duration = 0.0 if virtual else durations[len(wires)]
        start = max(busy.get(w, 0.0) for w in wires)
        if zz_rate and prim not in Z_TYPE:  # ZZ commutes with Z-type gates, so it can wait past them
            for w in wires:
                flush_zz(w, start)
        elif zz_rate and len(wires) == 2 and (min(wires), max(wires)) in zz_clock:
            # The gate's own pair stops accumulating during the gate: apply what it gathered before.
            flush_pair((min(wires), max(wires)), start)
        if idle:
            for w in wires:
                gap = start - busy.get(w, 0.0)
                if w in started and gap > 0 and relax_times(w):
                    relax(w, gap)
        out.append((prim, wires, base, index, weight))
        schedule.append({"gate": prim.lower(), "qubits": list(wires), "start_s": start, "duration_s": duration,
                         **({"parameter": index} if index >= 0 else {})})
        if not virtual:
            depth = max(level.get(w, 0) for w in wires) + 1
            for w in wires:
                level[w] = depth
        if len(wires) == 2:
            two += 1
            # The coupler's own infidelity when the chip gives one, as a depolarizing probability.
            error = device.coupler_error(*wires) if scale else 0.
            key = f"{min(wires)}-{max(wires)}"
            entry = couplers.setdefault(key, {})
            entry["gates"] = entry.get("gates", 0) + 1
            if error:
                pending[(min(wires), max(wires))] = (wires, error * scale, duration)
                entry["gate_error"] = entry.get("gate_error", 0.0) + error * scale
                totals["gate_2q"] += error * scale
            if spectator:
                for w in wires:
                    for s in adjacency[w]:
                        if s not in wires and s in started:
                            out.append(("DEPOL", [s], min(0.75, spectator * 3 / 2), -1, 1))
                            charge(s, "spectator_error", spectator, "spectator")
        elif not virtual:
            one += 1
            error = device.qubit(wires[0], "error_1q") if scale else 0.
            charge(wires[0], "gates_1q", 1)
            if error:
                gate_error(wires, error * scale, duration)
                charge(wires[0], "gate_error_1q", error * scale, "gate_1q")
        for w in wires:
            busy[w] = start + duration
            if zz_rate and w not in started:  # a newly started qubit couples to its started neighbours
                for other in adjacency[w]:
                    if other in started:
                        zz_clock[(min(w, other), max(w, other))] = start
            started.add(w)
        if zz_rate and len(wires) == 2:  # the gate's own pair: ZZ is part of the calibrated gate
            pair = (min(wires), max(wires))
            if pair in zz_clock:
                zz_clock[pair] = max(zz_clock[pair], start + duration)
    flush(list(range(device.n_qubits)))
    end = max(busy.values(), default=0.0)
    if zz_rate:
        for w in sorted(started):
            flush_zz(w, end)
    if idle:
        for w in sorted(started):
            if end - busy[w] > 0 and relax_times(w):
                relax(w, end - busy[w])
    budget = {"qubits": qubits, "couplers": couplers, "totals": totals,
              "note": "error sums are first-order (infidelities add); estimated_fidelity = exp(-sum of totals)"}
    return out, {"one_qubit_gates": one, "two_qubit_gates": two, "depth": max(level.values(), default=0),
                 "duration_s": end, "schedule": schedule, "noise_budget": budget}
