"""Quantum chip descriptions and a catalogue of common architectures.

A Device is a chip's connectivity, native gates, error rates, coherence times
and gate durations, in one JSON-serialisable description. The catalogue
generates the main architecture families from their published layouts; a
custom chip is the same description written by hand or by a design tool.

    from quantum_circuit_simulator import devices
    devices.catalog()                                # every family and its parameters
    chip = devices.get("heavy-hex-127")
    grid = devices.get("square-grid", rows=8, columns=8)
    noisy = chip.scaled_noise(2.0)                   # for noise sweeps
    devices.Device.from_dict(json.load(open("my_chip.json")))

Error rates are stored as vendors report them: average gate infidelity for
gates and assignment error probabilities for readout. noise_model() converts
gate infidelities to the depolarizing probabilities Quantum Circuit Simulator simulates.
Catalogue noise values are illustrative typical figures for each technology,
not calibration data for any particular chip; override them for real devices.
Names describe architecture styles and do not imply a vendor's endorsement.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources

from .topology import Topology

SUPERCONDUCTING = dict(error_1q=3e-4, error_2q=5e-3, readout=(0.01, 0.015), t1=200e-6, t2=150e-6,
                       time_1q=40e-9, time_2q=300e-9)
TRAPPED_ION = dict(error_1q=2e-4, error_2q=2e-3, readout=(2e-3, 3e-3), t1=None, t2=1.0,
                   time_1q=10e-6, time_2q=200e-6)
NEUTRAL_ATOM = dict(error_1q=1e-3, error_2q=5e-3, readout=(0.01, 0.01), t1=4.0, t2=1e-3,
                    time_1q=0.5e-6, time_2q=0.4e-6)
ILLUSTRATIVE = ("Illustrative typical values for this technology, not calibration data for a particular chip; "
                "override them to model a real device.")


DEVICE_SCHEMA = "quantum-circuit-simulator-device/1"


@dataclass(frozen=True)
class Device:
    """A chip: coupling graph, native gates, noise, coherence and timing.

    error_1q and error_2q are average gate infidelities; readout is
    (p01, p10), the probabilities of reading 1 for |0> and 0 for |1>; t1, t2
    and gate times are in seconds (None for effectively unlimited). An
    annealer has no gate-model noise: its error fields are None.
    """
    name: str
    family: str
    topology: Topology
    native_gates: dict
    error_1q: float | None = None
    error_2q: float | None = None
    readout: tuple | None = None
    t1: float | None = None
    t2: float | None = None
    time_1q: float | None = None
    time_2q: float | None = None
    description: str = ""
    source: str = ""
    parameters: dict = field(default_factory=dict)
    coupler_errors: dict = field(default_factory=dict)  # (a, b) -> two-qubit infidelity, overriding error_2q
    qubit_overrides: dict = field(default_factory=dict)  # q -> {error_1q, readout, t1, t2}, overriding the chip's
    crosstalk: dict = field(default_factory=dict)        # {"spectator_error": r, "zz_hz": zeta}
    coordinates: tuple = ()  # (x, y) of each qubit in lattice units, for drawing the chip; () when unknown

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {sorted(FAMILIES)}")
        for key in ("error_1q", "error_2q"):
            value = getattr(self, key)
            if value is not None and not (math.isfinite(value) and 0 <= value < 1):
                raise ValueError(f"{key} must be in [0, 1)")
        if self.readout is not None:
            if len(self.readout) != 2 or not all(0 <= p < 0.5 for p in self.readout):
                raise ValueError("readout must be (p01, p10), each in [0, 0.5)")
            object.__setattr__(self, "readout", tuple(float(p) for p in self.readout))
        for key in ("t1", "t2", "time_1q", "time_2q"):
            value = getattr(self, key)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{key} must be positive or None")
        if not isinstance(self.native_gates, dict) or set(self.native_gates) - {"one_qubit", "two_qubit"}:
            raise ValueError('native_gates must be {"one_qubit": [...], "two_qubit": [...]}')
        canonical, edges = {}, set(self.topology.edges)
        for pair, error in (self.coupler_errors or {}).items():
            key = tuple(sorted(int(q) for q in pair))
            if key not in edges:
                raise ValueError(f"coupler_errors names {key}, which is not a coupler")
            if not (math.isfinite(error) and 0 <= error < 1):
                raise ValueError("coupler errors must be in [0, 1)")
            canonical[key] = float(error)
        object.__setattr__(self, "coupler_errors", canonical)
        overrides = {}
        for q, values in (self.qubit_overrides or {}).items():
            q = int(q)
            if not 0 <= q < self.topology.n_qubits:
                raise ValueError(f"qubit_overrides names qubit {q}, which the chip does not have")
            unknown = set(values) - {"error_1q", "readout", "t1", "t2"}
            if unknown:
                raise ValueError(f"qubit overrides take error_1q, readout, t1 and t2, not {sorted(unknown)}")
            clean = {}
            for key, value in values.items():
                if value is None:
                    continue
                if key == "readout":
                    if len(value) != 2 or not all(0 <= p < 0.5 for p in value):
                        raise ValueError("a qubit's readout must be (p01, p10), each in [0, 0.5)")
                    clean[key] = tuple(float(p) for p in value)
                elif key == "error_1q":
                    if not (math.isfinite(value) and 0 <= value < 1):
                        raise ValueError("a qubit's error_1q must be in [0, 1)")
                    clean[key] = float(value)
                else:
                    if not (math.isfinite(value) and value > 0):
                        raise ValueError(f"a qubit's {key} must be positive")
                    clean[key] = float(value)
            if clean:
                overrides[q] = clean
        object.__setattr__(self, "qubit_overrides", overrides)
        crosstalk = dict(self.crosstalk or {})
        unknown = set(crosstalk) - {"spectator_error", "zz_hz"}
        if unknown:
            raise ValueError(f"crosstalk takes spectator_error and zz_hz, not {sorted(unknown)}")
        if not 0 <= crosstalk.get("spectator_error", 0.) < 1 or not math.isfinite(crosstalk.get("zz_hz", 0.)):
            raise ValueError("crosstalk spectator_error must be in [0, 1) and zz_hz finite")
        object.__setattr__(self, "crosstalk", {k: float(v) for k, v in crosstalk.items() if v})
        coordinates = tuple((float(x), float(y)) for x, y in (self.coordinates or ()))
        if coordinates and len(coordinates) != self.topology.n_qubits:
            raise ValueError("coordinates need one (x, y) per qubit")
        if not all(math.isfinite(v) for xy in coordinates for v in xy):
            raise ValueError("coordinates must be finite")
        object.__setattr__(self, "coordinates", coordinates)

    def positions(self):
        """(x, y) of every qubit: the chip's own coordinates, else a deterministic force-directed layout."""
        return list(self.coordinates) if self.coordinates else spring_layout(self.n_qubits, self.edges)

    def qubit(self, q, key):
        """A qubit's error_1q, readout, t1 or t2: its own value, else the chip's."""
        return self.qubit_overrides.get(q, {}).get(key, getattr(self, key))

    def coupler_error(self, a, b):
        """Two-qubit infidelity of the coupler (a, b): its own value, else error_2q."""
        return self.coupler_errors.get(tuple(sorted((a, b))), self.error_2q)

    @property
    def n_qubits(self):
        return self.topology.n_qubits

    @property
    def edges(self):
        return self.topology.edges

    # ------------------------------------------------------------ structure
    def neighbors(self, qubit):
        return sorted({b for a, b in self.edges if a == qubit} | {a for a, b in self.edges if b == qubit})

    def distances(self, source):
        """Coupler hops from `source` to every qubit (-1 when unreachable)."""
        adjacency = self._adjacency()
        dist = [-1] * self.n_qubits
        dist[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if dist[b] < 0:
                    dist[b] = dist[a] + 1
                    queue.append(b)
        return dist

    def _adjacency(self):
        adjacency = [[] for _ in range(self.n_qubits)]
        for a, b in self.edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency

    def summary(self):
        """Counts, degrees and (for up to 2,000 qubits) connectivity and diameter."""
        degrees = [len(n) for n in self._adjacency()]
        facts = {"name": self.name, "family": self.family, "n_qubits": self.n_qubits,
                 "couplers": len(self.edges), "max_degree": max(degrees, default=0),
                 "mean_degree": round(sum(degrees) / max(1, len(degrees)), 3),
                 "native_gates": self.native_gates, "error_1q": self.error_1q, "error_2q": self.error_2q,
                 "readout": self.readout, "t1": self.t1, "t2": self.t2,
                 "time_1q": self.time_1q, "time_2q": self.time_2q, "parameters": self.parameters,
                 "qubits_with_own_noise": len(self.qubit_overrides), "couplers_with_own_error": len(self.coupler_errors),
                 "crosstalk": self.crosstalk}
        if self.n_qubits <= 2000:
            far = self.distances(0)
            facts["connected"] = min(far) >= 0
            if facts["connected"]:
                facts["diameter"] = max(max(self.distances(q)) for q in range(0, self.n_qubits,
                                                                            max(1, self.n_qubits // 64)))
                facts["diameter_note"] = "exact for up to 64 qubits, sampled from 64 sources above"
        return facts

    # ------------------------------------------------------------ noise
    def noise_model(self, scale=1.0):
        """Uniform noise for simulating circuits on this chip.

        Per-coupler and per-qubit errors and crosstalk apply when a circuit
        is compiled onto the chip; this uniform model uses the chip-wide
        values, except readout, which follows each qubit. Gate infidelity r on k qubits becomes a depolarizing probability
        p = r (d + 1) / d with d = 2^k, which has the same average fidelity;
        readout is applied to every wire. `scale` multiplies every error.
        Idle decoherence from t1 and t2 needs a schedule and is not included.
        """
        if self.family == "annealer":
            from .interop import ConversionError
            raise ConversionError(f"{self.name} is an annealer and has no gate-model noise",
                                  "use a gate-model device for circuit simulation")
        from .interop import Noise
        p1 = min(0.75, (self.error_1q or 0.) * scale * 3 / 2)
        p2 = min(0.9375, (self.error_2q or 0.) * scale * 5 / 4)
        readout = self.readout_map(scale)
        return Noise(depolarizing_1q=p1, depolarizing_2q=p2, readout=readout)

    def readout_map(self, scale=1.0):
        """Readout errors scaled by `scale`: one pair for every wire, or a mapping of wire to pair
        when qubits differ; None without readout error."""
        def scaled(pair):
            return tuple(min(0.499, p * scale) for p in pair) if pair else None
        if not any("readout" in v for v in self.qubit_overrides.values()):
            return scaled(self.readout)
        pairs = {q: scaled(self.qubit(q, "readout")) for q in range(self.n_qubits)}
        return {q: p for q, p in pairs.items() if p}

    def with_noise(self, **changes):
        """A copy with some noise or timing fields replaced."""
        allowed = {"error_1q", "error_2q", "readout", "t1", "t2", "time_1q", "time_2q", "coupler_errors",
                   "qubit_overrides", "crosstalk"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown noise fields {sorted(unknown)}; use {sorted(allowed)}")
        return replace(self, **changes)

    def scaled_noise(self, factor):
        """A copy with every gate and readout error multiplied by `factor`."""
        if not (math.isfinite(factor) and factor >= 0):
            raise ValueError("factor must be finite and non-negative")
        scale = lambda x: None if x is None else min(0.999, x * factor)  # noqa: E731
        return replace(self, error_1q=scale(self.error_1q), error_2q=scale(self.error_2q),
                       readout=tuple(min(0.499, p * factor) for p in self.readout) if self.readout else None,
                       coupler_errors={k: scale(v) for k, v in self.coupler_errors.items()},
                       qubit_overrides={q: {k: (scale(v) if k == "error_1q" else
                                                tuple(min(0.499, p * factor) for p in v) if k == "readout" else v)
                                            for k, v in values.items()} for q, values in self.qubit_overrides.items()},
                       crosstalk={"spectator_error": scale(self.crosstalk.get("spectator_error", 0.)) or 0.,
                                  "zz_hz": self.crosstalk.get("zz_hz", 0.) * factor},
                       name=f"{self.name} (noise x{factor:g})")

    def uncoupled_pairs(self, circuit):
        """Two-qubit operations of a `Circuit` on qubit pairs this chip does not couple."""
        if circuit.n_qubits > self.n_qubits:
            raise ValueError(f"the circuit has {circuit.n_qubits} qubits; {self.name} has {self.n_qubits}")
        coupled = set(self.edges)
        pairs = []
        for stage in circuit._impl.stages:
            # Noise channels on a pair follow a gate there; count gates only.
            if stage.name == "DEPOLARIZE":
                continue
            if len(stage.wires) == 2 and tuple(sorted(stage.wires)) not in coupled:
                pairs.append(tuple(sorted(stage.wires)))
        return pairs

    # ------------------------------------------------------------ JSON
    def to_dict(self):
        return {"schema": DEVICE_SCHEMA, "name": self.name, "family": self.family,
                "n_qubits": self.n_qubits, "edges": [list(e) for e in self.edges],
                "native_gates": self.native_gates, "error_1q": self.error_1q, "error_2q": self.error_2q,
                "readout": list(self.readout) if self.readout else None, "t1": self.t1, "t2": self.t2,
                "time_1q": self.time_1q, "time_2q": self.time_2q, "description": self.description,
                "source": self.source, "parameters": self.parameters,
                **({"coupler_errors": [[a, b, e] for (a, b), e in sorted(self.coupler_errors.items())]}
                   if self.coupler_errors else {}),
                **({"qubit_overrides": [{"qubit": q, **{k: list(v) if k == "readout" else v for k, v in values.items()}}
                                        for q, values in sorted(self.qubit_overrides.items())]}
                   if self.qubit_overrides else {}),
                **({"crosstalk": dict(self.crosstalk)} if self.crosstalk else {}),
                **({"coordinates": [list(xy) for xy in self.coordinates]} if self.coordinates else {})}

    @classmethod
    def from_dict(cls, data):
        """A Device from to_dict() output or a hand-written chip description."""
        known = {"schema", "name", "family", "n_qubits", "edges", "native_gates", "error_1q", "error_2q",
                 "readout", "t1", "t2", "time_1q", "time_2q", "description", "source", "parameters",
                 "coupler_errors", "qubit_overrides", "crosstalk", "coordinates"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown device fields {sorted(unknown)}")
        # Files written before the package was renamed in 0.5.0rc1 say pauli-cones-device/1.
        if data.get("schema", DEVICE_SCHEMA) not in (DEVICE_SCHEMA, "pauli-cones-device/1"):
            raise ValueError("unsupported device schema")
        for key in ("name", "family", "n_qubits", "edges"):
            if key not in data:
                raise ValueError(f"device description needs {key!r}")
        return cls(name=data["name"], family=data["family"],
                   topology=Topology(int(data["n_qubits"]), tuple(tuple(e) for e in data["edges"])),
                   native_gates=data.get("native_gates", {"one_qubit": [], "two_qubit": []}),
                   error_1q=data.get("error_1q"), error_2q=data.get("error_2q"),
                   readout=tuple(data["readout"]) if data.get("readout") else None,
                   t1=data.get("t1"), t2=data.get("t2"), time_1q=data.get("time_1q"), time_2q=data.get("time_2q"),
                   description=data.get("description", ""), source=data.get("source", ""),
                   parameters=data.get("parameters", {}),
                   coupler_errors={(int(a), int(b)): float(e) for a, b, e in data.get("coupler_errors", [])},
                   qubit_overrides={int(item["qubit"]): {k: v for k, v in item.items() if k != "qubit"}
                                    for item in data.get("qubit_overrides", [])},
                   crosstalk=data.get("crosstalk", {}),
                   coordinates=tuple(tuple(xy) for xy in data.get("coordinates", [])))

    def to_json(self, path=None):
        text = json.dumps(self.to_dict(), indent=1)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text


FAMILIES = {"superconducting", "trapped-ion", "neutral-atom", "topological", "annealer", "custom"}


# ---------------------------------------------------------------- layouts
def _heavy_hex_edges(rows, row_length):
    """Heavy-hex lattice: `rows` lines of qubits joined by bridge qubits every
    four columns, alternating offsets 0 and 2, with the first line missing its
    last column and the last line its first, numbered line by line as IBM does.
    rows=7, row_length=15 gives the 127-qubit Eagle layout (144 couplers)."""
    if rows < 2 or row_length < 5:
        raise ValueError("heavy-hex needs at least 2 rows of at least 5 qubits")
    lines, bridges, n = [], [], 0
    for r in range(rows):
        cols = range(row_length - 1) if r == 0 else range(1, row_length) if r == rows - 1 else range(row_length)
        lines.append({c: n + i for i, c in enumerate(cols)})
        n += len(cols)
        if r < rows - 1:
            offsets = [c for c in range(0 if r % 2 == 0 else 2, row_length, 4)]
            bridges.append([(c, n + i) for i, c in enumerate(offsets)])
            n += len(offsets)
    edges = []
    coordinates = [None] * n
    for r, line in enumerate(lines):
        cols = sorted(line)
        edges += [(line[a], line[b]) for a, b in zip(cols, cols[1:])]
        for c, q in line.items():
            coordinates[q] = (c, 2 * r)
    for r, row_bridges in enumerate(bridges):
        for c, q in row_bridges:
            if c not in lines[r] or c not in lines[r + 1]:
                raise ValueError("bridge column falls outside a trimmed row; use an odd number of rows and row_length = 4k + 3")
            edges += [(lines[r][c], q), (q, lines[r + 1][c])]
            coordinates[q] = (c, 2 * r + 1)
    return n, edges, coordinates


def _grid_edges(rows, columns):
    t = Topology.grid(rows, columns)
    return t.n_qubits, list(t.edges), [(c, r) for r in range(rows) for c in range(columns)]


def _radius_edges(rows, columns, radius):
    edges = []
    reach = int(math.floor(radius))
    for r in range(rows):
        for c in range(columns):
            for dr in range(0, reach + 1):
                for dc in range(-reach, reach + 1):
                    if (dr, dc) <= (0, 0) or math.hypot(dr, dc) > radius + 1e-12:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < columns:
                        edges.append((r * columns + c, rr * columns + cc))
    return rows * columns, edges, [(c, r) for r in range(rows) for c in range(columns)]


def _chimera_edges(m, n, t):
    """Chimera C(m, n, t): an m x n grid of K_{t,t} cells, D-Wave's linear numbering."""
    def index(i, j, u, k):
        return ((i * n + j) * 2 + u) * t + k
    edges = []
    for i in range(m):
        for j in range(n):
            edges += [(index(i, j, 0, a), index(i, j, 1, b)) for a in range(t) for b in range(t)]
            for k in range(t):
                if i + 1 < m:
                    edges.append((index(i, j, 0, k), index(i + 1, j, 0, k)))  # vertical qubits chain down
                if j + 1 < n:
                    edges.append((index(i, j, 1, k), index(i, j + 1, 1, k)))  # horizontal qubits chain right
    # Each cell draws its vertical qubits as a column and its horizontal qubits as a row.
    coordinates = [None] * (2 * m * n * t)
    for i in range(m):
        for j in range(n):
            for k in range(t):
                coordinates[index(i, j, 0, k)] = (j * (t + 2) + t, i * (t + 2) + k)
                coordinates[index(i, j, 1, k)] = (j * (t + 2) + k, i * (t + 2) + t)
    return 2 * m * n * t, edges, coordinates


@lru_cache(maxsize=8)
def _dwave_graph(kind, size, t):
    try:
        import dwave_networkx as dnx
    except ImportError as exc:
        from .interop import ConversionError
        raise ConversionError(f"the {kind} layout needs dwave-networkx", "pip install dwave-networkx") from exc
    graph = dnx.pegasus_graph(size) if kind == "pegasus" else dnx.zephyr_graph(size, t)
    label = {node: i for i, node in enumerate(sorted(graph.nodes))}
    layout = (dnx.pegasus_layout if kind == "pegasus" else dnx.zephyr_layout)(graph, scale=len(label) ** 0.5)
    coordinates = [None] * len(label)
    for node, i in label.items():
        x, y = layout[node]
        coordinates[i] = (round(float(x), 6), round(float(-y), 6))
    return len(label), [(label[a], label[b]) for a, b in graph.edges], coordinates


def _device(name, family, size_edges, gates, noise, description, parameters, source=ILLUSTRATIVE):
    n, edges, coordinates = size_edges
    return Device(name=name, family=family, topology=Topology(n, tuple(edges)), native_gates=gates,
                  description=description, source=source, parameters=parameters,
                  coordinates=tuple(coordinates), **noise)


def spring_layout(n, edges, iterations=200, seed=0):
    """A deterministic Fruchterman-Reingold layout, scaled so neighbours sit about one unit apart."""
    import numpy as np
    if n == 1:
        return [(0.0, 0.0)]
    rng = np.random.default_rng(seed)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[np.cos(angle), np.sin(angle)] * math.sqrt(n) + rng.normal(scale=0.01, size=(n, 2))
    if n <= 2000:
        a = np.array(edges, dtype=int).reshape(-1, 2)
        k, temperature = 1.0, math.sqrt(n)
        for _ in range(iterations):
            delta = pos[:, None, :] - pos[None, :, :]
            distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-9)
            force = (delta / distance[..., None] * (k * k / distance)[..., None]).sum(axis=1)
            if len(a):
                pull = pos[a[:, 0]] - pos[a[:, 1]]
                length = np.maximum(np.linalg.norm(pull, axis=1), 1e-9)[:, None]
                attraction = pull * length / k
                np.add.at(force, a[:, 0], -attraction)
                np.add.at(force, a[:, 1], attraction)
            step = np.linalg.norm(force, axis=1, keepdims=True)
            pos += force / np.maximum(step, 1e-9) * np.minimum(step, temperature)
            temperature *= 0.97
        if len(a):
            pos /= np.median(np.linalg.norm(pos[a[:, 0]] - pos[a[:, 1]], axis=1))
    pos -= pos.min(axis=0)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in pos]


SC_GATES = {"one_qubit": ["rz", "sx", "x"], "two_qubit": ["cz"]}


# ---------------------------------------------------------------- topological (Majorana tetrons)
# A tetron qubit computes by parity measurements: Z through one wire's parity, X through an
# interference loop across both wires, and joint parities between neighbouring tetrons. Clifford
# gates are sequences of these measurements; a T gate consumes a magic state; any other angle is
# synthesized from T gates. Idle tetrons suffer parity flips, not T1 decay: a Z-parity switch
# (lifetime tau_z) flips the bit, an X-parity switch (tau_x) flips the phase. The compiler reads
# these values from parameters["topological"].
TOPOLOGICAL_GATES = {"one_qubit": ["h", "s", "x", "z", "t"], "two_qubit": ["cz"]}
MAJORANA_SOURCES = {
    "tau_z_s": "Z-parity lifetime 22 +- 1 s on one wire of one Majorana 2 tetron, Pb/InAs, 50 mK "
               "(Microsoft, arXiv:2606.03884, 2026; not peer reviewed)",
    "measurement_time_s": "10 us pulsed readout integration on Majorana 2 (arXiv:2606.03884)",
    "tau_x_s": "X-parity lifetime 14.5 +- 0.3 us on a Majorana 1-generation InAs-Al tetron "
               "(arXiv:2507.08795); not published for Majorana 2",
    "measurement_error_z": "single-shot Z assignment error 0.5% on a Majorana 1-generation tetron "
                           "(arXiv:2507.08795); not published for Majorana 2",
    "measurement_error_x": "single-shot X assignment error 16% on a Majorana 1-generation tetron "
                           "(arXiv:2507.08795); not published for Majorana 2",
    "magic_state_error": "assumption: no magic state has been published; set equal to the X "
                         "measurement error, since injecting one needs an X-type measurement",
    "synthesis_precision": "assumption: each arbitrary rotation is synthesized to this precision from "
                           "about 3 log2(1/precision) T gates (Ross and Selinger, arXiv:1403.2975)",
    "measurements": "assumption: a one-qubit Clifford as one Z-type and one X-type measurement, CZ as a "
                    "joint ZZ parity plus one Z-type and one X-type measurement; set from your protocol",
}
MAJORANA_2 = {"tau_z_s": 22.0, "tau_x_s": 14.5e-6, "measurement_time_s": 10e-6, "measurement_error_z": 0.005,
              "measurement_error_x": 0.16, "magic_state_error": 0.16, "synthesis_precision": 1e-3}
# What-if values for designed tetron arrays: Microsoft's stated ~1 us operation scale, the measured
# Z-parity lifetime, and targets for the rest. Not data from any device.
TETRON_TARGETS = {"tau_z_s": 22.0, "tau_x_s": 1e-3, "measurement_time_s": 1e-6, "measurement_error_z": 1e-3,
                  "measurement_error_x": 1e-3, "magic_state_error": 1e-3, "synthesis_precision": 1e-3}


def _majorana_2(**model):
    return _topological(
        "microsoft_majorana_2", _tetron_grid(2, 2),
        "Microsoft Majorana 2 (June 2026): the published unit cell of four tetron qubits (two grounded, two "
        "floating) on a Pb/InAs/InAsSb topoconductor, with joint parity measurements between neighbours. "
        "Microsoft has published one wire's Z-parity lifetime and its readout time, not gate fidelities; "
        "values it has not published come from the Majorana 1-generation tetron or are labelled "
        "assumptions in parameters['sources'].",
        {**MAJORANA_2, **{k: v for k, v in model.items() if v is not None}},
        "Microsoft, arXiv:2606.03884 (Majorana 2) and arXiv:2507.08795 (tetron measurements); see "
        "parameters['sources']. Independent physicists dispute that these devices are topological qubits.",
        {"sources": MAJORANA_SOURCES,
         "tetrons": ["AB (grounded)", "BA (grounded)", "AA (floating)", "BB (floating)"],
         "tetron_geometry_um": {"wire_length": 3.5, "wire_width": 0.035, "backbone_length": 1.0,
                                "backbone_width": 0.02, "dots_per_tetron": 5}})


def _tetron_array(rows=2, columns=4, **model):
    return _topological(
        f"tetron-array-{rows}x{columns}", _tetron_grid(rows, columns),
        "Array of Majorana tetron qubits with joint parity measurements between neighbours, as in "
        "Microsoft's roadmap (4x2 for error detection, 27x13 for lattice surgery). Noise values are what-if "
        "targets, not data from any device.",
        {**TETRON_TARGETS, **{k: v for k, v in model.items() if v is not None}}, ILLUSTRATIVE,
        {"rows": rows, "columns": columns, "sources": {"roadmap": "Microsoft, arXiv:2502.12252 (CC BY 4.0)"}})


def _topological(name, size_edges, description, model, source, extra=None):
    """A tetron chip whose gate errors and times follow from its parity measurements."""
    ez, ex, t = model["measurement_error_z"], model["measurement_error_x"], model["measurement_time_s"]
    noise = dict(error_1q=1 - (1 - ez) * (1 - ex),          # one Z-type and one X-type measurement
                 error_2q=1 - (1 - ez) ** 2 * (1 - ex),     # a joint ZZ, one Z-type and one X-type
                 readout=(ez, ez), t1=None, t2=None, time_1q=2 * t, time_2q=3 * t)
    return _device(name, "topological", size_edges, TOPOLOGICAL_GATES, noise, description,
                   {**(extra or {}), "topological": dict(model)}, source=source)


def _tetron_grid(rows, columns):
    """Tetrons on a rectangular array, joint parities between nearest neighbours; coordinates in tetron
    pitches, wider than tall, as cad draws each tetron as an H of two wires."""
    n, edges, _ = _grid_edges(rows, columns)
    return n, edges, [(c * 1.6, r) for r in range(rows) for c in range(columns)]


def _annealer(name, size_edges, description, parameters):
    from .annealing import ANNEALER_PROPERTIES
    return _device(name, "annealer", size_edges, {"one_qubit": [], "two_qubit": []}, dict(), description,
                   {**parameters, "annealer": dict(ANNEALER_PROPERTIES)},
                   source="Graph from dwave-networkx (Apache-2.0); h, J and annealing-time ranges as published "
                          "for D-Wave's Advantage-generation solvers.")

# name -> (builder(**params), default params, one-line description)
CATALOG = {
    "heavy-hex-127": (lambda: _device(
        "heavy-hex-127", "superconducting", _heavy_hex_edges(7, 15),
        {"one_qubit": ["rz", "sx", "x"], "two_qubit": ["ecr"]}, SUPERCONDUCTING,
        "IBM Eagle-style 127-qubit heavy-hex chip, numbered as on the device; native ECR.", {}),
        {}, "IBM Eagle-style 127-qubit heavy-hex layout (the kicked-Ising benchmark's chip)"),
    "heavy-hex": (lambda rows=7, row_length=15: _device(
        f"heavy-hex-{rows}x{row_length}", "superconducting", _heavy_hex_edges(rows, row_length), SC_GATES,
        SUPERCONDUCTING, "Heavy-hex superconducting lattice of any size (IBM-style); native CZ.",
        {"rows": rows, "row_length": row_length}),
        {"rows": 7, "row_length": 15}, "Heavy-hex superconducting lattice; row_length = 4k + 3"),
    "square-grid": (lambda rows=8, columns=8: _device(
        f"square-grid-{rows}x{columns}", "superconducting", _grid_edges(rows, columns), SC_GATES,
        SUPERCONDUCTING, "Square-lattice superconducting chip (Google- and Rigetti-style); native CZ.",
        {"rows": rows, "columns": columns}),
        {"rows": 8, "columns": 8}, "Square-lattice superconducting chip, nearest-neighbour couplers"),
    "line": (lambda n=16: _device(
        f"line-{n}", "superconducting", (n, [(i, i + 1) for i in range(n - 1)], [(i, 0) for i in range(n)]),
        SC_GATES, SUPERCONDUCTING,
        "Linear chain of qubits with nearest-neighbour couplers.", {"n": n}),
        {"n": 16}, "Linear chain"),
    "all-to-all-ions": (lambda n=32: _device(
        f"all-to-all-ions-{n}", "trapped-ion",
        (n, [(a, b) for a in range(n) for b in range(a + 1, n)], [(i, 0) for i in range(n)]),  # ions sit in a line
        {"one_qubit": ["rz", "rx", "ry"], "two_qubit": ["rxx"]}, TRAPPED_ION,
        "Trapped-ion chip with all-to-all connectivity (IonQ- and Quantinuum-style); native "
        "Molmer-Sorensen (RXX) gate.", {"n": n}),
        {"n": 32}, "Trapped ions, all-to-all, native RXX (Molmer-Sorensen)"),
    "neutral-atom-grid": (lambda rows=10, columns=10, radius=1.5: _device(
        f"neutral-atom-grid-{rows}x{columns}", "neutral-atom", _radius_edges(rows, columns, radius),
        {"one_qubit": ["rz", "rx"], "two_qubit": ["cz"]}, NEUTRAL_ATOM,
        "Neutral-atom array (QuEra- and Pasqal-style) coupling atoms within `radius` lattice spacings "
        "(1.5 includes diagonals); native CZ.", {"rows": rows, "columns": columns, "radius": radius}),
        {"rows": 10, "columns": 10, "radius": 1.5}, "Neutral-atom array, Rydberg coupling within a radius"),
    "microsoft_majorana_2": (_majorana_2, {k: None for k in MAJORANA_2},
        "Microsoft Majorana 2: four tetron qubits, measurement-based, published parity lifetimes (topological)"),
    "tetron-array": (_tetron_array, {"rows": 2, "columns": 4, **{k: None for k in TETRON_TARGETS}},
        "Majorana tetron array of any size, measurement-based (topological; Microsoft roadmap)"),
    "chimera": (lambda m=16, n=None, t=4: _device(
        f"chimera-{m}x{n or m}x{t}", "annealer", _chimera_edges(m, n or m, t), {"one_qubit": [], "two_qubit": []},
        dict(), "D-Wave-style Chimera annealer graph: an m x n grid of K_{t,t} unit cells.",
        {"m": m, "n": n or m, "t": t}, source="Published Chimera graph definition."),
        {"m": 16, "n": None, "t": 4}, "D-Wave-style Chimera annealer graph"),
    "pegasus": (lambda m=16: _device(
        f"pegasus-{m}", "annealer", _dwave_graph("pegasus", m, None), {"one_qubit": [], "two_qubit": []}, dict(),
        "D-Wave-style Pegasus annealer graph, built by dwave-networkx.", {"m": m},
        source="dwave-networkx pegasus_graph."),
        {"m": 16}, "D-Wave-style Pegasus annealer graph (needs dwave-networkx)"),
    "dwave-advantage": (lambda: _annealer(
        "dwave-advantage", _dwave_graph("pegasus", 16, None),
        "D-Wave Advantage-style annealer: Pegasus P16, 5,640 qubits, 15 couplers per qubit, with the "
        "Advantage generation's published h and J ranges and annealing times. Every qubit is present; "
        "a real solver reports a few inactive ones.", {"m": 16}),
        {}, "D-Wave Advantage-style annealer, Pegasus P16, 5,640 qubits (needs dwave-networkx)"),
    "dwave-advantage2": (lambda: _annealer(
        "dwave-advantage2", _dwave_graph("zephyr", 12, 4),
        "D-Wave Advantage2-style annealer: Zephyr Z12, 4,800 qubits, 20 couplers per qubit, with the "
        "Advantage generation's published h and J ranges and annealing times. Every qubit is present.",
        {"m": 12, "t": 4}),
        {}, "D-Wave Advantage2-style annealer, Zephyr Z12, 4,800 qubits (needs dwave-networkx)"),
    "zephyr": (lambda m=12, t=4: _device(
        f"zephyr-{m}x{t}", "annealer", _dwave_graph("zephyr", m, t), {"one_qubit": [], "two_qubit": []}, dict(),
        "D-Wave-style Zephyr annealer graph, built by dwave-networkx.", {"m": m, "t": t},
        source="dwave-networkx zephyr_graph."),
        {"m": 12, "t": 4}, "D-Wave-style Zephyr annealer graph (needs dwave-networkx)"),
}


def catalog():
    """Every device: generated layouts (with their default parameters) and real chips (calibration snapshots)."""
    families = {"heavy-hex-127": "superconducting", "heavy-hex": "superconducting",
                "square-grid": "superconducting", "line": "superconducting", "all-to-all-ions": "trapped-ion",
                "neutral-atom-grid": "neutral-atom", "microsoft_majorana_2": "topological",
                "tetron-array": "topological", "chimera": "annealer", "pegasus": "annealer",
                "zephyr": "annealer", "dwave-advantage": "annealer", "dwave-advantage2": "annealer"}
    generated = [{"name": name, "kind": "layout", "family": families[name], "description": text,
                  "parameters": defaults} for name, (_, defaults, text) in CATALOG.items()]
    real = [{"name": name, "kind": "real", "family": d["family"], "n_qubits": d["n_qubits"],
             "description": d["description"], "snapshot": d["parameters"]["hardware"].get("snapshot_date"),
             "processor": d["parameters"]["hardware"].get("processor")} for name, d in sorted(real_devices().items())]
    return generated + real


@lru_cache(maxsize=1)
def real_devices():
    """Real chips from bundled calibration snapshots (IBM, Google): name -> device description."""
    path = resources.files("quantum_circuit_simulator") / "data" / "devices.json.gz"
    if not path.is_file():
        return {}
    with path.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        return json.load(f)["devices"]


def get(name, **parameters):
    """A generated catalogue layout, a real chip from a calibration snapshot, or a device JSON file."""
    if name.endswith(".json"):
        with open(name, encoding="utf-8") as f:
            return Device.from_dict(json.load(f))
    if name in real_devices():
        if parameters:
            raise ValueError(f"{name} is a real chip and takes no parameters; derive variants with "
                             "design.vary_qubits, Device.scaled_noise or Device.with_noise")
        return Device.from_dict(real_devices()[name])
    if name not in CATALOG:
        raise KeyError(f"unknown device {name!r}; run `quantum-circuit-simulator devices` for layouts and real "
                       "chips, or pass a .json file")
    builder, defaults, _ = CATALOG[name]
    unknown = set(parameters) - set(defaults)
    if unknown:
        raise ValueError(f"{name} takes parameters {sorted(defaults)}, not {sorted(unknown)}")
    chip = builder(**parameters)
    # Record the generator, so design.expand can grow the chip with the same architecture.
    return replace(chip, parameters={**chip.parameters, "catalog": name, "catalog_parameters": {**defaults, **parameters}})
