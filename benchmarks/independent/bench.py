"""Matched head-to-head benchmarks: Quantum Circuit Simulator vs Qulacs, Aer, pauli-prop,
PennyLane-Lightning (adjoint), quimb (tensor network).

Every (workload, tool) pair runs in a fresh subprocess; the parent records cold
wall time (interpreter start -> first value) and kills on timeout. Inside the
worker: build, prepare, first value, warm value (median of 3 fresh vectors),
first and warm gradient, peak RSS. All values are cross-checked afterwards.

    python bench.py run <out.jsonl> [workload ...]
    python bench.py worker <workload> <tool>     (internal)
"""
import json, math, os, resource, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.environ.get("BENCH_THREADS", "1")
REPS = int(os.environ.get("BENCH_REPS", "3"))
THREADS = {k: NT for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "QULACS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS")}


# ------------------------------------------------------------------ workloads
def grid_edges(rows, cols, layer):
    if layer % 2 == 0:
        return [(r * cols + k, r * cols + k + 1) for r in range(rows) for k in range((layer // 2) % 2, cols - 1, 2)]
    return [(k * cols + r, (k + 1) * cols + r) for r in range(cols) for k in range((layer // 2) % 2, rows - 1, 2)]


def hea(rows, cols, layers, nparams, p2):
    """Hardware-efficient ansatz on a grid: RY/RZ on every wire, CZ brickwork."""
    rng = np.random.default_rng(rows * 100 + cols * 10 + layers)
    n = rows * cols
    ops = []
    for layer in range(layers):
        for q in range(n):
            ops.append(("RY", q, float(rng.uniform(-1, 1)), (2 * layer) % nparams))
            ops.append(("RZ", q, float(rng.uniform(-1, 1)), (2 * layer + 1) % nparams))
        for a, b in grid_edges(rows, cols, layer):
            ops.append(("CZ", a, b))
            if p2:
                ops.append(("depol", (a, b), p2))
    return n, ops


def tfim(rows, cols):
    n = rows * cols
    obs = []
    for layer in (0, 1, 2, 3):
        for a, b in grid_edges(rows, cols, layer):
            obs.append((-1.0, {a: "Z", b: "Z"}))
    obs = list({tuple(sorted(p.items())): (c, p) for c, p in obs}.values())
    obs += [(-0.5, {q: "X"}) for q in range(n)]
    return obs


def mean_z(n):
    return [(1.0 / n, {q: "Z"}) for q in range(n)]


def centre_z(rows, cols):
    return [(1.0, {(rows // 2) * cols + cols // 2: "Z"})]


WORKLOADS = {}


def W(name, rows, cols, layers, nparams, p2, obs, note, p1=0.0):
    n, ops = hea(rows, cols, layers, nparams, p2)
    if p1:  # single-qubit depolarizing on both wires after every CZ (PauliPropagation.jl has no 2-wire channel)
        out = []
        for op in ops:
            out.append(op)
            if op[0] == "CZ":
                out += [("depol", (op[1],), p1), ("depol", (op[2],), p1)]
        ops = out
    o = {"tfim": tfim(rows, cols), "meanz": mean_z(n), "centre": centre_z(rows, cols)}[obs]
    WORKLOADS[name] = dict(n=n, ops=ops, nparams=nparams, obs=o, noisy=bool(p2 or p1), note=note,
                           rows=rows, cols=cols, layers=layers, observable=obs)


# A. noiseless VQE energy + full gradient (statevector/adjoint territory)
W("A_vqe_3x4_L2", 3, 4, 2, 4, 0, "tfim", "12q TFIM energy, 2 layers, 4 shared params")
W("A_vqe_4x5_L2", 4, 5, 2, 4, 0, "tfim", "20q TFIM energy, 2 layers")
W("A_vqe_4x5_L4", 4, 5, 4, 8, 0, "tfim", "20q TFIM energy, 4 layers")
W("A_vqe_4x6_L6", 4, 6, 6, 12, 0, "tfim", "24q TFIM energy, 6 layers")
# B. noisy local observables on grids (Pauli-propagation territory)
W("B_noisy_2x5_L4", 2, 5, 4, 8, 0.01, "meanz", "10q noisy mean-Z, density-matrix feasible")
W("B_noisy_6x6_L3", 6, 6, 3, 6, 0.01, "meanz", "36q noisy mean-Z, 3 layers")
W("B_noisy_6x6_L5", 6, 6, 5, 10, 0.01, "meanz", "36q noisy mean-Z, 5 layers")
W("B_noisy_20x20_L3", 20, 20, 3, 6, 0.01, "meanz", "400q noisy mean-Z, 3 layers")
W("B_noisy_32x32_L3", 32, 32, 3, 6, 0.01, "meanz", "1024q noisy mean-Z, 3 layers")
# C. noiseless single local observable, growing depth (TN light-cone territory)
W("C_centre_10x10_L4", 10, 10, 4, 8, 0, "centre", "100q centre Z, 4 layers")
W("C_centre_10x10_L6", 10, 10, 6, 12, 0, "centre", "100q centre Z, 6 layers")
W("C_centre_10x10_L8", 10, 10, 8, 16, 0, "centre", "100q centre Z, 8 layers")
W("C_centre_32x32_L6", 32, 32, 6, 12, 0, "centre", "1024q centre Z, 6 layers")

# D. single-qubit depolarizing variants, matched to PauliPropagation.jl's channel set
for _r, _c, _l in ((2, 5, 4), (6, 6, 3), (6, 6, 5), (20, 20, 3), (32, 32, 3), (6, 6, 6), (10, 10, 6)):
    W(f"D_noisy1q_{_r}x{_c}_L{_l}", _r, _c, _l, 2 * _l, 0, "meanz", "1q-depolarizing mean-Z", p1=0.0075)


def eagle_edges():
    """IBM Eagle's 127-qubit heavy-hex coupling map (144 edges): seven rows
    joined by bridge qubits, numbered row by row as on the device."""
    rows, edges, n = [], [], 0
    for r in range(7):
        cols = range(14) if r == 0 else range(1, 15) if r == 6 else range(15)
        rows.append({c: n + i for i, c in enumerate(cols)})
        n += len(cols)
        if r < 6:
            bridges = (0, 4, 8, 12) if r % 2 == 0 else (2, 6, 10, 14)
            rows.append(("bridge", bridges, list(range(n, n + 4))))
            n += 4
    lines = [x for x in rows if isinstance(x, dict)]
    for line in lines:
        cols = sorted(line)
        edges += [(line[a], line[b]) for a, b in zip(cols, cols[1:])]
    for i, x in enumerate(rows):
        if isinstance(x, tuple):
            above, below = rows[i - 1], rows[i + 1]
            for c, q in zip(x[1], x[2]):
                edges += [(above[c], q), (q, below[c])]
    assert n == 127 and len(edges) == 144
    return edges


def kicked_ising(steps, theta_h, observable):
    """Kim et al. (Nature 2023) kicked Ising on Eagle: per step RX(theta_h) on
    every qubit, then RZZ(-pi/2) on every edge. All ZZ terms commute, and
    RZZ(-pi/2) = CZ followed by RZ(-pi/2) on both qubits up to a global phase,
    so each step is all CZs then one RZ(-deg(q) pi/2) per qubit. theta_h is the
    one shared parameter."""
    edges, n = eagle_edges(), 127
    degree = [0] * n
    for a, b in edges:
        degree[a] += 1; degree[b] += 1
    ops = []
    for _ in range(steps):
        ops += [("RX", q, theta_h, 0) for q in range(n)]
        ops += [("CZ", a, b) for a, b in edges]
        ops += [("RZ", q, -degree[q] * math.pi / 2, -1) for q in range(n)]
    obs = [(1.0 / n, {q: "Z"}) for q in range(n)] if observable == "mz" else [(1.0, {62: "Z"})]
    return n, ops, obs


for _steps, _theta, _obs in ((5, 0.0, "mz"), (5, math.pi / 8, "mz"), (5, math.pi / 4, "mz"),
                             (5, 3 * math.pi / 8, "mz"), (5, math.pi / 2, "mz"), (20, math.pi / 8, "z62")):
    _n, _ops, _o = kicked_ising(_steps, _theta, _obs)
    WORKLOADS[f"K_eagle_{_obs}_S{_steps}_t{round(_theta / math.pi * 8)}pi8"] = dict(
        n=_n, ops=_ops, nparams=1, obs=_o, noisy=False, layers=_steps, observable=_obs,
        note="IBM Eagle kicked Ising (Kim et al. 2023), theta_h shared", shift_scale=1e-3, zero_first=True)


def _wires(op):
    return (op[1], op[2]) if op[0] == "CZ" else tuple(op[1]) if op[0] == "depol" else (op[1],)


def cone_parts(w, mode):
    """Cut a workload down to its observable's backward light cone, as an expert
    would before handing it to a tool that does not do this itself (Quantum
    Circuit Simulator does). `cone` keeps one circuit for the union of every
    term's cone; `split` gives each observable term its own circuit, summed
    afterwards. Each part is relabelled onto the fewest qubits, the smallest
    system on which the cone never reaches an edge. A two-wire depolarizing
    channel with one wire outside the cone acts on the inside wire as a
    one-wire channel of strength 4p/5: both damp a Pauli that is non-identity
    there by 1 - 16p/15."""
    groups = [w["obs"]] if mode == "cone" else [[t] for t in w["obs"]]
    parts = []
    for obs in groups:
        live = {q for _, p in obs for q in p}
        kept = []
        for op in reversed(w["ops"]):
            inside = [q for q in _wires(op) if q in live]
            if not inside:
                continue
            if op[0] == "CZ":
                live.update(_wires(op))
            elif op[0] == "depol" and len(inside) < len(op[1]):
                op = ("depol", tuple(inside), 4 * op[2] / 5)
            kept.append(op)
        kept.reverse()
        used = sorted({q for _, p in obs for q in p} | {q for op in kept for q in _wires(op)})
        m = {q: i for i, q in enumerate(used)}
        ops = [(o[0], m[o[1]], m[o[2]]) if o[0] == "CZ" else ("depol", tuple(m[q] for q in o[1]), o[2])
               if o[0] == "depol" else (o[0], m[o[1]], *o[2:]) for o in kept]
        parts.append(dict(w, n=len(used), ops=ops, obs=[(c, {m[q]: a for q, a in p.items()}) for c, p in obs]))
    return parts


CONES = os.environ.get("BENCH_CONES", "")  # "", "cone" or "split": see cone_parts

TOOLS =os.environ.get("BENCH_TOOLS", "").split(",") if os.environ.get("BENCH_TOOLS") else ["pauli_cones", "pauli_cones_graph", "qulacs", "aer", "lightning", "pauli_prop", "quimb"]


def eligible(w, tool):
    n, noisy = (max(p["n"] for p in cone_parts(w, CONES)) if CONES else w["n"]), w["noisy"]
    if tool in ("qulacs", "aer"):
        return n <= (10 if noisy else 26)
    if tool == "lightning":
        return not noisy and n <= 26
    if tool in ("quimb", "spd"):
        return not noisy
    if tool == "pauli_cones_graph":
        return n <= 256 or bool(os.environ.get("BENCH_GRAPH_WIDE"))
    return True


# ------------------------------------------------------------------ adapters
class PauliCones:
    def __init__(self, w, backend="native"):
        from quantum_circuit_simulator import Circuit
        t = time.perf_counter()
        c = Circuit(w["n"], w["nparams"], max_terms=1_000_000)
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                c.gate(op[0], [op[1]], theta=op[2], param=op[3])
            elif op[0] == "CZ":
                c.gate("CZ", [op[1], op[2]])
            elif op[0] == "depol":
                c.depolarize(list(op[1]), op[2])
        self.build = time.perf_counter() - t
        t = time.perf_counter()
        threads = int(os.environ.get("BENCH_PC_THREADS", "1"))
        extra = {"threads": threads} if threads != 1 else {}  # older builds have no threads option
        self.plan = c.prepare(w["obs"], backend=backend, graph_max_nodes=5_000_000, graph_max_edges=20_000_000, **extra)
        self.prepare = time.perf_counter() - t

    def value(self, s): return self.plan.evaluate_value(list(s))

    def grad(self, s):
        r = self.plan.evaluate(list(s)); return r.value, list(r.gradient)


class Qulacs:
    """Qulacs uses RX(t)=exp(+i t X/2); angles are negated. Gradient: backprop (adjoint)."""
    def __init__(self, w):
        import qulacs
        from qulacs import ParametricQuantumCircuit, Observable, QuantumState, DensityMatrix
        from qulacs.gate import DepolarizingNoise, TwoQubitDepolarizingNoise
        self.w = w
        t = time.perf_counter()
        n = w["n"]
        self.c = ParametricQuantumCircuit(n)
        self.pidx = []
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                getattr(self.c, f"add_parametric_{op[0]}_gate")(op[1], -op[2])
                self.pidx.append((op[2], op[3]))
            elif op[0] == "CZ":
                self.c.add_CZ_gate(op[1], op[2])
            elif op[0] == "depol":
                # qulacs depolarizing(p): with prob p a uniformly random non-identity Pauli
                if len(op[1]) == 2:
                    self.c.add_gate(TwoQubitDepolarizingNoise(op[1][0], op[1][1], op[2]))
                else:
                    self.c.add_gate(DepolarizingNoise(op[1][0], op[2]))
        self.obs = Observable(n)
        for coef, p in w["obs"]:
            self.obs.add_operator(coef, " ".join(f"{a} {q}" for q, a in p.items()))
        self.state = DensityMatrix(n) if w["noisy"] else QuantumState(n)
        self.build = time.perf_counter() - t
        self.prepare = 0.0

    def _bind(self, s):
        for i, (theta, par) in enumerate(self.pidx):
            self.c.set_parameter(i, -(theta + (s[par] if par >= 0 else 0.)))

    def value(self, s):
        self._bind(s)
        self.state.set_zero_state()
        self.c.update_quantum_state(self.state)
        return self.obs.get_expectation_value(self.state).real

    def grad(self, s):
        if self.w["noisy"]:
            raise NotImplementedError("qulacs backprop is statevector-only")
        v = self.value(s)
        g = self.c.backprop(self.obs)
        out = [0.] * self.w["nparams"]
        for gi, (_, par) in zip(g, self.pidx):
            if par >= 0:
                out[par] -= gi
        return v, out


class Aer:
    def __init__(self, w):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import depolarizing_error
        t = time.perf_counter()
        n = w["n"]
        self.w = w
        p = ParameterVector("p", w["nparams"])
        qc = QuantumCircuit(n)
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                getattr(qc, op[0].lower())(op[2] + p[op[3]] if op[3] >= 0 else op[2], op[1])
            elif op[0] == "CZ":
                qc.cz(op[1], op[2])
            elif op[0] == "depol":
                k = len(op[1])  # Aer lambda = p 4^k / (4^k - 1)
                qc.append(depolarizing_error(op[2] * 4 ** k / (4 ** k - 1), k), list(op[1]))
        terms = [("".join(pp.get(q, "I") for q in reversed(range(n))), c) for c, pp in w["obs"]]
        qc.save_expectation_value(SparsePauliOp.from_list(terms), list(range(n)))
        self.qc, self.p = qc, p
        self.sim = AerSimulator(method="density_matrix" if w["noisy"] else "statevector",
                                max_parallel_threads=int(NT))
        self.build = time.perf_counter() - t
        t = time.perf_counter()
        self.qc = qc  # no transpile: Aer runs these instructions natively
        self.prepare = time.perf_counter() - t

    def value(self, s):
        b = self.qc.assign_parameters(dict(zip(self.p, s)))
        return float(self.sim.run(b, shots=1).result().data(0)["expectation_value"])

    def grad(self, s):
        raise NotImplementedError("Aer has no built-in adjoint gradient; parameter shift not timed")


class Lightning:
    def __init__(self, w):
        import pennylane as qml
        t = time.perf_counter()
        n = w["n"]
        self.w = w
        dev = qml.device("lightning.qubit", wires=n)
        H = qml.Hamiltonian([c for c, _ in w["obs"]],
                            [qml.prod(*[getattr(qml, "Pauli" + a)(q) for q, a in p.items()]) if len(p) > 1
                             else getattr(qml, "Pauli" + list(p.values())[0])(list(p)[0]) for _, p in w["obs"]])
        ops = w["ops"]

        def circuit(params):
            for op in ops:
                if op[0] in ("RX", "RY", "RZ"):
                    getattr(qml, op[0])(op[2] + params[op[3]] if op[3] >= 0 else op[2], wires=op[1])
                elif op[0] == "CZ":
                    qml.CZ(wires=[op[1], op[2]])
            return qml.expval(H)
        from pennylane import numpy as pnp
        self.pnp = pnp
        self.f = qml.QNode(circuit, dev, diff_method="adjoint")
        self.g = qml.grad(self.f)
        self.build = time.perf_counter() - t
        self.prepare = 0.0

    def value(self, s): return float(self.f(self.pnp.array(s, requires_grad=False)))

    def grad(self, s):
        x = self.pnp.array(s, requires_grad=True)
        return float(self.f(x)), [float(v) for v in self.g(x)]


class PauliProp:
    """IBM pauli-prop, exact (no truncation, atol=0); depolarizing as Pauli-Lindblad."""
    def __init__(self, w):
        from itertools import product
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer.noise import PauliLindbladError
        from pauli_prop.propagation import circuit_to_rotation_gates, evolve_through_cliffords
        t = time.perf_counter()
        n = w["n"]
        self.w = w
        qc = QuantumCircuit(n)
        self.rot = []
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                getattr(qc, op[0].lower())(op[2], op[1])  # numeric; rebinding below
                self.rot.append((op[2], op[3]))
            elif op[0] == "CZ":
                qc.cz(op[1], op[2])
            elif op[0] == "depol":
                # k-wire depolarizing damps every non-identity Pauli by
                # 1 - p 4^k/(4^k-1); with equal Lindblad rates on all 4^k-1
                # generators, each Pauli anticommutes with 4^k/2 of them.
                k = len(op[1]); d = 4 ** k
                scale = 1 - op[2] * d / (d - 1)
                labels = ["".join(l) for l in product("IXYZ", repeat=k) if set(l) != {"I"}]
                qc.append(PauliLindbladError(labels, [-math.log(scale) / d] * (d - 1)), list(op[1]))
        terms = [("".join(pp.get(q, "I") for q in reversed(range(n))), c) for c, pp in w["obs"]]
        self.obs = SparsePauliOp.from_list(terms)
        self.build = time.perf_counter() - t
        t = time.perf_counter()
        self.cliff, transformed = evolve_through_cliffords(qc)
        self.rg = circuit_to_rotation_gates(transformed)
        # rotation order is preserved; each rotation maps to one original rotation
        assert len(self.rg.thetas) >= len(self.rot)
        self.base = list(self.rg.thetas)
        self.SparsePauliOp = SparsePauliOp
        self.prepare = time.perf_counter() - t
        self.nrot = len(self.rot)

    def value(self, s):
        from pauli_prop.propagation import propagate_through_rotation_gates
        th = list(self.base)
        # rotations from gates come first in order; noise generators follow their own order.
        # Signs of transformed generators are absorbed in base; additive shifts keep sign:
        k = 0
        for i, b in enumerate(self.base):
            pass
        for i, (theta, par) in enumerate(self.rot):
            th[self.map[i]] = self.sign[i] * (theta + (s[par] if par >= 0 else 0.))
        self.rg.thetas[:] = th
        # BENCH_PP_ATOL > 0 runs pauli-prop truncated, its intended mode.
        atol = float(os.environ.get("BENCH_PP_ATOL", "0"))
        ev, disc = propagate_through_rotation_gates(self.obs, self.rg, max_terms=10_000_000, atol=atol, frame="h")
        if disc and not atol:
            raise RuntimeError(f"pauli-prop truncated ({disc})")
        ev = self.SparsePauliOp(ev.paulis.evolve(self.cliff, frame="h"), ev.coeffs, copy=False,
                                ignore_pauli_phase=False)
        return float(np.sum(ev.coeffs[~np.any(ev.paulis.x, axis=1)]).real)

    def grad(self, s):
        raise NotImplementedError("pauli-prop has no gradient API; parameter shift not timed")


class Spd:
    """Sparse Pauli dynamics (Begusic, Gray and Chan, Sci. Adv. 10, eadk4321
    (2024); github.com/tbegusic/spd, on SPD_PATH) as in its IBM example:
    Clifford parts of every rotation are moved into the Pauli frame once,
    then each evaluation propagates only the remaining rotations. A CZ is the
    rotations exp(-i pi/4 Z_a), exp(-i pi/4 Z_b) and exp(+i pi/4 Z_a Z_b) up to
    a global phase. Noise channels are unsupported. BENCH_PP_ATOL sets its
    coefficient threshold."""
    def __init__(self, w):
        if w["noisy"]:
            raise NotImplementedError("spd has no noise channels")
        sys.path.insert(0, os.environ["SPD_PATH"])
        from qiskit.quantum_info import PauliList
        from spd.OperatorSequence import OperatorSequence
        from spd.SparsePauliDynamics import Simulation
        t = time.perf_counter()
        n = w["n"]
        self.w, self.Simulation = w, Simulation

        def label(paulis):
            chars = ["I"] * n
            for q, a in paulis.items():
                chars[n - 1 - q] = a
            return "".join(chars)

        # (Pauli, base coefficient, parameter): exp(-i c P), c = theta / 2.
        rotations = []
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                rotations.append((label({op[1]: op[0][1]}), op[2] / 2, op[3]))
            elif op[0] == "CZ":
                a, b = op[1], op[2]
                rotations += [(label({a: "Z"}), math.pi / 4, -1), (label({b: "Z"}), math.pi / 4, -1),
                              (label({a: "Z", b: "Z"}), -math.pi / 4, -1)]
        rotations.reverse()  # spd applies the first operator to the observable first
        # Levels of mutually commuting rotations: one-wire rotations on distinct
        # wires, or diagonal (Z-only) rotations.
        levels, level, wires, diagonal = [], [], set(), True
        for r in rotations:
            support = {n - 1 - i for i, c in enumerate(r[0]) if c != "I"}
            z = set(r[0]) <= {"I", "Z"}
            if level and not ((diagonal and z) or not (support & wires)):
                levels.append(level); level, wires, diagonal = [], set(), True
            level.append(r); wires |= support; diagonal = diagonal and z
        levels.append(level)
        # Within a level every rotation commutes, so fixed rotations about the
        # same Pauli add up; in the kicked Ising the one-wire Z parts of the CZs
        # and the RZ gates cancel, leaving one ZZ rotation per coupler.
        merged = []
        for lv in levels:
            fixed, out = {}, []
            for r in lv:
                if r[2] >= 0:
                    out.append(r)
                else:
                    fixed[r[0]] = fixed.get(r[0], 0.) + r[1]
            out += [(pauli, c, -1) for pauli, c in fixed.items() if abs(c) > 1e-15]
            if out:
                merged.append(out)
        levels = merged or [[("I" * n, 0., -1)]]
        self.build = time.perf_counter() - t
        t = time.perf_counter()
        seq = OperatorSequence([PauliList([r[0] for r in lv]) for lv in levels],
                               [[r[1] for r in lv] for lv in levels])
        seq.reduce_clifford()
        keep = [(k, j) for k, lv in enumerate(levels) for j, r in enumerate(lv) if r[2] >= 0 or seq.coeffs[k][j] != 0.]
        self.ops = PauliList([seq.ops[k][j] for k, j in keep]) if keep else None
        self.residual = np.array([seq.coeffs[k][j] for k, j in keep], dtype=float)
        self.param = np.array([levels[k][j][2] for k, j in keep], dtype=int)
        self.obs = seq.clifford_ops.apply_to_PauliList(PauliList([label(p) for _, p in w["obs"]]))
        self.obs_coeffs = np.array([c for c, _ in w["obs"]], dtype=float)
        self.threshold = float(os.environ.get("BENCH_PP_ATOL", "0"))
        self.prepare = time.perf_counter() - t

    def value(self, s):
        from qiskit.quantum_info import SparsePauliOp
        shift = np.where(self.param >= 0, np.asarray(list(s) + [0.], dtype=float)[self.param] / 2, 0.)
        observable = SparsePauliOp(self.obs, self.obs_coeffs)
        ops = SparsePauliOp(self.ops, self.residual + shift) if self.ops is not None else SparsePauliOp(["I" * self.w["n"]], [0.])
        return float(np.real(self.Simulation.from_pauli_list(observable, ops, threshold=self.threshold).run_circuit()))

    def grad(self, s):
        raise NotImplementedError("spd has no gradient")


class Quimb:
    def __init__(self, w):
        import quimb.tensor as qtn
        import quimb as qu
        t = time.perf_counter()
        self.w = w
        self.qtn, self.qu = qtn, qu
        self.build = time.perf_counter() - t
        self.prepare = 0.0

    def _circ(self, s):
        c = self.qtn.Circuit(self.w["n"])
        for op in self.w["ops"]:
            if op[0] in ("RX", "RY", "RZ"):
                c.apply_gate(op[0], op[2] + (s[op[3]] if op[3] >= 0 else 0.), op[1])
            elif op[0] == "CZ":
                c.apply_gate("CZ", op[1], op[2])
        return c

    def value(self, s):
        c = self._circ(s)
        P = {"X": self.qu.pauli("X"), "Y": self.qu.pauli("Y"), "Z": self.qu.pauli("Z")}
        tot = 0.0
        for coef, p in self.w["obs"]:
            where = tuple(p)
            G = P[p[where[0]]]
            for q in where[1:]:
                G = self.qu.kron(G, P[p[q]])
            tot += coef * float(np.real(c.local_expectation(G, where, optimize="auto-hq", simplify_sequence="ADCRS")))
        return tot

    def grad(self, s):
        raise NotImplementedError("quimb gradient needs an autodiff backend; not timed")


def make(tool, w):
    if CONES:
        return Parts(tool, cone_parts(w, CONES))
    if tool == "pauli_cones": return PauliCones(w)
    if tool == "pauli_cones_graph": return PauliCones(w, "graph")
    a = {"qulacs": Qulacs, "aer": Aer, "lightning": Lightning, "pauli_prop": PauliProp, "quimb": Quimb,
         "spd": Spd}[tool](w)
    if tool == "pauli_prop":
        _fit_pauli_prop(a)
    return a


class Parts:
    """One adapter per light-cone part; values and gradients add up."""
    def __init__(self, tool, parts):
        global CONES
        mode, CONES = CONES, ""
        try:
            self.a = [make(tool, p) for p in parts]
        finally:
            CONES = mode
        self.build = sum(a.build for a in self.a)
        self.prepare = sum(a.prepare for a in self.a)

    def value(self, s): return sum(a.value(s) for a in self.a)

    def grad(self, s):
        v, g = 0.0, [0.0] * len(s)
        for a in self.a:
            pv, pg = a.grad(s)
            v += pv
            g = [x + y for x, y in zip(g, pg)]
        return v, g


def worker(wname, tool):
    t0 = time.perf_counter()
    w = WORKLOADS[wname]
    rec = dict(workload=wname, tool=tool)
    vecs = vectors(w)
    try:
        a = make(tool, w)
        rec.update(build_s=a.build, prepare_s=a.prepare)
        t = time.perf_counter(); rec["value"] = a.value(vecs[0]); rec["first_value_s"] = time.perf_counter() - t
        rec["since_import_to_first_value_s"] = time.perf_counter() - t0
        ts = []
        for v in vecs[1:1 + REPS]:
            t = time.perf_counter(); a.value(v); ts.append(time.perf_counter() - t)
        rec["warm_value_s"] = statistics.median(ts); rec["warm_value_samples"] = ts
        try:
            t = time.perf_counter(); gv, g = a.grad(vecs[0]); rec["first_grad_s"] = time.perf_counter() - t
            rec["gradient"] = g
            ts = []
            for v in vecs[1 + REPS:]:
                t = time.perf_counter(); a.grad(v); ts.append(time.perf_counter() - t)
            rec["warm_grad_s"] = statistics.median(ts); rec["warm_grad_samples"] = ts
        except NotImplementedError as e:
            rec["grad_unsupported"] = str(e)
        if hasattr(a, "plan"):
            rec["diagnostics"] = {k: v for k, v in a.plan.diagnostics.items() if isinstance(v, (int, float, str))}
        # Setting the same circuit up again times one more circuit's setup
        # without once-per-process costs such as imports and library loading.
        t = time.perf_counter(); make(tool, w); rec["rebuild_s"] = time.perf_counter() - t
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print("RESULT " + json.dumps(rec), flush=True)


def vectors(w):
    """The timed shift vectors: the first is all zero for workloads checked
    against exact values at their base angles."""
    rng = np.random.default_rng(1)
    vecs = [list(rng.normal(scale=w.get("shift_scale", 0.3), size=w["nparams"])) for _ in range(1 + 2 * REPS)]
    if w.get("zero_first"):
        vecs[0] = [0.] * w["nparams"]
    return vecs


def _fit_pauli_prop(a):
    """Map each source rotation to its transformed rotation index and sign by probing."""
    # pauli-prop moves Cliffords to the end; rotations remain in order but noise
    # generators are interleaved. Identify gate rotations as the ones whose
    # thetas equal +-(base angle) in order.
    a.map, a.sign = [], []
    j = 0
    for theta, _ in a.rot:
        while not math.isclose(abs(a.base[j]), abs(theta), rel_tol=0, abs_tol=1e-12):
            j += 1
        a.map.append(j); a.sign.append(1.0 if math.isclose(a.base[j], theta, abs_tol=1e-12) else -1.0)
        j += 1


def run(out, names, timeout):
    import platform
    py = sys.executable
    for name in names:
        w = WORKLOADS[name]
        for tool in TOOLS:
            if not eligible(w, tool):
                continue
            env = dict(os.environ, **THREADS)
            t = time.perf_counter()
            try:
                p = subprocess.run([py, "-I", __file__, "worker", name, tool], capture_output=True, text=True,
                                   timeout=timeout, env=env)
                wall = time.perf_counter() - t
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                rec = json.loads(line[-1][7:]) if line else dict(workload=name, tool=tool,
                                                                error=f"no result: {p.stderr[-300:]}")
                rec["cold_process_wall_s"] = wall
            except subprocess.TimeoutExpired:
                rec = dict(workload=name, tool=tool, error=f"timeout>{timeout}s")
            rec.update(cones=CONES or "none", pp_atol=os.environ.get("BENCH_PP_ATOL", "0"), threads=NT, reps=REPS, python=py, n=w["n"], layers=w["layers"], noisy=w["noisy"], observable=w["observable"],
                       obs_terms=len(w["obs"]), nparams=w["nparams"])
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            short = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                     if k in ("workload", "tool", "value", "prepare_s", "first_value_s", "warm_value_s",
                              "warm_grad_s", "cold_process_wall_s", "peak_rss_mb", "error")}
            print(short, flush=True)



def export(path, names):
    """Workloads + the exact timed vectors as JSON for the Julia adapters.
    Each workload also carries its light-cone parts for pp_tuned.jl."""
    def js(p):
        return dict(n=p["n"], ops=[[o[0], list(o[1]) if isinstance(o[1], tuple) else o[1], *o[2:]] for o in p["ops"]],
                    obs=[[c, {str(q): a for q, a in pp.items()}] for c, pp in p["obs"]])
    out = {}
    for name in names:
        w = WORKLOADS[name]
        out[name] = dict(js(w), nparams=w["nparams"], noisy=w["noisy"], layers=w["layers"],
                         observable=w["observable"], vecs=vectors(w),
                         parts={mode: [js(p) for p in cone_parts(w, mode)] for mode in ("cone", "split")})
    json.dump(out, open(path, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3:] or list(WORKLOADS))
    elif sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        names = sys.argv[3:] or list(WORKLOADS)
        run(sys.argv[2], names, timeout=float(os.environ.get("BENCH_TIMEOUT", 180)))

