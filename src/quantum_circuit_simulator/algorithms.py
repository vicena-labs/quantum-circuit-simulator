"""Algorithm development: objectives, optimizers, VQE, QAOA, zero-noise extrapolation and shot noise.

    from quantum_circuit_simulator import algorithms as al, hamiltonians as hm
    h = hm.heisenberg(hm.chain(8))
    result = al.vqe(h, al.hardware_efficient(8, layers=3), steps=200)
    result.value, result.reference, result.trace[-1]

An Objective prepares one plan for a parameterised circuit and a Hamiltonian
and returns <H> and its exact gradient at any parameter vector; optimizers
drive it and keep the full trace. Everything works on an ideal circuit or
compiled onto a device (routing, gate errors, idle relaxation, crosstalk and
readout). Results are plain data (dicts and lists) for an agent to report or
plot.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ._core._reference import DEFAULT_MAX_TERMS
from .api import ConeOverflow, Truncation
from .interop import Angle, ConversionError, Program, convert

# ---------------------------------------------------------------- ansatze


def hardware_efficient(n, layers=2, edges=None, rotations=("ry", "rz"), entangler="cz"):
    """A layered ansatz: a parameterised rotation of each kind on every qubit, then an entangling layer.

    edges defaults to a line; entangler is "cz" or "cx". Parameters are
    theta_0, theta_1, ... in gate order; a final rotation layer closes it.
    """
    edges = [(i, i + 1) for i in range(n - 1)] if edges is None else [tuple(e) for e in edges]
    instructions, names = [], []

    def rotate():
        for q in range(n):
            for kind in rotations:
                name = f"theta_{len(names)}"
                names.append(name)
                instructions.append((kind, [q], [Angle(name)]))
    for _ in range(layers):
        rotate()
        instructions += [(entangler, list(e), []) for e in edges]
    rotate()
    return Program(n, instructions, names)


# ---------------------------------------------------------------- objective
# Above this many qubits the state vector is too slow to be worth timing against Pauli propagation.
STATEVECTOR_OBJECTIVE_QUBITS = 20


def _timed(evaluate, n_parameters):
    """Seconds for one evaluation at a generic point: zero angles can make propagation trivially cheap."""
    point = np.random.default_rng(0).uniform(-1, 1, n_parameters) if n_parameters else None
    start = time.perf_counter()
    evaluate(point)
    return time.perf_counter() - start


class Objective:
    """<H> and its gradient for a parameterised circuit, from one plan reused at every call.

    circuit is a Program, OpenQASM, a Qiskit circuit or a Circuit; parameter
    vectors follow its parameter order (Program.parameters, Qiskit's
    circuit.parameters). With a device, the circuit is compiled onto it first
    (noise_scale scales every error). The plan is the first that fits: exact
    sparse propagation (the graph backend's one reverse pass beyond 16
    parameters), the state vector, then truncation at `tolerance`, in which
    case each call also reports the rigorous error bound.
    """

    def __init__(self, circuit, hamiltonian, *, device=None, noise_scale=1.0, layout="greedy", router="greedy",
                 tolerance=1e-5, max_terms=DEFAULT_MAX_TERMS):
        from .hamiltonians import load
        self.hamiltonian = load(hamiltonian)
        self.order = list(getattr(circuit, "parameters", None) or [])
        if device is not None:
            from .compiler import compile_circuit
            compiled = compile_circuit(circuit, device, layout=layout, router=router, noise_scale=noise_scale,
                                       max_terms=max_terms)
            self.circuit, internal = compiled.circuit, compiled.parameters
            self.terms = compiled.observable(self.hamiltonian.terms)
            self.report = compiled.report
        else:
            converted = convert(circuit, max_terms=max_terms)
            self.circuit, internal = converted.circuit, converted.parameters
            self.terms = self.hamiltonian.terms
            self.report = None
        names = [getattr(p, "name", p) for p in internal]
        self.order = self.order or names
        # Position of each internal parameter in the caller's order.
        self._index = [self.order.index(name) for name in names]
        self.method, self._plan, self.error_bound = self._plan_for(tolerance)
        self.calls = 0

    @property
    def n_parameters(self):
        return len(self.order)

    def _plan_for(self, tolerance):
        """The fastest exact plan, else a truncated one. Both exact methods are timed once when both fit."""
        from .statevector import StateVectorPlan, is_noisy
        many = self.circuit.n_parameters > 16
        candidates = []
        for backend in (("graph", "native") if many else ("native",)):
            try:
                plan = self.circuit.prepare(self.terms, backend=backend)
                candidates.append((_timed(plan.evaluate, self.circuit.n_parameters), plan.backend, plan))
                break
            except ConeOverflow:
                continue
        if not is_noisy(self.circuit) and self.circuit.n_qubits <= STATEVECTOR_OBJECTIVE_QUBITS:
            plan = StateVectorPlan(self.circuit, self.terms)
            candidates.append((_timed(plan.evaluate, self.circuit.n_parameters), "state_vector", plan))
        if candidates:
            _, method, plan = min(candidates, key=lambda c: c[0])
            return method, plan, False
        plan = self.circuit.prepare(self.terms, truncation=Truncation(tolerance),
                                    backend="graph" if many else "native")
        return f"{plan.backend} truncated at {tolerance:g}", plan, True

    def _shifts(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_parameters,):
            raise ValueError(f"expected {self.n_parameters} parameter values")
        return x[self._index] if self._index else None

    def __call__(self, x, gradients=True):
        """(value, gradient in the caller's order, error bound or None)."""
        self.calls += 1
        shifts = self._shifts(x)
        if self.method == "state_vector":
            value, internal = self._plan.evaluate(shifts, gradients=gradients)
            bound = None
        else:
            result = self._plan.evaluate(shifts, gradients=gradients)
            value, internal, bound = result.value, np.asarray(result.gradient), result.error_bound
        gradient = np.zeros(self.n_parameters)
        if gradients and len(internal):
            np.add.at(gradient, self._index, internal)
        return float(value), gradient, bound

    def value(self, x):
        return self(x, gradients=False)[0]


# ---------------------------------------------------------------- optimizers
@dataclass
class OptimizeResult:
    """The best point found and the full trace, one row per iteration."""
    value: float
    parameters: list
    method: str
    iterations: int
    evaluations: int
    converged: bool
    trace: list = field(default_factory=list)
    solver: str = ""
    reference: float | None = None   # exact optimum when known (ground energy, best cut)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def minimize(objective, x0, *, method="adam", steps=200, learning_rate=0.05, tolerance=1e-8, seed=0,
             keep_parameters=True, callback=None):
    """Minimise an Objective (or any f(x) -> (value, gradient, bound)) from x0.

    method: "adam" (gradients), "gd" (plain gradient descent), "spsa"
    (two values per step, no gradients: what hardware runs can afford) or
    "lbfgs" (SciPy). Stops after `steps` iterations or when the value changes
    by less than `tolerance` over 10 iterations. Every iteration records
    value, gradient norm, error bound, seconds and (keep_parameters) the
    parameters.
    """
    x = np.array(x0, dtype=float)
    trace, start = [], time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluations = [0]

    def call(point, gradients=True):
        evaluations[0] += 1
        return objective(point, gradients) if gradients else objective(point, False)

    def record(i, value, gradient, bound, point):
        row = {"iteration": i, "value": value, "gradient_norm": float(np.linalg.norm(gradient)) if gradient is not None else None,
               "error_bound": bound, "seconds": time.perf_counter() - start}
        if keep_parameters:
            row["parameters"] = [float(v) for v in point]
        trace.append(row)
        if callback:
            callback(row)

    if method == "lbfgs":
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError as exc:
            raise ImportError('L-BFGS needs SciPy: pip install "quantum-circuit-simulator[algorithms]"') from exc
        state = {"i": 0}

        def fun(point):
            value, gradient, bound = call(point)
            record(state["i"], value, gradient, bound, point)
            state["i"] += 1
            return value, gradient
        res = scipy_minimize(fun, x, jac=True, method="L-BFGS-B", options={"maxiter": steps, "ftol": tolerance})
        best = min(trace, key=lambda r: r["value"])
        return OptimizeResult(best["value"], list(res.x), method, int(res.nit), evaluations[0], bool(res.success),
                              trace, getattr(objective, "method", ""))
    m, v = np.zeros_like(x), np.zeros_like(x)
    best_value, best_x, converged = math.inf, x.copy(), False
    for i in range(steps):
        if method == "spsa":
            # Standard gain sequences (Spall 1998): a_k = a / (k + 1 + A)^0.602, c_k = c / (k + 1)^0.101.
            a_k = learning_rate * 2 / (i + 1 + 0.1 * steps) ** 0.602
            c_k = 0.1 / (i + 1) ** 0.101
            delta = rng.choice([-1.0, 1.0], size=x.size)
            plus = call(x + c_k * delta, False)[0]
            minus = call(x - c_k * delta, False)[0]
            gradient = (plus - minus) / (2 * c_k) * delta
            value, bound = (plus + minus) / 2, None
            x = x - a_k * gradient
        else:
            value, gradient, bound = call(x)
            if method == "adam":
                m = 0.9 * m + 0.1 * gradient
                v = 0.999 * v + 0.001 * gradient ** 2
                step = learning_rate * (m / (1 - 0.9 ** (i + 1))) / (np.sqrt(v / (1 - 0.999 ** (i + 1))) + 1e-12)
            elif method == "gd":
                step = learning_rate * gradient
            else:
                raise ValueError("method must be adam, gd, spsa or lbfgs")
            record(i, value, gradient, bound, x)
            if value < best_value:
                best_value, best_x = value, x.copy()
            x = x - step
        if method == "spsa":
            record(i, value, gradient, bound, x)
            if value < best_value:
                best_value, best_x = value, x.copy()
        if i >= 10 and abs(trace[-1]["value"] - trace[-11]["value"]) < tolerance:
            converged = True
            break
    if method == "spsa":   # SPSA's recorded values are noisy averages; report the final point exactly
        best_value, best_x = call(x, False)[0], x.copy()
    return OptimizeResult(best_value, list(best_x), method, len(trace), evaluations[0], converged, trace,
                          getattr(objective, "method", ""))


def vqe(hamiltonian, ansatz=None, *, device=None, noise_scale=1.0, method="adam", steps=200, learning_rate=0.05,
        x0=None, seed=0, reference=True, **options):
    """Variational ground-state search: minimise <H> over an ansatz (hardware-efficient by default).

    reference=True adds the exact ground-state energy when the Hamiltonian is
    small enough (exact diagonalisation up to 22 qubits with SciPy).
    """
    from .hamiltonians import load
    h = load(hamiltonian)
    ansatz = ansatz or hardware_efficient(h.n_qubits, layers=2)
    objective = Objective(ansatz, h, device=device, noise_scale=noise_scale)
    rng = np.random.default_rng(seed)
    start = rng.normal(scale=0.1, size=objective.n_parameters) if x0 is None else np.asarray(x0, float)
    result = minimize(objective, start, method=method, steps=steps, learning_rate=learning_rate, seed=seed, **options)
    if reference and h.n_qubits <= 22:
        try:
            result.reference = h.ground_state_energy()
        except ImportError:
            pass
    result.extra = {"hamiltonian": h.summary(), "parameters": objective.order, "compiled": objective.report}
    return result


def qaoa(cost, depth=1, *, device=None, noise_scale=1.0, method="adam", steps=150, learning_rate=0.05,
         x0=None, seed=0, shots=1024, mixer="x", **options):
    """Optimise depth-p QAOA angles for a diagonal cost Hamiltonian, then sample bitstrings.

    Returns the optimised energy, the best sampled bitstring and its cost, and
    for up to 24 qubits the exact minimum cost and the approximation ratio
    (energy - max) / (min - max), the standard figure of merit.
    """
    from .hamiltonians import load, qaoa as qaoa_program
    h = load(cost)
    if any(set(p.values()) - {"Z"} for _, p in h.terms):
        raise ConversionError("QAOA needs a diagonal (Z-only) cost Hamiltonian")
    program = qaoa_program(h, depth, mixer)
    objective = Objective(program, h, device=device, noise_scale=noise_scale)
    start = np.tile([0.1, 0.1], depth) if x0 is None else np.asarray(x0, float)
    result = minimize(objective, start, method=method, steps=steps, learning_rate=learning_rate, seed=seed, **options)
    extra = {"hamiltonian": h.summary(), "parameters": program.parameters, "compiled": objective.report}
    if h.n_qubits <= 24:
        diagonal = cost_diagonal(h)
        low, high = float(diagonal.min()), float(diagonal.max())
        result.reference = low
        extra["approximation_ratio"] = (result.value - high) / (low - high) if low != high else 1.0
        if shots:
            from .statevector import sample
            target = program.bind(result.parameters)
            counts = sample(target, shots, seed=seed)["counts"] if device is None else None
            if counts:
                best = min(counts, key=lambda b: diagonal[int(b, 2)])
                extra.update(best_bitstring=best, best_sampled_cost=float(diagonal[int(best, 2)]),
                             probability_of_optimum=sum(c for b, c in counts.items()
                                                        if abs(diagonal[int(b, 2)] - low) < 1e-9) / shots)
    result.extra = extra
    return result


def cost_diagonal(hamiltonian):
    """Every basis state's energy for a Z-only Hamiltonian (up to 26 qubits), indexed by bitstring integer."""
    n = hamiltonian.n_qubits
    if n > 26:
        raise ValueError("the cost diagonal is limited to 26 qubits")
    index = np.arange(2 ** n, dtype=np.int64)
    out = np.zeros(2 ** n)
    for c, p in hamiltonian.terms:
        parity = np.zeros(2 ** n, dtype=np.int64)
        for q in p:
            parity ^= (index >> q) & 1
        out += c * (1 - 2 * parity)
    return out


# ---------------------------------------------------------------- zero-noise extrapolation
def zne(circuit, observable, device, *, scales=(1.0, 1.5, 2.0, 3.0), fit="richardson", values=None,
        layout="greedy", router="greedy", ideal=True):
    """Zero-noise extrapolation: solve at amplified noise and extrapolate to zero.

    Noise is amplified by scaling every device error (gate, idle, readout,
    crosstalk) by each factor, the idealised form of pulse stretching. fit:
    "linear" (least squares, degree 1), "quadratic", "richardson" (the
    polynomial through all points) or "exponential" (a + b e^{-c s}). Returns
    the estimate, a leave-one-out spread as its fit uncertainty, every
    point, and the noise-free value for comparison when ideal=True.
    """
    from .compiler import compile_circuit
    from .experiments import solve
    points = []
    for s in scales:
        compiled = compile_circuit(circuit, device, layout=layout, router=router, noise_scale=s)
        shifts = compiled.shifts(values) if compiled.parameters else None
        result = solve(compiled.circuit, compiled.observable(observable), shifts)
        points.append({"noise_scale": s, "value": result["value"], "method": result["method"],
                       "exact": result["exact"]})
    xs = np.array([p["noise_scale"] for p in points])
    ys = np.array([p["value"] for p in points])
    estimate = _extrapolate(xs, ys, fit)
    spread = [abs(_extrapolate(np.delete(xs, i), np.delete(ys, i), fit) - estimate)
              for i in range(len(xs))] if len(xs) > _degree(fit, len(xs)) + 1 else []
    out = {"estimate": estimate, "fit": fit, "fit_uncertainty": max(spread) if spread else None,
           "points": points, "unmitigated": float(ys[np.argmin(np.abs(xs - 1))])}
    if ideal:
        compiled = compile_circuit(circuit, device, layout=layout, router=router, noise_scale=0.0)
        shifts = compiled.shifts(values) if compiled.parameters else None
        exact = solve(compiled.circuit, compiled.observable(observable), shifts)["value"]
        out.update(ideal=exact, error=estimate - exact, unmitigated_error=out["unmitigated"] - exact)
    return out


def _degree(fit, n):
    return {"linear": 1, "quadratic": 2, "richardson": n - 1, "exponential": 2}[fit]


def _extrapolate(xs, ys, fit):
    if fit not in ("linear", "quadratic", "richardson", "exponential"):
        raise ValueError("fit must be linear, quadratic, richardson or exponential")
    if fit == "exponential":
        try:
            from scipy.optimize import curve_fit
        except ImportError as exc:
            raise ImportError('the exponential fit needs SciPy: pip install "quantum-circuit-simulator[algorithms]"') from exc
        try:
            (a, b, c), _ = curve_fit(lambda s, a, b, c: a + b * np.exp(-c * s), xs, ys,
                                     p0=(0.0, ys[0], 0.5), maxfev=20000)
            return float(a + b)
        except RuntimeError:
            fit = "richardson"   # no exponential fits these points; fall back to the polynomial
    degree = _degree(fit, len(xs))
    if len(xs) <= degree:
        raise ValueError(f"{fit} needs more than {degree} noise scales")
    return float(np.polyval(np.polyfit(xs, ys, degree), 0.0))


# ---------------------------------------------------------------- shot noise
def shot_noise(circuit, observable, shots, *, values=None, allocation="weighted", seed=None):
    """The statistical error a hardware run of `shots` total shots would have on <observable>.

    Terms are measured in qubit-wise commuting groups, one circuit variant
    each. A group's single-shot variance, sum over terms P, Q of
    c_P c_Q (<PQ> - <P><Q>), is computed exactly from Pauli expectations (PQ is
    again a Pauli word within a group), so the estimate scales to any size
    Pauli propagation reaches. Shots go to groups in proportion to sqrt of
    variance ("weighted", optimal) or equally ("equal"). Returns the exact
    value, its standard error, per-group details and, with a seed, one
    simulated measured value drawn from the matching normal distribution.
    """
    from .experiments import solve
    from .hamiltonians import load
    h = load(observable)
    converted = convert(circuit, values=values)
    shifts = converted.shifts(values) if converted.parameters else None
    c = converted.circuit

    def expect(terms):
        return solve(c, terms, shifts)["value"] if terms else 0.0
    groups, variances = [], []
    for members in h.groups():
        mean = {tuple(sorted(p.items())): expect([(1.0, p)]) for _, p in members}
        products = {}
        for c1, p1 in members:
            for c2, p2 in members:
                word = _product(p1, p2)
                products[tuple(sorted(word.items()))] = products.get(tuple(sorted(word.items())), 0.0) + c1 * c2
        second = sum(w * expect([(1.0, dict(k))]) if k else w for k, w in products.items())
        first = sum(cf * mean[tuple(sorted(p.items()))] for cf, p in members)
        variance = max(0.0, second - first ** 2)
        groups.append({"terms": len(members), "variance": variance})
        variances.append(variance)
    if not groups:
        return {"value": h.offset, "standard_error": 0.0, "groups": [], "shots": shots}
    weights = np.sqrt(variances) if allocation == "weighted" and sum(variances) > 0 else np.ones(len(groups))
    allotted = np.maximum(1, np.floor(shots * weights / weights.sum())).astype(int)
    for g, n in zip(groups, allotted):
        g["shots"] = int(n)
    error = math.sqrt(sum(v / n for v, n in zip(variances, allotted)))
    value = expect(h.terms)
    out = {"value": value, "standard_error": error, "shots": int(allotted.sum()), "circuit_variants": len(groups),
           "allocation": allocation, "groups": groups}
    if seed is not None:
        out["simulated_measurement"] = float(np.random.default_rng(seed).normal(value, error))
    return out


def _product(p1, p2):
    """The Pauli word of P1 P2 for qubit-wise commuting words (same axis or identity on each wire)."""
    out = dict(p1)
    for q, a in p2.items():
        if out.get(q) == a:
            del out[q]
        else:
            out[q] = a
    return out
