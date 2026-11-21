"""Choose a solver automatically, and sweep circuits over chips, noise and size.

    from quantum_circuit_simulator.experiments import solve, sweep
    result = solve(circuit, terms)            # value, method, exact or not, and why
    table = sweep("qaoa-ring", sizes=[8, 16, 32], devices=["heavy-hex-127", "all-to-all-ions:n=32"],
                  noise_scales=[0, 0.5, 1, 2])
    print(table.markdown())

solve() tries the fastest exact method first and degrades visibly: exact
sparse propagation; then exact local density matrices when every cone has at
most 6 wires; then truncation at 1e-4 and 1e-5, reporting the change between
the two as the convergence check alongside the rigorous bound. Every result
says which method produced it and whether it is exact.

sweep() compiles the circuit onto each chip at each noise scale (and each size,
for a circuit family or a function of n) and solves it, with the ideal
noise-free value beside every noisy one.
"""
from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field

from .api import ConeOverflow, Truncation


def solve(circuit, terms, shifts=None, *, gradients=False, cross_check=False, tolerances=(1e-4, 1e-5),
          exact_budget=250_000):
    """Evaluate observable terms on a `Circuit` with the best available method.

    Returns a dict: value, method, exact, `reasons` recording each method
    tried and why it was set aside, and with gradients=True the gradient in
    the circuit's parameter order. Methods in order: exact sparse propagation
    (the graph backend's single reverse pass when there are more than 16
    parameters and gradients are wanted); the state vector (noise-free, up to
    24 qubits) or density matrix (noisy, up to 10 qubits); exact local density
    matrices when every cone has at most 6 wires; then truncation at each
    tolerance, reporting the rigorous error bound and the change between the
    last two tolerances as the convergence check. The exact attempt stops at
    `exact_budget` terms (or the circuit's own cap, if lower).

    cross_check=True evaluates the value a second, independent way where one
    fits (state simulation against Pauli propagation) and reports
    `cross_check` with the other method, its value and the difference.
    """
    reasons = []
    out = None
    many = gradients and circuit.n_parameters > 16
    for backend in (("graph", "native") if many else ("native",)):
        try:
            plan = _capped(circuit, exact_budget).prepare(terms, backend=backend)
            result = plan.evaluate(shifts, gradients=gradients)
            out = _row(result, plan.backend, True, reasons, gradients)
            break
        except ConeOverflow as exc:
            reasons.append(f"{backend} exact within {min(exact_budget, circuit.max_terms)} terms: {exc}")
    if out is None:
        out = _state_method(circuit, terms, shifts, gradients, reasons)
    if out is None:
        try:
            result = circuit.prepare(terms, backend="dense").evaluate(shifts, gradients=gradients)
            out = _row(result, "dense", True, reasons, gradients)
        except ConeOverflow as exc:
            reasons.append(f"dense exact: {exc}")
    if out is None:
        out = _truncated(circuit, terms, shifts, gradients, tolerances, reasons)
    if cross_check:
        out["cross_check"] = _cross_check(circuit, terms, shifts, out)
    return out


def _row(result, method, exact, reasons, gradients):
    row = {"value": result.value, "method": method, "exact": exact, "reasons": reasons}
    if gradients:
        row["gradient"] = list(result.gradient)
    return row


# State simulation limits inside the router: small enough to stay fast.
ROUTER_STATEVECTOR_QUBITS, ROUTER_DENSITY_QUBITS = 24, 10


def _state_method(circuit, terms, shifts, gradients, reasons):
    from .statevector import StateVectorPlan, is_noisy
    noisy = is_noisy(circuit)
    limit = ROUTER_DENSITY_QUBITS if noisy else ROUTER_STATEVECTOR_QUBITS
    kind = "density matrix" if noisy else "state vector"
    if circuit.n_qubits > limit:
        reasons.append(f"{kind}: {circuit.n_qubits} qubits exceed {limit}")
        return None
    if noisy and gradients and circuit.n_parameters:
        reasons.append("density matrix: no gradients for noisy circuits")
        return None
    value, gradient = StateVectorPlan(circuit, terms).evaluate(shifts, gradients=gradients)
    row = {"value": value, "method": kind.replace(" ", "_"), "exact": True, "reasons": reasons}
    if gradients:
        row["gradient"] = list(gradient) if len(gradient) else [0.0] * circuit.n_parameters
    return row


def _truncated(circuit, terms, shifts, gradients, tolerances, reasons):
    values = []
    for tolerance in tolerances:
        try:
            result = circuit.prepare(terms, truncation=Truncation(tolerance)).evaluate(shifts, gradients=gradients)
            values.append((tolerance, result))
        except ConeOverflow as exc:
            reasons.append(f"truncated at {tolerance:g}: {exc}")
            break
    if not values:
        raise ConeOverflow("no method fits: " + "; ".join(reasons))
    tolerance, result = values[-1]
    out = _row(result, f"native truncated at {tolerance:g}", False, reasons, gradients)
    out["error_bound"] = result.error_bound
    if gradients:
        out["gradient_error_bound"] = list(result.gradient_error_bound)
    if len(values) > 1:
        out["convergence_change"] = abs(values[-1][1].value - values[-2][1].value)
    else:
        out["convergence_change"] = None
        reasons.append("only one tolerance fitted the term cap; convergence is unchecked")
    return out


def _cross_check(circuit, terms, shifts, first):
    """A second value from an independent method, or why none fits."""
    from .statevector import StateVectorPlan, is_noisy
    if first["method"] in ("state_vector", "density_matrix"):
        try:
            other = circuit.prepare(terms, backend="dense").evaluate(shifts, gradients=False).value
            method = "dense"
        except ConeOverflow:
            return {"method": None, "reason": "no second exact method fits"}
    else:
        limit = 12 if is_noisy(circuit) else 26
        if circuit.n_qubits > limit:
            return {"method": None, "reason": f"{circuit.n_qubits} qubits exceed the state simulators' {limit}"}
        other = StateVectorPlan(circuit, terms).evaluate(shifts, gradients=False)[0]
        method = "density_matrix" if is_noisy(circuit) else "state_vector"
    difference = abs(other - first["value"])
    allowed = (first.get("error_bound") or 0.0) + 1e-9 * max(1.0, abs(other))
    return {"method": method, "value": other, "difference": difference, "agrees": difference <= allowed}


def _capped(circuit, cap):
    """The circuit with its term cap lowered to `cap` (a copy; the original is unchanged)."""
    if cap >= circuit.max_terms:
        return circuit
    import copy
    lowered = copy.copy(circuit)
    lowered._impl = copy.copy(circuit._impl)
    lowered._impl.max_terms = cap
    return lowered


@dataclass
class SweepResult:
    """Rows of a sweep, with table, CSV and JSON views."""
    rows: list = field(default_factory=list)

    COLUMNS = ("size", "device", "noise_scale", "value", "ideal", "deviation", "method", "exact",
               "swaps", "two_qubit_gates", "depth", "duration_s", "seconds", "error")

    def to_json(self):
        return json.dumps(self.rows, indent=1)

    def csv(self):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.rows)
        return buffer.getvalue()

    def markdown(self):
        def cell(value):
            if isinstance(value, float):
                return f"{value:.6g}"
            return "" if value is None else str(value)
        lines = ["| " + " | ".join(self.COLUMNS) + " |", "|" + "---|" * len(self.COLUMNS)]
        lines += ["| " + " | ".join(cell(row.get(c)) for c in self.COLUMNS) + " |" for row in self.rows]
        return "\n".join(lines)


def sweep(circuit, observable=None, *, devices=None, noise_scales=(1.0,), sizes=None, values=None,
          router="greedy", layout="greedy", idle_noise=True, family_parameters=None, progress=None,
          checkpoint=None):
    """Compile and solve `circuit` on each device at each noise scale (and size).

    circuit is OpenQASM text, a Qiskit circuit, a circuit-family name from
    quantum_circuit_simulator.circuits (with `sizes`), or a function n -> circuit.
    observable is a string, or a function n -> string; a family supplies its
    own when it is None. devices are Device objects, catalogue names, or
    "name:k=v,k=v" strings; None evaluates only the ideal circuit. A device
    written "fit:name..." (or "fit:" followed by a .json path) grows to fit
    each size with the same architecture (quantum_circuit_simulator.design.expand).

    progress is called after each point with its row (contracts.Progress
    prints JSON lines); checkpoint is a path to a JSON-lines file of completed
    points, so a rerun of the same sweep resumes where it stopped.
    """
    from . import circuits, devices as catalogue
    from .compiler import compile_circuit
    from .interop import convert, parse_observable
    from .design import expand
    chips = [(True, _device(d[4:], catalogue)) if isinstance(d, str) and d.startswith("fit:")
             else (False, _device(d, catalogue)) for d in (devices or [])]
    if checkpoint is not None:
        from .contracts import Checkpoint
        request = {"circuit": circuit if isinstance(circuit, str) else repr(circuit), "observable": str(observable),
                   "devices": [c.to_dict() for _, c in chips], "fit": [f for f, _ in chips],
                   "noise_scales": list(noise_scales), "sizes": sizes, "values": values, "router": router,
                   "layout": layout, "idle_noise": idle_noise, "family_parameters": family_parameters}
        checkpoint = Checkpoint(checkpoint, request)
    rows = []

    def done(key, row):
        rows.append(row)
        if checkpoint is not None and checkpoint.get(key) is None:
            checkpoint.put(key, row)
        if progress:
            progress(**{k: row.get(k) for k in ("size", "device", "noise_scale", "value", "error") if k in row})
    for n in (sizes or [None]):
        source, obs = _instance(circuit, observable, n, circuits, family_parameters or {})
        key = f"{n}|ideal|0.0"
        if checkpoint is not None and checkpoint.get(key):
            ideal = checkpoint.get(key)
            done(key, ideal)
        else:
            ideal_conv = convert(source)
            shifts = ideal_conv.shifts(values) if ideal_conv.parameters else None
            start = time.perf_counter()
            result = solve(ideal_conv.circuit, parse_observable(obs, ideal_conv.circuit.n_qubits), shifts)
            ideal = {"size": n, "device": "ideal", "noise_scale": 0.0, "value": result["value"],
                     "ideal": result["value"], "deviation": 0.0, "method": result["method"], "exact": result["exact"],
                     "seconds": time.perf_counter() - start}
            done(key, ideal)
        width = None
        for fit, base in chips:
            if fit and width is None:
                width = convert(source).circuit.n_qubits
            chip = expand(base, width) if fit else base
            for scale in noise_scales:
                key = f"{n}|{chip.name}|{scale}"
                if checkpoint is not None and checkpoint.get(key):
                    done(key, checkpoint.get(key))
                    continue
                row = {"size": n, "device": chip.name, "noise_scale": scale, "ideal": ideal["value"]}
                start = time.perf_counter()
                try:
                    compiled = compile_circuit(source, chip, router=router, layout=layout, noise=True,
                                               idle_noise=idle_noise, noise_scale=scale)
                    result = solve(compiled.circuit, compiled.observable(obs),
                                   compiled.shifts(values) if compiled.parameters else None)
                    row.update(value=result["value"], deviation=result["value"] - ideal["value"],
                               method=result["method"], exact=result["exact"],
                               **{k: compiled.report[k] for k in ("swaps", "two_qubit_gates", "depth", "duration_s")})
                    if not result["exact"]:
                        row.update(error_bound=result["error_bound"],
                                   convergence_change=result["convergence_change"])
                except (ConeOverflow, ValueError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = time.perf_counter() - start
                done(key, row)
    return SweepResult(rows)


def _instance(circuit, observable, n, circuits, parameters):
    if isinstance(circuit, str) and circuit in circuits.library_names():
        # A library benchmark: the bundled size with at least n qubits; observable mean Z by default.
        from .interop import _parse_qasm2
        source = circuits.load(circuit, n)
        width = _parse_qasm2(source)[0]
        default = " + ".join(f"{1 / width:.12g}*Z{q}" for q in range(width))
        return source, observable(n) if callable(observable) else (observable or default)
    if isinstance(circuit, str) and circuit in circuits.FAMILIES:
        if n is None:
            raise ValueError("a circuit family needs sizes")
        source, default = circuits.family(circuit, n=n, **parameters)
        obs = observable(n) if callable(observable) else (observable or default)
        return source, obs
    if callable(circuit):
        source = circuit(n)
    else:
        source = circuit
    if observable is None:
        raise ValueError("give an observable")
    return source, observable(n) if callable(observable) else observable


def _device(spec, catalogue):
    if isinstance(spec, str) and spec.endswith(".json"):
        return catalogue.get(spec)
    if isinstance(spec, dict):
        return catalogue.Device.from_dict(spec)
    if not isinstance(spec, str):
        return spec
    name, _, rest = spec.partition(":")
    parameters = {}
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        for kind in (int, float):
            try:
                parameters[key] = kind(value)
                break
            except ValueError:
                continue
        else:
            parameters[key] = value
    return catalogue.get(name, **parameters)
