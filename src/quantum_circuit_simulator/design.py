"""Design new chips and compare them on benchmark circuits.

    from quantum_circuit_simulator import design, devices
    module = devices.get("heavy-hex", rows=3, row_length=7)
    chip = design.modular(module, modules=4, links="ring", link_error=2e-2)       # four linked modules
    chip = design.remove_qubits(chip, [5, 17])                                    # defective qubits
    chip = design.with_physics(chip, t1=300e-6, t2=200e-6, time_1q=30e-9, time_2q=200e-9,
                               control_error_2q=2e-3)
    design.compare([chip, devices.get("heavy-hex-127")], family="qaoa-ring", sizes=[8, 12])

Every function returns a new Device (the chip description used everywhere
else in the package), so a design can be saved with to_json(), loaded with
devices.get("chip.json"), compiled onto and swept like a catalogue chip.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .devices import Device
from .topology import Topology


def from_edges(name, n_qubits, edges, *, family="custom", native_gates=None, **noise):
    """A chip from a coupler list. noise: error_1q, error_2q, readout, t1, t2, time_1q, time_2q."""
    return Device(name=name, family=family, topology=Topology(n_qubits, tuple(tuple(e) for e in edges)),
                  native_gates=native_gates or {"one_qubit": ["rz", "sx", "x"], "two_qubit": ["cz"]},
                  description="Custom chip.", source="Designed with quantum_circuit_simulator.design.",
                  parameters={"designed": True}, **noise)


def add_couplers(chip, couplers, *, error=None):
    """Add couplers; `error` sets their two-qubit infidelity (otherwise the chip's error_2q)."""
    new = {tuple(sorted(c)) for c in couplers} - set(chip.edges)
    errors = dict(chip.coupler_errors)
    if error is not None:
        errors.update({c: error for c in new})
    return _rebuild(chip, chip.n_qubits, list(chip.edges) + sorted(new), errors, f"{chip.name}+couplers")


def remove_couplers(chip, couplers):
    gone = {tuple(sorted(c)) for c in couplers}
    missing = gone - set(chip.edges)
    if missing:
        raise ValueError(f"not couplers of {chip.name}: {sorted(missing)}")
    errors = {k: v for k, v in chip.coupler_errors.items() if k not in gone}
    return _rebuild(chip, chip.n_qubits, [e for e in chip.edges if e not in gone], errors, f"{chip.name}-couplers")


def remove_qubits(chip, qubits):
    """Remove qubits (for example defective ones) and their couplers, renumbering the rest
    in order. The old-to-new numbering is kept in parameters["renumbering"]."""
    dead = set(qubits)
    if not dead <= set(range(chip.n_qubits)):
        raise ValueError(f"{chip.name} has qubits 0..{chip.n_qubits - 1}")
    keep = [q for q in range(chip.n_qubits) if q not in dead]
    if not keep:
        raise ValueError("cannot remove every qubit")
    renumber = {old: new for new, old in enumerate(keep)}
    edges = [(renumber[a], renumber[b]) for a, b in chip.edges if a not in dead and b not in dead]
    errors = {(renumber[a], renumber[b]): e for (a, b), e in chip.coupler_errors.items()
              if a not in dead and b not in dead}
    out = _rebuild(chip, len(keep), edges, errors, f"{chip.name}-{len(dead)}q",
                   coordinates=tuple(chip.coordinates[q] for q in keep) if chip.coordinates else ())
    return replace(out, parameters={**out.parameters, "renumbering": {str(k): v for k, v in renumber.items()}},
                   qubit_overrides={renumber[q]: v for q, v in chip.qubit_overrides.items() if q in renumber})


def modular(module, modules, *, links="ring", links_per_pair=1, link_error=None, name=None):
    """Tile `modules` copies of a chip and join neighbouring copies with links.

    links: "line" joins copy i to i+1, "ring" also joins the last to the first,
    "all" joins every pair. Each join couples `links_per_pair` qubits of one
    copy to the same qubits of the other, chosen among the lowest-degree qubits
    (the module's edge). link_error is the links' two-qubit infidelity, usually
    worse than on-module couplers; None keeps the chip's error_2q.
    """
    if modules < 2:
        raise ValueError("a modular chip needs at least 2 modules")
    if links not in ("line", "ring", "all"):
        raise ValueError('links must be "line", "ring" or "all"')
    n = module.n_qubits
    degree = [len(a) for a in module._adjacency()]
    ports = sorted(range(n), key=lambda q: (degree[q], q))[:max(1, links_per_pair)]
    edges, errors = [], {}
    for m in range(modules):
        edges += [(a + m * n, b + m * n) for a, b in module.edges]
        errors.update({(a + m * n, b + m * n): e for (a, b), e in module.coupler_errors.items()})
    pairs = ([(m, m + 1) for m in range(modules - 1)] + ([(modules - 1, 0)] if links == "ring" and modules > 2 else [])
             if links != "all" else [(i, j) for i in range(modules) for j in range(i + 1, modules)])
    for i, j in pairs:
        for k, q in enumerate(ports):
            # Port k of module i links to port of the reversed order in module j, so links spread.
            a, b = q + i * n, ports[-1 - k] + j * n
            edges.append((min(a, b), max(a, b)))
            if link_error is not None:
                errors[(min(a, b), max(a, b))] = link_error
    family = module.family if module.family != "annealer" else "custom"
    # Modules side by side, two lattice units apart.
    base = module.positions()
    width = max(x for x, _ in base) - min(x for x, _ in base) + 2
    coordinates = tuple((x + m * width, y) for m in range(modules) for x, y in base)
    overrides = {q + m * n: v for m in range(modules) for q, v in module.qubit_overrides.items()}
    out = replace(module, name=name or f"{module.name}x{modules}-{links}",
                  topology=Topology(n * modules, tuple(edges)), coupler_errors=errors, family=family,
                  coordinates=coordinates, qubit_overrides=overrides,
                  description=f"{modules} copies of {module.name} joined in a {links} by {links_per_pair} "
                              "link(s) per pair of modules.",
                  parameters={"module": module.name, "modules": modules, "links": links,
                              "links_per_pair": links_per_pair, "link_error": link_error})
    return out


def coherence_limited_error(duration, t1, t2, qubits=1):
    """Average gate infidelity from T1/T2 decay alone during a gate of `duration`.

    One qubit: F = (3 + e^(-t/T1) + 2 e^(-t/T2)) / 6. Two qubits decaying
    independently: the process fidelity (1 + e^(-t/T1) + 2 e^(-t/T2)) / 4 is
    squared, and F = (4 F_process + 1) / 5. None for T1 means no amplitude decay.
    """
    a = math.exp(-duration / t1) if t1 else 1.0
    b = math.exp(-duration / t2) if t2 else 1.0
    process = (1 + a + 2 * b) / 4
    if qubits == 1:
        return 1 - (2 * process + 1) / 3
    if qubits == 2:
        return 1 - (4 * process ** 2 + 1) / 5
    raise ValueError("qubits must be 1 or 2")


def with_physics(chip, *, t1, t2, time_1q, time_2q, control_error_1q=0.0, control_error_2q=0.0, readout=None):
    """Set coherence times and gate durations, deriving gate errors from them.

    Each gate's infidelity is its coherence limit plus a control error
    (miscalibration, leakage, crosstalk) that coherence does not capture.
    Per-coupler errors are kept and raised by the same coherence term.
    """
    if t2 is not None and t1 is not None and t2 > 2 * t1:
        raise ValueError("T2 cannot exceed 2 T1")
    e1 = coherence_limited_error(time_1q, t1, t2, 1) + control_error_1q
    e2_coherence = coherence_limited_error(time_2q, t1, t2, 2)
    e2 = e2_coherence + control_error_2q
    couplers = {k: min(0.999, v + e2_coherence) for k, v in chip.coupler_errors.items()}
    return replace(chip, t1=t1, t2=t2, time_1q=time_1q, time_2q=time_2q, error_1q=min(0.999, e1),
                   error_2q=min(0.999, e2), coupler_errors=couplers,
                   readout=tuple(readout) if readout is not None else chip.readout,
                   source="Gate errors derived from coherence times and durations plus a control error "
                          "(quantum_circuit_simulator.design.with_physics).")


def compare(chips, *, family="qaoa-ring", sizes=(8, 12), noise_scale=1.0, router="greedy", circuit=None,
            observable=None, family_parameters=None):
    """Benchmark chip designs: run a circuit family (or one circuit) on each chip and
    summarise per chip. Returns a SweepResult whose rows are one per chip:
    mean |deviation| from the ideal value, mean SWAPs, two-qubit gates and
    duration over the sizes it fits, and how many sizes failed."""
    from . import devices as catalogue
    from .experiments import SweepResult, _device, sweep
    resolved = [_device(chip, catalogue) for chip in chips]
    names = [chip.name for chip in resolved]
    if len(set(names)) != len(names):
        raise ValueError("chips to compare need distinct names")
    rows = sweep(circuit or family, observable, devices=resolved, noise_scales=[noise_scale],
                 sizes=None if circuit else list(sizes), router=router,
                 family_parameters=family_parameters).rows
    summary = []
    for name in names:
        mine = [r for r in rows if r["device"] == name]
        good = [r for r in mine if "error" not in r]
        mean = lambda key: sum(r[key] for r in good) / len(good) if good else None  # noqa: E731
        summary.append({"device": name, "sizes_run": len(good),
                        "sizes_failed": len(mine) - len(good),
                        "mean_abs_deviation": (sum(abs(r["deviation"]) for r in good) / len(good)) if good else None,
                        "mean_swaps": mean("swaps"), "mean_two_qubit_gates": mean("two_qubit_gates"),
                        "mean_duration_s": mean("duration_s")})
    summary.sort(key=lambda r: (r["mean_abs_deviation"] is None, r["mean_abs_deviation"] or 0))
    result = SweepResult(summary)
    result.COLUMNS = ("device", "sizes_run", "sizes_failed", "mean_abs_deviation", "mean_swaps",
                      "mean_two_qubit_gates", "mean_duration_s")
    return result


def _rebuild(chip, n, edges, errors, name, coordinates=None):
    return replace(chip, name=name, topology=Topology(n, tuple(edges)), coupler_errors=errors,
                   coordinates=chip.coordinates if coordinates is None else coordinates,
                   qubit_overrides={} if n != chip.n_qubits else chip.qubit_overrides,
                   parameters={**chip.parameters, "designed": True})


# ---------------------------------------------------------------- qubit spread and crosstalk
def vary_qubits(chip, spread=0.2, *, seed=0):
    """A copy whose qubits differ, as on a real chip: each qubit's error_1q, readout,
    T1 and T2 are the chip's values times exp(N(0, spread)), with T2 kept below 2 T1."""
    import numpy as np
    rng = np.random.default_rng(seed)
    overrides = {}
    for q in range(chip.n_qubits):
        factor = lambda: float(np.exp(rng.normal(0, spread)))  # noqa: E731
        values = {}
        if chip.error_1q:
            values["error_1q"] = min(0.999, chip.error_1q * factor())
        if chip.readout:
            values["readout"] = tuple(min(0.499, p * factor()) for p in chip.readout)
        if chip.t1:
            values["t1"] = chip.t1 * factor()
        if chip.t2:
            t2 = chip.t2 * factor()
            values["t2"] = min(t2, 2 * values.get("t1", chip.t1 or math.inf))
        overrides[q] = values
    return replace(chip, qubit_overrides=overrides, parameters={**chip.parameters, "qubit_spread": spread,
                                                               "qubit_spread_seed": seed})


def with_crosstalk(chip, *, spectator_error=0.0, zz_hz=0.0):
    """A copy with crosstalk: a depolarizing `spectator_error` on each neighbour of a
    two-qubit gate, and an always-on ZZ rate `zz_hz` between every coupled pair of
    active qubits, idle or not (a conditional phase of 2 pi zz_hz t). Coherent ZZ
    makes exact propagation expensive; experiments.solve truncates when it must."""
    return replace(chip, crosstalk={"spectator_error": spectator_error, "zz_hz": zz_hz})


# ---------------------------------------------------------------- growing chips
def expand(chip, n_qubits):
    """The same architecture with at least `n_qubits` qubits.

    Catalogue chips are regenerated larger (heavy-hex keeps rows odd and rows
    about half the row length, grids keep their aspect ratio, ions and lines
    take the size directly, annealer graphs grow their size parameter); modular
    chips get more modules; other chips are tiled in a line. Chip-wide noise,
    timing and crosstalk carry over; per-qubit and per-coupler values, which
    name particular qubits, carry over only for added modules of a modular chip.
    """
    if n_qubits <= chip.n_qubits:
        return chip
    p = chip.parameters
    if "catalog" in p:
        grown = _grow_catalogue(p["catalog"], p["catalog_parameters"], n_qubits)
        out = replace(grown, error_1q=chip.error_1q, error_2q=chip.error_2q, readout=chip.readout, t1=chip.t1,
                      t2=chip.t2, time_1q=chip.time_1q, time_2q=chip.time_2q, crosstalk=dict(chip.crosstalk),
                      native_gates=chip.native_gates, source=chip.source)
        if "topological" in p:   # a tetron chip keeps its measurement model as it grows
            out = replace(out, parameters={**out.parameters, "topological": dict(p["topological"])})
    elif "module" in p and "modules" in p:
        size = chip.n_qubits // p["modules"]
        module = replace(chip, name=p["module"], topology=Topology(size, tuple(
            e for e in chip.edges if e[0] < size and e[1] < size)), coupler_errors={
            k: v for k, v in chip.coupler_errors.items() if k[0] < size and k[1] < size},
            qubit_overrides={q: v for q, v in chip.qubit_overrides.items() if q < size}, parameters={},
            coordinates=chip.coordinates[:size])
        out = modular(module, -(-n_qubits // size), links=p["links"], links_per_pair=p["links_per_pair"],
                      link_error=p["link_error"])
    else:
        out = modular(chip, -(-n_qubits // chip.n_qubits), links="line", name=f"{chip.name}-tiled")
    return replace(out, parameters={**out.parameters, "expanded_from": chip.name})


def series(chip, sizes):
    """Chips of the same architecture for each size in `sizes` (each at least that many qubits)."""
    return [expand(chip, n) for n in sizes]


def _grow_catalogue(name, parameters, n):
    from . import devices
    if name in ("heavy-hex-127", "heavy-hex"):
        k = 1
        while True:
            row_length = 4 * k + 3                 # heavy-hex needs row_length = 4k + 3
            rows = (row_length + 1) // 2           # about half the row length, as on the 127-qubit chip
            rows = max(3, rows if rows % 2 else rows - 1)  # and an odd number of rows
            chip = devices.get("heavy-hex", rows=rows, row_length=row_length)
            if chip.n_qubits >= n:
                return chip
            k += 1
    if name in ("square-grid", "neutral-atom-grid"):
        r, c = parameters["rows"], parameters["columns"]
        f = math.sqrt(n / (r * c))
        rows, cols = max(r, math.ceil(r * f)), max(c, math.ceil(c * f))
        while rows * cols < n:
            cols += 1
        extra = {"radius": parameters["radius"]} if name == "neutral-atom-grid" else {}
        return devices.get(name, rows=rows, columns=cols, **extra)
    if name == "line":
        return devices.get("line", n=n)
    if name in ("tetron-array", "microsoft_majorana_2"):
        r, c = parameters.get("rows") or 2, parameters.get("columns") or 2
        f = math.sqrt(n / (r * c))
        rows, cols = max(r, math.ceil(r * f)), max(c, math.ceil(c * f))
        while rows * cols < n:
            cols += 1
        return devices.get("tetron-array", rows=rows, columns=cols)
    if name == "all-to-all-ions":
        return devices.get("all-to-all-ions", n=n)
    if name == "chimera":
        t = parameters["t"]
        m = max(parameters["m"], math.ceil(math.sqrt(n / (2 * t))))
        return devices.get("chimera", m=m, t=t)
    if name == "pegasus":
        m = parameters["m"]
        while devices.get("pegasus", m=m).n_qubits < n:
            m += 1
        return devices.get("pegasus", m=m)
    if name == "zephyr":
        m, t = parameters["m"], parameters["t"]
        while 4 * t * m * (2 * m + 1) < n:
            m += 1
        return devices.get("zephyr", m=m, t=t)
    raise ValueError(f"cannot grow {name}")
