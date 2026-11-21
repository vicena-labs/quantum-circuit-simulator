"""Output contracts: versioned result schemas, provenance, progress events and checkpoints.

Every CLI result is one JSON document tagged `schema: quantum-circuit-simulator-<kind>/<n>`,
with a `provenance` record that says what produced it: package and library
versions, the platform, a SHA-256 of the canonical request, the seed and the
time. A field may be added to a schema without a version change; removing or
redefining one raises the version. Long tasks report progress as JSON lines
on standard error, one event per completed unit, and sweeps can checkpoint
completed points to a JSON-lines file and resume from it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

# Result kinds and what each carries beyond `ok`, `schema` and `provenance`.
OUTPUTS = {
    "result": "results (value, gradient, error bounds) per parameter vector, backend, diagnostics, numerics",
    "plan": "cone sizes, rigorous term bound and recommended commands",
    "compile": "swaps, depth, duration, layouts, noise_budget (per qubit, per coupler, totals) and, with "
               "--schedule, every gate's start time and duration",
    "sweep": "rows: size, device, noise_scale, value, ideal, deviation, method, exact, swaps, depth, duration_s",
    "compare": "rows ranking chips: mean deviation, SWAPs, gates and duration",
    "device": "a chip: coupling graph, coordinates, native gates, noise, coherence and timing",
    "hamiltonian": "terms, summary (qubits, weights, norm, commuting groups) and, when small, the exact ground energy",
    "sample": "counts by bitstring (wire 0 rightmost), shots, method, whether the distribution is exact",
    "statevector": "amplitudes as [real, imaginary] pairs, wire 0 the least significant bit",
    "optimize": "value, parameters, reference optimum, converged, and the trace of every iteration",
    "zne": "estimate, fit, fit_uncertainty, points per noise scale, unmitigated and ideal values",
    "shot-noise": "value, standard_error, shots per commuting group and their variances",
    "anneal": "best energy and sample, energy statistics, ground energy and its probability when small, "
              "embedding (qubits used, chain lengths) and chain-break fraction on a chip",
    "circuits": "circuit families and the benchmark library (name, suite, qubits, gates, source), or one circuit's OpenQASM",
    "cad": "path, format, qubits, couplers and size in micrometres of the written layout",
}


def schema_name(kind, version=1):
    if kind not in OUTPUTS:
        raise ValueError(f"unknown result kind {kind!r}")
    return f"quantum-circuit-simulator-{kind}/{version}"


def provenance(request=None, *, seed=None, solver=None):
    """What produced a result: versions, platform, request hash, seed, solver and UTC time."""
    import numpy
    record = {"package": "quantum-circuit-simulator", "version": __version__, "python": platform.python_version(),
              "numpy": numpy.__version__, "platform": platform.platform(),
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if request is not None:
        record["request_sha256"] = digest(request)
    if seed is not None:
        record["seed"] = seed
    if solver is not None:
        record["solver"] = solver
    return record


def digest(value):
    """SHA-256 of a value's canonical JSON (sorted keys, no spaces), or of text and bytes as given."""
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def envelope(kind, body, request=None, **facts):
    """A successful result document: ok, schema, schema_version, the body and provenance."""
    return {"ok": True, "schema": schema_name(kind), "schema_version": 1, **body,
            "provenance": provenance(request, **facts)}


class Progress:
    """Progress events as JSON lines: {"event": "progress", "task", "done", "total", "elapsed_s", ...}.

    Writes to `stream` (standard error by default) when enabled; calling a
    disabled Progress does nothing, so library code can always report.
    """

    def __init__(self, task, total=None, *, enabled=True, stream=None):
        self.task, self.total, self.enabled = task, total, enabled
        self.stream = stream or sys.stderr
        self.done, self.start = 0, time.perf_counter()

    def __call__(self, **facts):
        self.done += 1
        if self.enabled:
            event = {"event": "progress", "task": self.task, "done": self.done, "total": self.total,
                     "elapsed_s": round(time.perf_counter() - self.start, 3), **facts}
            self.stream.write(json.dumps(event, default=str) + "\n")
            self.stream.flush()


class Checkpoint:
    """Completed units of a task as JSON lines, keyed so a rerun skips what is done.

    The first line records the task's request hash; a checkpoint written for a
    different request is refused instead of silently mixing results.
    """

    def __init__(self, path, request):
        self.path, self.hash, self.rows = Path(path), digest(request), {}
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if lines:
                header = json.loads(lines[0])
                if header.get("request_sha256") != self.hash:
                    raise ValueError(f"checkpoint {path} belongs to a different request; delete it or choose "
                                     "another path")
                for line in lines[1:]:
                    if line.strip():
                        item = json.loads(line)
                        self.rows[item["key"]] = item["row"]
        else:
            self.path.write_text(json.dumps({"checkpoint": "quantum-circuit-simulator", "request_sha256": self.hash}) + "\n",
                                 encoding="utf-8")

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row):
        self.rows[key] = row
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "row": row}, allow_nan=False) + "\n")
