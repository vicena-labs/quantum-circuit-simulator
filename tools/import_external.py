"""Regenerate the bundled real-chip and benchmark-circuit data from openly licensed packages.

    python -m pip install qiskit-ibm-runtime cirq-google "mqt.bench" supermarq
    python tools/import_external.py

Writes src/quantum_circuit_simulator/data/devices.json.gz and circuits.json.gz.
Sources, all redistributable with attribution (see THIRD_PARTY.md):
- IBM calibration snapshots: qiskit-ibm-runtime's fake_provider (Apache-2.0);
  qubit coordinates from each snapshot, else Qiskit's gate maps (Apache-2.0).
- Google calibrations and device specifications: cirq-google (Apache-2.0).
- Benchmark circuits: MQT Bench (MIT) and Supermarq (Apache-2.0).
The output is data, regenerated deliberately; the package never imports these sources.
"""
from __future__ import annotations

import gzip
import importlib.metadata as metadata
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "src" / "quantum_circuit_simulator" / "data"
# Circuits generated so far, one JSON line per (benchmark, size), so an interrupted run resumes.
CACHE = ROOT / "build" / "import-circuits.jsonl"


def main():
    """`python tools/import_external.py [devices|circuits]`; both when no part is named."""
    parts = sys.argv[1:] or ["devices", "circuits"]
    DATA.mkdir(exist_ok=True)
    if "devices" in parts:
        devices = {**ibm_devices(), **google_devices()}
        write(DATA / "devices.json.gz", {"schema": "quantum-circuit-simulator-device-bundle/1", "devices": devices})
        print(f"{len(devices)} devices", flush=True)
    if "circuits" in parts:
        circuits = mqt_circuits() + supermarq_circuits()
        write(DATA / "circuits.json.gz", {"schema": "quantum-circuit-simulator-circuit-bundle/1", "circuits": circuits})
        print(f"{len(circuits)} circuits", flush=True)


def write(path, payload):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------- shared conversion
def clip_error(e):
    """Device errors must lie in [0, 1); a calibration of 1 marks a faulty element."""
    return None if e is None or not math.isfinite(e) else min(max(float(e), 0.0), 0.999)


def clip_readout(p):
    return None if p is None or not math.isfinite(p) else min(max(float(p), 0.0), 0.499)


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def device_dict(name, n, edges, *, native, qubits, couplers, time_1q, time_2q, coordinates, description, source,
                hardware):
    """A device description in the package's JSON format from per-qubit and per-coupler values.

    qubits: q -> {error_1q, readout (p01, p10), t1, t2}; couplers: (a, b) -> two-qubit infidelity.
    Chip-wide values are medians; each qubit and coupler keeps its own as an override.
    T2 is capped at 2 T1, the physical limit, where a calibration exceeds it.
    """
    capped = 0
    overrides = []
    for q in range(n):
        v = qubits.get(q, {})
        t1, t2 = v.get("t1"), v.get("t2")
        if t1 and t2 and t2 > 2 * t1:
            t2, capped = 2 * t1, capped + 1
        entry = {"qubit": q}
        if v.get("error_1q") is not None:
            entry["error_1q"] = clip_error(v["error_1q"])
        if v.get("readout") and None not in v["readout"]:
            entry["readout"] = [clip_readout(p) for p in v["readout"]]
        if t1:
            entry["t1"] = t1
        if t2:
            entry["t2"] = t2
        if len(entry) > 1:
            overrides.append(entry)
    t1s = [o.get("t1") for o in overrides]
    t2s = [o.get("t2") for o in overrides]
    chip_t1, chip_t2 = median(t1s), median(t2s)
    if chip_t1 and chip_t2 and chip_t2 > 2 * chip_t1:
        chip_t2 = 2 * chip_t1
    readouts = [o["readout"] for o in overrides if "readout" in o]
    hardware = {k: v for k, v in hardware.items() if v not in (None, {}, [])}
    hardware["t2_capped_at_2t1"] = capped
    return {
        "schema": "quantum-circuit-simulator-device/1", "name": name, "family": "superconducting",
        "n_qubits": n, "edges": [list(e) for e in sorted(edges)], "native_gates": native,
        "error_1q": clip_error(median([o.get("error_1q") for o in overrides])),
        "error_2q": clip_error(median(list(couplers.values()))),
        "readout": [clip_readout(median([r[0] for r in readouts])), clip_readout(median([r[1] for r in readouts]))]
        if readouts else None,
        "t1": chip_t1, "t2": chip_t2, "time_1q": time_1q, "time_2q": time_2q,
        "description": description, "source": source,
        "parameters": {"real_device": True, "hardware": hardware},
        "coupler_errors": [[a, b, clip_error(e)] for (a, b), e in sorted(couplers.items()) if e is not None],
        "qubit_overrides": overrides,
        "coordinates": [list(map(float, xy)) for xy in coordinates] if coordinates else [],
    }


# ---------------------------------------------------------------- IBM
def ibm_devices():
    import inspect
    import qiskit
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit.visualization import gate_map
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    version = metadata.version("qiskit-ibm-runtime")
    qiskit_maps = _qiskit_coordinates(gate_map)
    out = {}
    for backend in FakeProviderForBackendV2().backends():
        if backend.num_qubits < 2 or backend.name in ("fake_fractional", "fake_nighthawk"):
            continue   # one-qubit Armonk; synthetic, not calibrated, devices
        folder = Path(inspect.getfile(type(backend))).parent
        conf = json.loads(next(folder.glob("conf_*.json")).read_text())
        props = json.loads(next(folder.glob("props_*.json")).read_text())
        out.update([_ibm_device(backend, conf, props, qiskit_maps, version, qiskit.__version__)])
    return out


def _qiskit_coordinates(gate_map):
    """Qiskit's published (row, column) qubit maps by qubit count."""
    import ast
    import inspect
    import re
    source = inspect.getsource(gate_map)
    maps = {}
    for size in map(int, set(re.findall(r"qubit_coordinates_map\[(\d+)\]", source))):
        start = source.find("[", source.find(f"qubit_coordinates_map[{size}] =") + len(f"qubit_coordinates_map[{size}] ="))
        depth, end = 0, start
        for end in range(start, len(source)):   # the matching closing bracket
            depth += {"[": 1, "]": -1}.get(source[end], 0)
            if depth == 0:
                break
        try:
            maps[size] = ast.literal_eval(source[start:end + 1])
        except (ValueError, SyntaxError):
            continue
    return maps


def _ibm_device(backend, conf, props, qiskit_maps, version, qiskit_version):
    n = backend.num_qubits
    target = backend.target
    two = [g for g in ("ecr", "cz", "cx") if g in target.operation_names]
    one = [g for g in ("rz", "sx", "x") if g in target.operation_names]
    qubits = {}
    for q, items in enumerate(props["qubits"]):
        values = {item["name"]: item["value"] for item in items}
        t1 = values.get("T1")
        t2 = values.get("T2")
        p01, p10 = values.get("prob_meas1_prep0"), values.get("prob_meas0_prep1")
        if p01 is None and values.get("readout_error") is not None:
            p01 = p10 = values["readout_error"]
        sx = target["sx"].get((q,)) if "sx" in target.operation_names else None
        qubits[q] = {"t1": t1 * 1e-6 if t1 else None, "t2": t2 * 1e-6 if t2 else None,
                     "readout": (p01, p10) if p01 is not None else None,
                     "error_1q": sx.error if sx is not None else None,
                     "frequency_ghz": values.get("frequency"), "anharmonicity_ghz": values.get("anharmonicity")}
    couplers, durations = {}, []
    for gate in two:
        for qargs, prop in target[gate].items():
            if prop is None or len(qargs) != 2:
                continue
            key = tuple(sorted(qargs))
            if prop.error is not None:
                couplers[key] = min(couplers.get(key, 1.0), prop.error)
            if prop.duration:
                durations.append(prop.duration)
    edges = set(couplers) | {tuple(sorted(pair)) for pair in conf.get("coupling_map", [])}
    one_durations = [p.duration for p in target["sx"].values() if p is not None and p.duration] if "sx" in target.operation_names else []
    coordinates = None
    if conf.get("coords") and len(conf["coords"]) == n:
        coordinates = [(x, -y) for x, y in conf["coords"]]
    elif n in qiskit_maps and len(qiskit_maps[n]) == n:
        coordinates = [(c, -r) for r, c in qiskit_maps[n]]
    processor = conf.get("processor_type") or {}
    name = "ibm_" + backend.name.removeprefix("fake_")
    family = f"{processor.get('family', 'IBM')} r{processor.get('revision', '?')}".strip()
    date = props.get("last_update_date", "")[:10]
    hardware = {
        "vendor": "IBM", "processor": family, "native_two_qubit_gate": two[0] if two else None,
        "snapshot_date": date, "quantum_volume": conf.get("quantum_volume"), "clops": conf.get("clops_h") or conf.get("clops"),
        "qubit_frequency_ghz": [qubits[q]["frequency_ghz"] for q in range(n)],
        "anharmonicity_ghz": [qubits[q]["anharmonicity_ghz"] for q in range(n)],
        "measurement_duration_s": median([props_value(items, "readout_length", 1e-9) for items in props["qubits"]]),
    }
    if not any(hardware["qubit_frequency_ghz"]):
        hardware.pop("qubit_frequency_ghz")
    if not any(v for v in hardware["anharmonicity_ghz"] if v):
        hardware.pop("anharmonicity_ghz")
    return name, device_dict(
        name, n, edges, native={"one_qubit": one, "two_qubit": two[:1]}, qubits=qubits, couplers=couplers,
        time_1q=median(one_durations), time_2q=median(durations), coordinates=coordinates,
        description=f"IBM {backend.name.removeprefix('fake_').capitalize()} ({family}, {n} qubits, native "
                    f"{(two or ['?'])[0].upper()}): measured calibration of {date}, every qubit and coupler with its own values.",
        source=f"IBM calibration snapshot of {date}, distributed in qiskit-ibm-runtime {version} fake_provider "
               f"(Apache-2.0); coordinates from the snapshot or Qiskit {qiskit_version} gate maps.",
        hardware=hardware)


def props_value(items, name, scale):
    for item in items:
        if item["name"] == name and item.get("value") is not None:
            return item["value"] * scale
    return None


# ---------------------------------------------------------------- Google
GOOGLE = {
    # processor: (description, one-qubit duration s, two-qubit duration s, two-qubit metric, two-qubit gate)
    "rainbow": ("Google Rainbow (Sycamore family, 23 qubits)", 25e-9, 32e-9,
                "two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle", "sqrt_iswap"),
    "weber": ("Google Weber (Sycamore family, 53 qubits)", 25e-9, 32e-9,
              "two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle", "sqrt_iswap"),
    "willow_pink": ("Google Willow Pink (Willow family, 105 qubits)", 25e-9, 42e-9,
                    "two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle", "cz"),
}


def google_devices():
    import cirq_google as cg
    version = metadata.version("cirq-google")
    out = {}
    for processor, (text, t1q, t2q, metric, gate) in GOOGLE.items():
        device = cg.engine.create_device_from_processor_id(processor)
        cal = cg.engine.load_median_device_calibration(processor)
        grid = sorted(device.metadata.qubit_set)
        index = {q: i for i, q in enumerate(grid)}
        metrics = {k: {key: v[0] for key, v in cal[k].items()} for k in cal.keys()}

        def one(name, q):
            return metrics.get(name, {}).get((q,))
        qubits = {}
        for q, i in index.items():
            pauli = one("single_qubit_rb_pauli_error_per_gate", q)
            t1 = one("single_qubit_idle_t1_micros", q)
            # Average infidelity is d/(d+1) times the Pauli error: 2/3 for one qubit, 4/5 for two.
            qubits[i] = {"error_1q": pauli * 2 / 3 if pauli is not None else None,
                         "readout": (one("single_qubit_p00_error", q), one("single_qubit_p11_error", q)),
                         "t1": t1 * 1e-6 if t1 else None}
        couplers = {}
        for pair in device.metadata.qubit_pairs:
            a, b = sorted(pair)
            pauli = metrics.get(metric, {}).get((a, b)) or metrics.get(metric, {}).get((b, a))
            couplers[tuple(sorted((index[a], index[b])))] = pauli * 4 / 5 if pauli is not None else None
        edges = set(couplers)
        couplers = {k: v for k, v in couplers.items() if v is not None}
        name = "google_" + processor
        out[name] = device_dict(
            name, len(grid), edges, native={"one_qubit": ["phased_xz", "rz"], "two_qubit": [gate]},
            qubits=qubits, couplers=couplers, time_1q=t1q, time_2q=t2q,
            coordinates=[(q.col, -q.row) for q in grid],
            description=f"{text}, native {gate.upper()}: median calibration from Google's published device data, "
                        "every qubit and coupler with its own values. T2 is not published here, so idle dephasing "
                        "follows T1 only.",
            source=f"Google calibration and device specification distributed in cirq-google {version} (Apache-2.0); "
                   "gate durations from Google's publications.",
            hardware={"vendor": "Google", "processor": text, "native_two_qubit_gate": gate,
                      "grid_qubits": [[q.row, q.col] for q in grid]})
    return out


# ---------------------------------------------------------------- circuits
MQT_SIZES = (4, 8, 16, 32, 64, 127)
# Benchmarks with mid-circuit measurement or classical control, which the simulators do not run.
MQT_DYNAMIC = {"dynamic_qft", "ghz_dynamic", "iqpe", "teleportation", "superdense_coding", "seven_qubit_steane_code",
               "shors_nine_qubit_code"}
MAX_GATES = 20_000


def mqt_circuits():
    """Every static MQT Bench benchmark at each size, each generated in its own process with a time limit."""
    import multiprocessing as mp
    from mqt.bench.benchmarks import get_available_benchmark_names
    CACHE.parent.mkdir(exist_ok=True)
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            entry = json.loads(line)
            done[tuple(entry["job"])] = entry["item"]
    jobs = [(name, size) for name in get_available_benchmark_names() if name not in MQT_DYNAMIC for size in MQT_SIZES
            if (name, size) not in done]
    print(f"  {len(done)} MQT jobs cached, {len(jobs)} to run", flush=True)
    context = mp.get_context("spawn")
    results, running = dict(done), []

    def record(job, item):
        results[job] = item
        with CACHE.open("a") as f:
            f.write(json.dumps({"job": list(job), "item": item}) + "\n")
    while jobs or running:
        while jobs and len(running) < 4:
            job = jobs.pop(0)
            queue = context.Queue()
            process = context.Process(target=_mqt_worker, args=(job, queue), daemon=True)
            process.start()
            running.append((job, process, queue, time.monotonic()))
        time.sleep(0.2)
        for entry in list(running):
            job, process, queue, started = entry
            if not queue.empty():
                record(job, queue.get())
                process.join(5)
                running.remove(entry)
            elif not process.is_alive():
                running.remove(entry)
                record(job, None)
                print(f"  skipped mqt/{job[0]} at {job[1]}: failed", flush=True)
            elif time.monotonic() - started > 120:
                process.terminate()
                running.remove(entry)
                record(job, None)
                print(f"  skipped mqt/{job[0]} at {job[1]}: over 120 s", flush=True)
    out, seen = [], set()
    for job in sorted(results, key=lambda j: (j[0], j[1])):
        item = results[job]
        if item and (item["name"], item["n_qubits"]) not in seen:
            seen.add((item["name"], item["n_qubits"]))
            out.append(item)
    print(f"  {len(out)} MQT Bench circuits", flush=True)
    return out


def _mqt_worker(job, queue):
    try:
        queue.put(_mqt_one(*job))
    except Exception:
        queue.put(None)


def _mqt_one(name, size):
    import warnings
    warnings.filterwarnings("ignore")
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qiskit import qasm2, transpile
    from quantum_circuit_simulator.interop import SUPPORTED_GATES
    qc = get_benchmark(name, BenchmarkLevel.INDEP, circuit_size=size)
    if qc.num_qubits > 156:
        return None
    qc.remove_final_measurements()
    if any(i.operation.name not in SUPPORTED_GATES + ["barrier"] for i in qc.data):
        qc = transpile(qc, basis_gates=SUPPORTED_GATES, optimization_level=0)
    if qc.size() > MAX_GATES or any(i.operation.name in ("measure", "reset") for i in qc.data):
        return None
    return {"name": f"mqt/{name}", "suite": "MQT Bench", "n_qubits": qc.num_qubits, "gates": qc.size(),
            "two_qubit_gates": sum(1 for i in qc.data if len(i.qubits) == 2), "qasm": qasm2.dumps(qc),
            "source": f"MQT Bench {metadata.version('mqt.bench')}, target-independent level (MIT)"}


SUPERMARQ = ("GHZ", "MerminBell", "HamiltonianSimulation", "QAOAVanillaProxy", "QAOAFermionicSwapProxy", "VQEProxy")


def supermarq_circuits():
    import cirq
    import supermarq.benchmarks as sb
    from qiskit import qasm2
    from quantum_circuit_simulator.interop import SUPPORTED_GATES
    version = metadata.version("supermarq")
    out = []
    for cls in SUPERMARQ:
        for size in (4, 8, 16, 32):
            if cls == "MerminBell" and size > 8:
                continue   # its circuits grow exponentially
            if cls in ("VQEProxy", "QAOAVanillaProxy", "QAOAFermionicSwapProxy") and size > 8:
                continue   # these proxies fit their angles by simulating the circuit first
            print(f"  supermarq {cls} at {size}", flush=True)
            try:
                bench = getattr(sb, cls)(size) if cls != "VQEProxy" else sb.VQEProxy(size, 1)
                circuits = bench.circuit()
            except Exception:
                continue
            for k, circuit in enumerate(circuits if isinstance(circuits, list) else [circuits]):
                circuit = cirq.drop_terminal_measurements(circuit) if hasattr(cirq, "drop_terminal_measurements") else circuit
                circuit = circuit[:-1] if any(cirq.is_measurement(op) for op in circuit[-1:].all_operations()) else circuit
                try:
                    text = cirq.qasm(circuit)
                    from qiskit import QuantumCircuit, transpile
                    qc = QuantumCircuit.from_qasm_str(text)
                    qc.remove_final_measurements()
                    if any(i.operation.name in ("measure", "reset") for i in qc.data):
                        continue
                    qc = transpile(qc, basis_gates=SUPPORTED_GATES, optimization_level=0)
                except Exception:
                    continue
                label = f"supermarq/{cls.lower()}" + (f"-{k}" if isinstance(circuits, list) and len(circuits) > 1 else "")
                out.append({"name": label, "suite": "Supermarq", "n_qubits": qc.num_qubits, "gates": qc.size(),
                            "two_qubit_gates": sum(1 for i in qc.data if len(i.qubits) == 2),
                            "qasm": qasm2.dumps(qc), "source": f"Supermarq {version} (Apache-2.0)"})
    return out


if __name__ == "__main__":
    main()
