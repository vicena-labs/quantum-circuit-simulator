"""Quantum Circuit Simulator truncated mode on bench.py workloads, one fresh process per run.

    python trunc_bench.py run <out.jsonl> <abs_coeff> <max_weight|Inf> <threads> <workload ...>
    python trunc_bench.py worker <workload> <abs_coeff> <max_weight|Inf> <threads>   (internal)

abs_coeff is PauliPropagation.jl's absolute min_abs_coeff; it is converted to
Quantum Circuit Simulator's relative min_coefficient by dividing by the observable l1 norm,
so both tools discard the same coefficients. BENCH_BACKEND=graph runs the
truncated graph backend instead of native.
"""
import json, os, statistics, subprocess, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.perf_counter()


def worker(name, coeff, weight, threads):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    from quantum_circuit_simulator import Circuit, Truncation
    w = bench.WORKLOADS[name]
    norm = sum(abs(c) for c, _ in w["obs"])
    backend = os.environ.get("BENCH_BACKEND", "native")
    rec = dict(workload=name, tool="pauli_cones_truncated" + ("_graph" if backend == "graph" else ""), abs_coeff=coeff, min_coefficient=coeff / norm,
               max_weight=weight, threads=threads)
    try:
        t = time.perf_counter()
        c = Circuit(w["n"], w["nparams"], max_terms=1_000_000)
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"): c.gate(op[0], [op[1]], theta=op[2], param=op[3])
            elif op[0] == "CZ": c.gate("CZ", [op[1], op[2]])
            else: c.depolarize(list(op[1]), op[2])
        rec["build_s"] = time.perf_counter() - t
        t = time.perf_counter()
        plan = c.prepare(w["obs"], truncation=Truncation(coeff / norm, None if weight == "Inf" else int(weight)),
                         threads=threads, backend=backend, graph_max_nodes=10_000_000, graph_max_edges=50_000_000)
        rec["prepare_s"] = time.perf_counter() - t
        vecs = [[float(v) for v in vec] for vec in bench.vectors(w)]
        reps = (len(vecs) - 1) // 2
        t = time.perf_counter(); r = plan.evaluate(vecs[0], gradients=False); rec["first_value_s"] = time.perf_counter() - t
        rec.update(value=r.value, error_bound=r.error_bound, backend=r.backend, since_start_to_first_value_s=time.perf_counter() - T0)
        ts = []
        for s in vecs[1:1 + reps]:
            t = time.perf_counter(); plan.evaluate(s, gradients=False); ts.append(time.perf_counter() - t)
        rec["warm_value_s"] = statistics.median(ts)
        g = plan.evaluate(vecs[0])
        rec.update(grad_value=g.value, gradient=list(g.gradient), gradient_error_bound=list(g.gradient_error_bound))
        ts = []
        for s in vecs[1 + reps:]:
            t = time.perf_counter(); plan.evaluate(s); ts.append(time.perf_counter() - t)
        rec["warm_grad_s"] = statistics.median(ts)
        t = time.perf_counter()
        c = Circuit(w["n"], w["nparams"], max_terms=1_000_000)
        for op in w["ops"]:
            if op[0] in ("RX", "RY", "RZ"): c.gate(op[0], [op[1]], theta=op[2], param=op[3])
            elif op[0] == "CZ": c.gate("CZ", [op[1], op[2]])
            else: c.depolarize(list(op[1]), op[2])
        c.prepare(w["obs"], truncation=Truncation(coeff / norm, None if weight == "Inf" else int(weight)),
                  threads=threads, backend=backend, graph_max_nodes=10_000_000, graph_max_edges=50_000_000)
        rec["rebuild_s"] = time.perf_counter() - t
        d = plan.diagnostics
        rec.update(peak_terms=d.get("max_single_cone_terms"), truncated_terms=d.get("truncated_terms"),
                   cone_groups=d.get("cone_groups") or d.get("unique_cone_groups"))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print("RESULT " + json.dumps(rec))


if sys.argv[1] == "worker":
    worker(sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("isolated", os.path.join(HERE, "isolated.py"))
    isolated = importlib.util.module_from_spec(spec); spec.loader.exec_module(isolated)
    out, coeff, weight, threads, *names = sys.argv[2:]
    timeout = float(os.environ.get("BENCH_TIMEOUT", 180))
    for name in names:
        t = time.perf_counter()
        try:
            code, stdout, stderr = isolated.run([sys.executable, "-I", __file__, "worker", name, coeff, weight, threads],
                                                timeout)
            lines = [l for l in stdout.splitlines() if l.startswith("RESULT ")]
            rec = json.loads(lines[-1][7:]) if lines else dict(workload=name, error=isolated.failure(code, stderr))
        except subprocess.TimeoutExpired:
            rec = dict(workload=name, error=f"timeout>{timeout:.0f}s")
        rec["cold_process_wall_s"] = time.perf_counter() - t
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        print({k: rec.get(k) for k in ("workload", "abs_coeff", "max_weight", "threads", "value", "error_bound",
                                       "warm_value_s", "warm_grad_s", "peak_terms", "error")}, flush=True)
