"""Quantum Circuit Simulator against PauliPropagation.jl, back to back on an idle machine.

    python -I paired.py <out-dir> <julia-env> <threads> [workload ...]

Each PauliPropagation.jl variant and circuit form runs in its own process, so a
memory kill in one loses no other. Rounds alternate which tool goes first. A
run starts only once the container has been idle for a minute, is repeated if
other work used more than 5% of one core per second while it ran, and is marked
`loaded` if three attempts all were. After the first round only
configurations within 1.5 times of their tool's fastest, per evaluation or
per single run, run again. QCS runs exactly, PauliPropagation.jl at its 1e-12
threshold. Load is read from the Linux cgroup; a neighbour outside the
container is invisible. Rerunning the command resumes an interrupted run.
"""
import importlib.util, json, os, resource, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CGROUP = Path(os.environ.get("BENCH_CGROUP", "/sys/fs/cgroup"))
ROUNDS, ATTEMPTS, NEAR = 3, 3, 1.5
IDLE_CORES, RUN_CORES, OTHER_ANON = 0.3, 0.05, 6 * 2**30
WINDOW, WINDOWS = float(os.environ.get("BENCH_IDLE_WINDOW", 10)), 6
LIMIT = float(os.environ.get("BENCH_TIMEOUT", 360))
QCS = [("pauli_cones", "exact"), ("pauli_cones_graph", "exact")]
PP = [(variant, parts) for parts in ("whole", "cone", "split") for variant in ("plain", "cache", "zones")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keyed(path, key):
    for line in (CGROUP / path).read_text().splitlines():
        name, value = line.split()
        if name == key:
            return int(value)
    return 0


def foreign_cpu():
    """Seconds of CPU the container used that neither this process nor its finished workers did."""
    ours = sum(u.ru_utime + u.ru_stime for u in map(resource.getrusage, (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)))
    return keyed("cpu.stat", "usage_usec") / 1e6 - ours


def wait_idle(log):
    quiet, waited = 0, False
    while quiet < WINDOWS:
        before, t = foreign_cpu(), time.perf_counter()
        time.sleep(WINDOW)
        cores = (foreign_cpu() - before) / (time.perf_counter() - t)
        anon = keyed("memory.stat", "anon")
        quiet = quiet + 1 if cores < IDLE_CORES and anon < OTHER_ANON else 0
        if quiet == 0 and not waited:
            waited = True
            log(f"waiting: other work uses {cores:.1f} cores, {anon / 2**30:.1f} GiB")


def attempt(isolated, cmd, env, log):
    for n in range(1, ATTEMPTS + 1):
        if n > 1:
            wait_idle(log)
        oom, before, t = keyed("memory.events", "oom_kill"), foreign_cpu(), time.perf_counter()
        try:
            code, out, err = isolated.run(cmd, LIMIT, env)
            lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
            rec = json.loads(lines[-1][7:]) if lines else dict(error=isolated.failure(code, err))
        except subprocess.TimeoutExpired:
            rec = dict(error=f"timeout>{LIMIT:.0f}s")
        wall = time.perf_counter() - t
        cores = max(0.0, (foreign_cpu() - before) / wall)
        rec.update(wall_s=wall, foreign_cores=cores, attempt=n, loaded=cores > RUN_CORES,
                   anon_after=keyed("memory.stat", "anon"), oom_kills=keyed("memory.events", "oom_kill") - oom)
        # A worker killed with helpers still running leaves their CPU time
        # uncounted, which reads as load: a timeout is never repeated for it.
        if not rec["loaded"] or "timeout" in str(rec.get("error")):
            break
    return rec


def times(rec):
    """(one more evaluation, one run with setup), or None for a failed run."""
    if rec.get("error") or rec.get("warm_value_s") is None:
        return None
    return rec["warm_value_s"], rec["warm_value_s"] + rec["rebuild_s"]


def near_fastest(records, side):
    """The configurations worth repeating after round one."""
    done = {r["config"]: times(r) for r in records if r["side"] == side and times(r)}
    if not done:
        return set()
    best = [min(t[k] for t in done.values()) for k in (0, 1)]
    return {c for c, t in done.items() if any(t[k] <= NEAR * best[k] for k in (0, 1))}


def main(out, julia_env, threads, names):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    julia_env = str(Path(julia_env).resolve())
    bench, isolated = _load("bench"), _load("isolated")
    names = names or list(bench.WORKLOADS)
    path, data = out / "paired.jsonl", out / "workloads.json"
    log = lambda text: print(time.strftime("%H:%M:%S"), text, flush=True)
    env = dict(os.environ, BENCH_THREADS=threads, BENCH_PC_THREADS=threads, BENCH_REPS="5", BENCH_GRAPH_WIDE="1")
    subprocess.run([sys.executable, "-I", str(HERE / "bench.py"), "export", str(data), *names], env=env, check=True)
    records = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for round_ in range(ROUNDS):
        for name in names:
            mine = [r for r in records if r["workload"] == name]
            first = [r for r in mine if r["round"] == 0]
            jobs = [("qcs", f"{t} {m}", [sys.executable, "-I", str(HERE / "bench.py"), "worker", name, t], {}) for t, m in QCS]
            jobs += [("pp", f"{v} {p}", ["julia", f"--project={julia_env}", "-t", threads, str(HERE / "pp_tuned.jl"),
                                         str(data), name, p, "1e-12"], {"PP_VARIANT": v}) for v, p in PP]
            if round_:
                keep = near_fastest(first, "qcs") | near_fastest(first, "pp")
                jobs = [j for j in jobs if j[1] in keep]
            if round_ % 2:
                jobs.reverse()
            jobs = [j for j in jobs if not any(r["round"] == round_ and r["config"] == j[1] for r in mine)]
            if jobs:
                wait_idle(log)
            for side, config, cmd, extra in jobs:
                rec = attempt(isolated, cmd, dict(env, **{k: threads for k in bench.THREADS}, **extra), log)
                rec.update(workload=name, side=side, config=config, round=round_, threads=int(threads))
                records.append(rec)
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"round {round_} {name} {side} {config}: {rec.get('error') or rec['warm_value_s']}"
                    + (" LOADED" if rec["loaded"] else ""))
    (out / "paired.md").write_text(table(records, names, threads))
    log("== paired run done")


def summarise(records, side, k):
    """(median over rounds, configuration) of the side's fastest unloaded configuration."""
    by = {}
    for r in records:
        if r["side"] == side and times(r) and not r["loaded"]:
            by.setdefault(r["config"], []).append(times(r)[k])
    return min(((statistics.median(v), c, len(v)) for c, v in by.items()), default=None)


def table(records, names, threads):
    out = [f"# Paired run, {threads} thread(s)", "",
           "Median over rounds of each tool's fastest configuration; runs that shared the machine are left out.", "",
           "| Workload | QCS repeated | PauliPropagation.jl repeated | Lead | QCS single | PauliPropagation.jl single | Lead |",
           "|---|---|---|---|---|---|---|"]
    for name in names:
        mine, cells = [r for r in records if r["workload"] == name], []
        for k in (0, 1):
            us, them = summarise(mine, "qcs", k), summarise(mine, "pp", k)
            cells += [f"{x[0] * 1e3:.4g} ms ({x[1]}, {x[2]} rounds)" if x else reason(mine, side)
                      for x, side in ((us, "qcs"), (them, "pp"))]
            cells.append(f"{them[0] / us[0]:.3g}×" if us and them else "—")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def reason(records, side):
    errors = [str(r.get("error") or "loaded")[:40].split(":")[0] for r in records if r["side"] == side]
    return "; ".join(sorted(set(errors))) or "not run"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
