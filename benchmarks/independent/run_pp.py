"""Drive pp_bench.jl: one fresh Julia process per (workload, threads, truncation).

    python run_pp.py <julia-env> <workloads.json> <out.jsonl> <threads> <min_abs_coeff> <max_weight> <workload ...>
    PP_MODE=surrogate times the reusable surrogate instead.
    PP_MODE=tuned runs pp_tuned.jl, the authors' recommended setup, on the
    circuit parts named by PP_PARTS (whole, cone or split); max_weight is ignored.
Cold wall time = process start (incl. Julia startup, package load, JIT) to exit.
"""
import json, os, subprocess, sys, time

def _isolated():
    # Loaded by path: drivers run with python -I, which leaves this directory off sys.path.
    import importlib.util
    spec = importlib.util.spec_from_file_location("isolated", os.path.join(os.path.dirname(os.path.abspath(__file__)), "isolated.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


isolated = _isolated()
env_dir, wl, out, threads, tol, maxw, *names = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
timeout = float(os.environ.get("BENCH_TIMEOUT", 180))
for name in names:
    t = time.perf_counter()
    try:
        mode = os.environ.get("PP_MODE", "propagate")
        args = ([os.path.join(here, "pp_tuned.jl"), wl, name, os.environ.get("PP_PARTS", "whole"), tol] if mode == "tuned"
                else [os.path.join(here, "pp_bench.jl"), wl, name, tol, maxw, mode])
        code, stdout, stderr = isolated.run(["julia", f"--project={env_dir}", "-t", threads, *args], timeout)
        lines = [l for l in stdout.splitlines() if l.startswith("RESULT ")]
        rec = json.loads(lines[-1][7:]) if lines else dict(workload=name, error=isolated.failure(code, stderr))
    except subprocess.TimeoutExpired:
        rec = dict(workload=name, error=f"timeout>{timeout:.0f}s")
    rec.setdefault("tool", "PauliPropagation.jl" + {"surrogate": " surrogate", "tuned": " tuned"}.get(os.environ.get("PP_MODE"), ""))
    if os.environ.get("PP_MODE") == "tuned":
        rec.setdefault("parts_mode", os.environ.get("PP_PARTS", "whole"))
    rec.update(threads=int(threads), min_abs_coeff=float(tol), max_weight=maxw,
               cold_process_wall_s=time.perf_counter() - t)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print({k: rec.get(k) for k in ("workload", "parts_mode", "threads", "min_abs_coeff", "value", "warm_value_s",
                                   "warm_grad_s", "final_terms", "error")}, flush=True)
