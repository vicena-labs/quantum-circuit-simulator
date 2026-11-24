#!/usr/bin/env bash
# The scoreboard: Quantum Circuit Simulator against every other tool, each run
# the way its authors recommend, on one machine, in one command.
#
#   benchmarks/independent/scoreboard.sh <out-dir> [threads] [workload ...]
#
# Needs Python with this checkout installed (`pip install .`) and qiskit,
# qiskit-aer, qulacs, pennylane, quimb, pauli-prop and numba
# (requirements-scoreboard.txt), git for sparse Pauli dynamics (spd),
# and Julia 1.10 or later; a PauliPropagation.jl environment is created in
# <out-dir>/julia-env on first use. PYTHON overrides the interpreter. Every run
# is a fresh process with a BENCH_TIMEOUT-second limit (default 120). The
# tables go to <out-dir>/scoreboard.md; the raw records stay beside them.
set -euo pipefail
# Stopping the scoreboard stops its drivers, and each driver kills its worker's
# process group, so no worker keeps running and loading the machine.
trap 'pkill -TERM -P $$ 2>/dev/null; exit 143' TERM INT
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
THREADS=${2:-1}
shift $(( $# < 2 ? $# : 2 ))
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
export BENCH_REPS=5 BENCH_TIMEOUT=${BENCH_TIMEOUT:-120} BENCH_THREADS=$THREADS BENCH_PC_THREADS=$THREADS
cd "$HERE"

echo "== checking the setup"
"$PY" -I - "$OUT/machine.json" "$THREADS" <<'EOF'
import json, os, platform, subprocess, sys
from importlib.metadata import version
tools = ["quantum-circuit-simulator", "qiskit", "qiskit-aer", "qulacs", "pennylane", "pennylane-lightning",
         "quimb", "pauli-prop"]
missing = []
versions = {}
for t in tools:
    try:
        versions[t] = version(t)
    except Exception:
        missing.append(t)
if missing:
    sys.exit(f"missing Python packages: {', '.join(missing)}")
cpu = platform.processor()
try:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
except FileNotFoundError:
    pass
if os.path.exists("/proc/cpuinfo"):
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), cpu)
mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
cpus = os.cpu_count()
# A container's CPU quota is invisible to os.cpu_count(); cgroup v2 states it.
try:
    quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
    if quota != "max":
        cpus = min(cpus, int(quota) // int(period))
except (OSError, ValueError):
    pass
try:
    limit = open("/sys/fs/cgroup/memory.max").read().strip()
    if limit != "max":
        mem = min(mem, int(limit) / 2**30)
except (OSError, ValueError):
    pass
json.dump(dict(cpu=cpu, logical_cpus=cpus, memory_gib=round(mem, 1), system=platform.platform(),
               python=platform.python_version(), threads=int(sys.argv[2]), versions=versions),
          open(sys.argv[1], "w"), indent=1)
print(json.dumps(versions))
EOF
command -v julia >/dev/null || { echo "Julia is not installed (https://julialang.org/install)"; exit 1; }
# Sparse Pauli dynamics is a research repository, not a package: pin the
# commit these adapters were checked against.
SPD_COMMIT=f694c9c
if [ ! -d "$OUT/spd" ]; then git clone -q https://github.com/tbegusic/spd.git "$OUT/spd"; fi
git -C "$OUT/spd" checkout -q "$SPD_COMMIT"
export SPD_PATH="$OUT/spd"
"$PY" -I -c "import json, sys; m = json.load(open(sys.argv[1])); m['versions']['spd'] = 'git ' + sys.argv[2]; json.dump(m, open(sys.argv[1], 'w'), indent=1)" "$OUT/machine.json" "$SPD_COMMIT"
if [ ! -f "$OUT/julia-env/Manifest.toml" ]; then
    julia --project="$OUT/julia-env" -e 'using Pkg; Pkg.add(["PauliPropagation", "JSON3"])'
fi
julia --project="$OUT/julia-env" -e 'using PauliPropagation; println("PauliPropagation.jl ", pkgversion(PauliPropagation))'

if [ $# -gt 0 ]; then WORKLOADS=("$@"); else
    read -r -a WORKLOADS <<< "$("$PY" -I -c "import importlib.util as u; s = u.spec_from_file_location('b', 'bench.py'); m = u.module_from_spec(s); s.loader.exec_module(m); print(' '.join(m.WORKLOADS))")"
fi
echo "== ${#WORKLOADS[@]} workloads, $THREADS thread(s)"
"$PY" -I bench.py export "$OUT/workloads.json" "${WORKLOADS[@]}"
rm -f "$OUT"/*.jsonl

echo "== Quantum Circuit Simulator, exact"
BENCH_GRAPH_WIDE=1 BENCH_TOOLS=pauli_cones,pauli_cones_graph "$PY" -I bench.py run "$OUT/qcs.jsonl" "${WORKLOADS[@]}"
for backend in native graph; do
    echo "== Quantum Circuit Simulator, threshold 1e-12, $backend"
    BENCH_BACKEND=$backend "$PY" -I trunc_bench.py run "$OUT/qcs_1e-12.jsonl" 1e-12 Inf "$THREADS" "${WORKLOADS[@]}"
done
for parts in whole cone split; do
    echo "== PauliPropagation.jl, $parts circuits"
    PP_MODE=tuned PP_PARTS=$parts "$PY" -I run_pp.py "$OUT/julia-env" "$OUT/workloads.json" "$OUT/pp.jsonl" \
        "$THREADS" 1e-12 Inf "${WORKLOADS[@]}"
done
for parts in "" cone split; do
    echo "== pauli-prop, spd, Qulacs, Aer, Lightning, quimb, ${parts:-whole} circuits"
    BENCH_CONES=$parts BENCH_PP_ATOL=1e-12 BENCH_TOOLS=pauli_prop,spd,qulacs,aer,lightning,quimb \
        "$PY" -I bench.py run "$OUT/others.jsonl" "${WORKLOADS[@]}"
done
"$PY" -I summarize_scoreboard.py "$OUT" > "$OUT/scoreboard.md"
echo "== done: $OUT/scoreboard.md"
