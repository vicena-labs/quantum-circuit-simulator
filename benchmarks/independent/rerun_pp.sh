#!/usr/bin/env bash
# Rerun only PauliPropagation.jl's rows of scoreboard runs already made (each
# argument is a result directory written by scoreboard.sh), with three variants
# per case in one process, so its limit is three times the others'. Keeps the
# previous rows in pp-previous.jsonl and rewrites the tables.
#
#   benchmarks/independent/rerun_pp.sh <out-dir> <threads> [<out-dir> <threads> ...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
while [ $# -ge 2 ]; do
    OUT=$(cd "$1" && pwd); THREADS=$2; shift 2
    read -r -a WORKLOADS <<< "$("$PY" -c "import json; print(' '.join(json.load(open('$OUT/workloads.json'))))")"
    [ -f "$OUT/pp-previous.jsonl" ] || cp "$OUT/pp.jsonl" "$OUT/pp-previous.jsonl"
    rm -f "$OUT/pp.jsonl"
    for parts in whole cone split; do
        echo "== PauliPropagation.jl, $parts circuits, $THREADS thread(s), $OUT"
        PP_MODE=tuned PP_PARTS=$parts BENCH_TIMEOUT=360 "$PY" -I "$HERE/run_pp.py" "$OUT/julia-env" "$OUT/workloads.json" \
            "$OUT/pp.jsonl" "$THREADS" 1e-12 Inf "${WORKLOADS[@]}"
    done
    "$PY" -I "$HERE/summarize_scoreboard.py" "$OUT" > "$OUT/scoreboard.md"
done
echo "== rerun done"
