# Independent comparison harness

**The scoreboard** is the comparison we publish from. One command runs every tool on one machine, each the way its authors recommend, and writes a table that marks every workload where Quantum Circuit Simulator is less than ten times faster than the fastest alternative:

```bash
pip install . -r benchmarks/independent/requirements-scoreboard.txt  # this checkout, the other tools; plus Julia 1.10 or later
benchmarks/independent/scoreboard.sh results/scoreboard-1t 1          # one thread per tool
benchmarks/independent/scoreboard.sh results/scoreboard-all 16             # every core of a 16-CPU machine
```

It checks that every tool is installed before it starts, records the machine and every version in `machine.json`, and writes `scoreboard.md`. The first published run, on a 16-CPU Vicena computer on 23–24 November 2026, is in [results-2026-11-24-vicena](results-2026-11-24-vicena/README.md). What it compares, and why, is in [the speed plan](../../docs/speed-plan.md):

- **Competitors at their best.** PauliPropagation.jl runs with `VectorPauliSum`, its `Performance` kernels and `min_abs_coeff = 1e-12` (`pp_tuned.jl`), in the fastest of a plain call, a workspace sized up front with `propagate!`, and `MultiPauliSum` zones across threads, as its documentation's performance example recommends; pauli-prop runs with `atol = 1e-12`. Sparse Pauli dynamics ([spd](https://github.com/tbegusic/spd), the code behind Begušić, Gray and Chan's simulation of IBM's experiment) runs as in its IBM example: the Clifford parts of every rotation move into the Pauli frame once, commuting rotations about the same Pauli are merged, and each evaluation propagates only what remains, at the same threshold; it has no noise channels or gradients. Every competitor also runs on circuits cut to the observable's light cone and relabelled onto the fewest qubits, as one circuit (`cone`) or one per observable term (`split`), and counts its fastest setup (`cone_parts` in `bench.py`).
- **Like for like.** Quantum Circuit Simulator reports exact results and results at the same 1e-12 threshold, each with its error against the exact value.
- **Two regimes.** *Repeated* times one more evaluation of a prepared circuit, as in an optimiser; *single* adds that circuit's setup, including our graph compilation and excluding once-per-process costs such as imports and Julia's compilation.

These scripts compare Quantum Circuit Simulator with Qiskit Aer, Qulacs, PennyLane Lightning, IBM pauli-prop, quimb, PauliPropagation.jl and sparse Pauli dynamics on identical circuits, observables and parameter vectors. Each run is a fresh process with a time limit, and refusals and timeouts are recorded as results. The current results are [the Vicena computer run](results-2026-11-24-vicena/README.md), summarised in [measured performance](../../docs/performance.md#comparison-on-a-vicena-computer); the earlier [laptop run](results-2026-11-20-laptop/v5_tables.md), which ran PauliPropagation.jl with its slower defaults, and the [CPU comparison](../results/README.md) on a Linux virtual machine are kept as history.

| File | Purpose |
| --- | --- |
| `bench.py` | Workloads and one adapter per Python tool. `run` times exact evaluation; `export` writes the workloads and timed vectors as JSON for Julia |
| `pp_bench.jl`, `run_pp.py` | PauliPropagation.jl: exact, truncated (`min_abs_coeff`, `max_weight`) or its reusable surrogate (`PP_MODE=surrogate`) |
| `trunc_bench.py` | Quantum Circuit Simulator truncated mode (`BENCH_BACKEND=native` or `graph`), converting an absolute coefficient threshold to Quantum Circuit Simulator's relative one |
| `scoreboard.sh`, `summarize_scoreboard.py` | The scoreboard: setup check, every run, and the tables |
| `speed_chart.py` | The README's speed chart (`docs/images/speed-*.png`) and its data for the Vicena website, from a scoreboard run |
| `paired.py` | Quantum Circuit Simulator and PauliPropagation.jl back to back, only while the container is otherwise idle, one process per variant and three alternating rounds |
| `rerun_pp.sh` | Reruns only PauliPropagation.jl in existing scoreboard results and rewrites their tables |
| `isolated.py` | Runs each worker in its own process group, so a timeout or a stopped run kills the worker and everything it started |
| `pp_tuned.jl` | PauliPropagation.jl as its authors recommend, on whole, cone or split circuits |
| `summarize_v5.py` | Builds the 20 November tables from their raw JSONL files |

## Workloads

- **A:** noiseless transverse-field Ising energy on grids, 12–24 qubits.
- **B:** noisy mean magnetisation with two-qubit depolarizing noise, 10–1,024 qubits.
- **C:** a noiseless central Z on 100- and 1,024-qubit grids, 4–8 layers.
- **D:** mean magnetisation with single-qubit depolarizing noise, 10–1,024 qubits.
- **K:** IBM's 127-qubit heavy-hex kicked-Ising experiment (Kim et al., Nature 618, 500 (2023)), 5 and 20 steps, with θ_h as one shared parameter.

RZZ(−π/2) is written as CZ followed by RZ(−π/2) on both qubits, which is exact up to a global phase; every tool agrees with Qiskit's RZZ circuit to 2e-14.

## Running

```bash
pip install qiskit qiskit-aer qulacs pennylane quimb pauli-prop quantum-circuit-simulator
BENCH_REPS=5 python bench.py export results/workloads.json
BENCH_THREADS=1 BENCH_TOOLS=pauli_cones,pauli_cones_graph,qulacs,aer,lightning,quimb,pauli_prop \
    BENCH_GRAPH_WIDE=1 BENCH_TIMEOUT=120 python -I bench.py run results/exact_1t.jsonl
python run_pp.py <julia-env> results/workloads.json results/pp_exact_1t.jsonl 1 0 Inf <workload ...>
BENCH_BACKEND=graph python -I trunc_bench.py run results/trunc_graph.jsonl 1e-4 Inf 1 <workload ...>
python summarize_v5.py results > tables.md
```

`trunc_bench.py` and `summarize_v5.py` read the workloads and timed vectors from `results/workloads.json`, which `bench.py export` regenerates exactly (the vectors are seeded); it is not stored because of its size. Run one tool at a time on an otherwise idle machine. Report every tool at 1 thread and at all cores, and keep refusals and timeouts in the tables.

## Lessons from building it

- **Adapter bugs:** each of these gave wrong answers or spurious failures until it was fixed and checked against a reference:
  - shifts added to fixed rotations;
  - a variable assigned inside a Julia `try` block that hid a function;
  - adapters that handled only two-qubit depolarizing noise.

  Every adapter is now checked against Qiskit or Quantum Circuit Simulator on a small case before timing.
- **Fair settings:** the 20 November run passed `min_abs_coeff = 0` to PauliPropagation.jl, used its slower `PauliSum` and ran full 1,024-qubit grids. A zero threshold keeps rounding noise around 1e-16 alive, `VectorPauliSum` and the `Performance` module are its documented fast path, and shallow circuits on large systems waste work outside the light cone. Fixing all three cut its time on the 1,024-qubit circuits from 4.2 s to 29–56 ms, and the scoreboard now applies the same treatment to every competitor.
- **Weight conventions differ:** PauliPropagation.jl counts Pauli weight over the whole string, Quantum Circuit Simulator only on wires still open. Only coefficient-only truncation is like for like.
- **Orphaned workers:** on the first Vicena run, stopping the scoreboard left workers from interrupted runs running for hours, loading the machine under later measurements. Workers now run in their own process groups, killed on timeout and when the run stops; check for stray `bench.py worker` processes before timing.
- **Large term counts:** on the deep noisy 6×6 circuit (one-qubit noise, light-cone cut) PauliPropagation.jl ended with about 72 million Pauli strings at 1e-12; this package's largest single cone peaked at 34 thousand terms. The two counts are not the same quantity (a final total over the whole observable against one cone's peak), and why they differ has not been tested. On an Apple M1 Pro its notebook's workspace and `MultiPauliSum` zones took 17.2 s to 8.2 s on one thread and 8.7 s to 1.4 s on eight, and were slower on circuits of a few hundred terms, so the scoreboard times every variant and keeps the fastest.
- **Time limit:** PauliPropagation.jl's rerun has 360 s per process, because three variants share it, where every other tool has 120 s. A variant slower than 120 s can therefore count as finished for it alone.
- **Machine load:** background applications made 8-thread figures unreliable and slowed Julia start-up to tens of seconds. Timings belong on a dedicated machine.

Return to [benchmarks](../README.md).
