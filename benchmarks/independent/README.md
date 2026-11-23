# Independent comparison harness

These scripts compare Quantum Circuit Simulator with Qiskit Aer, Qulacs, PennyLane Lightning, IBM pauli-prop, quimb and PauliPropagation.jl on identical circuits, observables and parameter vectors. Each run is a fresh process with a time limit, and refusals and timeouts are recorded as results. [Preliminary results](results-2026-11-20-laptop/v5_tables.md) from an Apple M1 Pro laptop are summarised in [measured performance](../../docs/performance.md#preliminary-comparison-of-040), beside the earlier [CPU comparison](../results/README.md) on a Linux virtual machine.

| File | Purpose |
| --- | --- |
| `bench.py` | Workloads and one adapter per Python tool. `run` times exact evaluation; `export` writes the workloads and timed vectors as JSON for Julia |
| `pp_bench.jl`, `run_pp.py` | PauliPropagation.jl: exact, truncated (`min_abs_coeff`, `max_weight`) or its reusable surrogate (`PP_MODE=surrogate`) |
| `trunc_bench.py` | Quantum Circuit Simulator truncated mode (`BENCH_BACKEND=native` or `graph`), converting an absolute coefficient threshold to Quantum Circuit Simulator's relative one |
| `summarize_v5.py` | Builds the tables from the raw JSONL files |

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
- **Default truncation:** PauliPropagation.jl truncates by default (`min_abs_coeff = 1e-10`); exact runs must pass 0.
- **Weight conventions differ:** PauliPropagation.jl counts Pauli weight over the whole string, Quantum Circuit Simulator only on wires still open. Only coefficient-only truncation is like for like.
- **Machine load:** background applications made 8-thread figures unreliable and slowed Julia start-up to tens of seconds. Timings belong on a dedicated machine.

Return to [benchmarks](../README.md).
