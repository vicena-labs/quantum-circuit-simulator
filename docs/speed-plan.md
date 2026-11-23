# Plan: ten times faster than the best tuned alternative on every benchmark

Status: proposal, started 23 November 2026. Owner: Vicena Labs.

## Goal and the rule that governs it

Quantum Circuit Simulator should be at least ten times faster than the fastest other tool on every workload of the [independent comparison](../benchmarks/independent/README.md), with each competitor run the way its own authors recommend. The goal is reached by making the package faster, never by choosing settings, workloads or reporting that flatter it:

- Every competitor runs with its documented fast path, a coefficient threshold just above rounding noise, and circuits cut to the light cone at the smallest system that holds it. Its figure is the fastest of all those configurations.
- The workload list is fixed before an improvement is measured. A workload may be added, never removed because we lose on it.
- Every loss, refusal and timeout stays in the published tables.
- Before figures are published, the setup goes to the compared tools' authors, and their corrections are applied and recorded.

This rule exists because the first comparison broke it by accident. PauliPropagation.jl ran with its slow default container, no coefficient threshold and full 1,024-qubit grids, and the published lead of up to 12,000× fell to a fraction of that once the package ran as intended.

## Lessons from the first comparison

- **Light cones were most of the lead.** On shallow circuits with local observables, most of the lead came from finding each observable's light cone automatically. An expert gets much of that back by cutting the circuit by hand, so that part is a convenience and is reported as one, not as kernel speed.
- **Exact against thresholded is not like for like.** Exact mode keeps terms around 1e-15 that a 1e-12 threshold drops. At the same threshold, the graph backend evaluates the 127-qubit circuit at π/2 in 0.17 ms against 104 ms; in exact mode it lost. Every row states both tools' thresholds.
- **One number hides four regimes.** Repeated evaluation, single evaluation, approximate and multi-core runs can rank tools differently. The graph backend wins repeated evaluations by hundreds of times and can lose a single one because of its compilation.
- **Small circuits measure overhead.** Below about 0.1 ms the Python call costs more than the simulation.
- **Error bounds are loose.** At π/2 the reported bound was 1.8e-4 against a true error of 7e-10. A tighter bound would let users choose cheaper thresholds safely.

## The scoreboard

One command reruns the comparison and prints, for each workload, our time, the best tuned competitor, and the ratio, flagging every ratio below 10×. It reports four regimes, because each is a different user need:

| Regime | Who needs it | What is timed |
| --- | --- | --- |
| Exact, repeated evaluation | optimisers and parameter sweeps | warm evaluation after any one-off preparation |
| Exact, single evaluation | one question about one circuit | preparation, compilation and first evaluation together |
| Approximate, matched error | deep circuits nobody solves exactly | time to reach the same error against an exact reference |
| All cores | Vicena's remote computers, up to 16 cores | each tool's best thread count |

An improvement counts only when the full test suite passes, values agree with an independent exact reference to 1e-12, gradients agree with finite differences, and no scoreboard row gets slower.

## Where we stand

Measured on 23 November 2026 against PauliPropagation.jl 0.9.2 tuned as above; the [scoreboard](../benchmarks/independent/README.md) reproduces them. Already ten times ahead: noisy circuits from 10 to 1,024 qubits, noiseless circuits of four layers and the 127-qubit kicked Ising at general angles. Behind or short of 10×:

| Gap | Cause |
| --- | --- |
| IBM 127-qubit circuit at angles 0 and π/2: 6× and 3.7× slower | At Clifford angles nothing branches, but exact mode carries rounding-noise terms (cos π/2 ≈ 6e-17) through every later gate |
| Noiseless 12–20 qubits, 2 layers: 3–5× ahead | At 0.03 ms the cost is the Python call, not the simulation |
| 100-qubit centre observable, 4 layers: 2.6× ahead | The competitor gains as much from the light cone as we do; our per-term cost has to fall |
| Single evaluations | Graph compilation (5–150 ms) is charged to one evaluation |
| 6–8 layers and the 20-step kicked Ising | Exact mode refuses at its term cap; approximate mode has not yet been compared with the tuned competitors |

## Improvements, in order

Each step is measured on the scoreboard before the next starts.

1. **Layer-fused kernel.** Merge noise into the adjacent gates and process each layer of gates in one pass over the terms. Built on the `kernel-fusion` branch: 79–87× faster on the hard 127-qubit cases and 7–26× at the Clifford angles, with values identical to 1e-14.
2. **Exact zeros stay zero.** Drop terms whose coefficient is below rounding error relative to the observable's norm when only the value is needed, and evaluate rotations at Clifford angles as the single Pauli map they are. Gradients keep the terms, whose derivatives are not zero.
3. **One native call for many evaluations.** Evaluate a batch of parameter vectors in C++ without returning to Python between them, which removes the fixed per-call cost that dominates small circuits.
4. **Cheaper preparation.** Cache compiled graphs by circuit structure and chip, and cut the Python conversion (about 0.3 s on the 127-qubit circuit, mostly `deepcopy` during import from QASM), so single evaluations win too.
5. **All cores.** Propagate independent light cones, and independent parameter vectors, on separate cores; compare against every competitor at its best thread count.
6. **Approximate mode against tuned competitors.** Rerun the truncated comparison at matched error, and size the term cap from available memory so deep circuits run on the remote computers instead of being refused.
7. **Deep noiseless circuits.** Where tensor networks (quimb) win, Pauli propagation alone may not reach 10×. The candidate is contracting each light cone as a tensor network when it is cheaper than propagating it. This is research, and the tables keep the loss until it is solved.

Return to [performance](performance.md).
