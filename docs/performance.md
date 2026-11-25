# Performance evidence and its limits

The architecture cleanup removes demonstrated work and adds alternate representations. It does not establish that Quantum Circuit Simulator is universally fastest. Use matched inputs, noise, requested outputs, tolerances and resource budgets when comparing it.

## Comparison on a Vicena computer

> **Single-run figures are invalid for PauliPropagation.jl**, whose setup time included Julia compilation.
>
> **Being rerun.** This run used PauliPropagation.jl without the workspace reuse and `MultiPauliSum` zones of its advanced-performance notebook, so leads over it, and the counts below, may be too large.

The [scoreboard](../benchmarks/independent/README.md) ran on a Vicena computer (AMD EPYC 7502P, 16-CPU quota, 32 GiB) on 23–24 November 2026, with the layer-fused kernel of commit `1a190eb`, once with one thread per tool and once with sixteen. Every other tool ran as its authors recommend and also on circuits cut to the light cone, and its fastest setup counts. [Both tables and every raw record](../benchmarks/independent/results-2026-11-24-vicena/README.md) keep every refusal and timeout; the README's [How fast it is](../README.md#how-fast-it-is) summarises them.

Of 26 workloads, for one more evaluation of a prepared circuit: 14 at least ten times faster than the fastest other tool, 2 faster by less (4–7×, small circuits where per-call overhead dominates), 3 that only Quantum Circuit Simulator finished (IBM's 127-qubit kicked Ising at general angles, 81–86 ms exact), 4 deep circuits it refused and quimb or Aer solved, and 3 that no tool finished. Sixteen threads left its times almost unchanged, since each of these workloads is one cone group or many tiny ones, while Aer and quimb gained; the counts stayed the same.

Including setup, it leads by ten times or more on only 4–5 workloads: setup, mostly building and preparing the circuit in Python, costs 100–300 ms on the 400- and 1,024-qubit grids, about one PauliPropagation.jl run there. The approximate-mode comparison against the tuned tools has not been rerun.

An earlier comparison of 0.4.0 on an Apple M1 Pro laptop (20 November 2026, [tables](../benchmarks/independent/results-2026-11-20-laptop/v5_tables.md)) ran PauliPropagation.jl with its slower default container, no coefficient threshold and uncut 1,024-qubit grids. Its leads of up to 12,000 times against that tool, and its truncated-mode comparison, are superseded.

## Reviewed source versus cleaned implementation

The recorded revision comparison uses the reviewed November 13 source archive and the cleaned implementation. Eleven distinct vectors are evaluated in isolated persistent workers, alternating method order. Each worker's angle-transfer cache is cleared before every timed public evaluation.

The timer covers parameter validation, fresh-angle binding, native propagation, weighted accumulation and the returned Evaluation. It excludes process startup, imports, installation, circuit construction and preparation.

| Synthetic fixture | Reviewed value + gradient | Cleaned value + gradient | Ratio | Cleaned value only |
| --- | ---: | ---: | ---: | ---: |
| 6 wires, depth 4, 16 parameters | 15.256 ms | 2.606 ms | 5.85× | 2.036 ms |
| 127-wire grid, depth 4, mean local Z | 235.333 ms | 30.987 ms | 7.59× | 25.465 ms |

Maximum old/new discrepancies were 2.22e-15 for values and 2.55e-15 for gradients. These are internal revision comparisons, not external simulator rankings. The 127-wire fixture is a synthetic grid, not the earlier calibrated heavy-hex qualification. Source hashes and every vector/result/timing are retained in [revision evidence](../benchmarks/results/revision_comparison_20261119.json).

Construction and preparation are separately recorded as single observations. They are not repeated setup benchmarks. These single setup observations do not establish a general setup-speed advantage; the repeated-evaluation ratios must not be relabeled as first-request speedups. The recorded revision uses the delivered numerical source, including the graph readout and preparation-fallback corrections; source hashes are retained in the evidence.

Reproduce against a supplied reviewed source directory:

```bash
python benchmarks/compare_revision.py \
  --baseline-root /path/to/reviewed/quantum-circuit-simulator \
  --compiler g++ --output benchmarks/results/revision_local.json
```

The driver builds both native C ABIs with the same compiler into temporary storage. It does not include a second vendored baseline in this release.

## Where the improvements come from

Early product-input contraction prevents resolved wires from expanding further. With the original Python wrappers held constant, the synthetic 127-wire depth-four kernel's retained peak fell from 2,401 terms to 81, and evaluation fell from 195.860 ms to 28.066 ms. At depth two, evaluation changed only from 1.225 ms to 1.148 ms. The gain depends on the workload.

A global-Z independent-relaxation fixture shrinks from 4,096 terms to one at 12 wires. At 16 wires the old kernel exceeds the 50,000-term cap while the new kernel completes with one term. This constructed failure case demonstrates a removable representation bottleneck; a refusal is not an infinite speedup.

Dedicated value-only execution uses scalar coefficient storage and avoids tangent propagation. Closed-form rotation transfers remove the small matrix-trace work that previously dominated fresh-angle rebinding. [Native-core evidence](../benchmarks/results/native_core_comparison_20261119.json) separates those kernel timings from the public API comparison.

## External CPU comparison

The optional [comparison harness](../benchmarks/README.md) evaluates Qulacs, Qiskit Aer, IBM pauli-prop and an independent dense channel-adjoint implementation on the common supported subsets. The recorded run used version 0.2.0a1 on a Linux x86-64 virtual machine (Intel Xeon Platinum 8573C, nine logical processors, one requested thread per worker) on 2026-11-19. It also compares native, graph and local-density Quantum Circuit Simulator paths. Full results, failures, timings, versions and final source hashes are in [CPU comparison evidence](../benchmarks/results/cpu_comparison_2026-11-19.json).

The suite records cold-process cost, preparation, fresh-angle resident evaluation, output accuracy and process RSS separately. Its five warm samples each contain three new vectors; the default has only one cold-process sample per pair. These are descriptive shared-host measurements, not confidence intervals or a production latency study.

The final run requested 84 backend/output pairs: 67 eligible successes, 13 unsupported or budget-excluded pairs, two dense resource-limit rejections, one truncated IBM stress result, and one measurement-worker timeout. There were no accuracy failures. Among successful comparisons, the maximum value error was 9.55e-15 and the maximum gradient-component error was 2.07e-14. Runtime and harness hashes were unchanged during the run and match the delivered source.

| Common case and output | Native | Graph | Comparator |
| --- | ---: | ---: | ---: |
| 4-wire Pauli noise, value | 0.401 ms | 0.441 ms | IBM pauli-prop: 1.254 ms |
| 4-wire T1/T2 and readout, value | 0.333 ms | 0.472 ms | Qulacs: 2.902 ms |
| 4 wires, 8 shared parameters, value + gradient | 2.640 ms | 1.103 ms | Independent dense adjoint: 6.874 ms |

These are warm costs per fresh vector. For the eight-parameter case, graph preparation was 38.195 ms versus 3.218 ms for native, giving an estimated break-even near 23 vectors. The graph's single cold gradient sample was slower: 210.492 ms versus 169.221 ms. Qulacs' two-wire cold value sample was also slightly faster than native (159.750 versus 162.861 ms), but one sample cannot establish a significant small difference. [All recorded tables](../benchmarks/results/README.md) retain these tradeoffs.

Qulacs, Aer and the IBM gradient adapter use exact per-gate parameter shifts, with all occurrences summed into the shared-parameter derivative. The independent dense comparator uses analytic reverse differentiation. Neither a finite-difference baseline nor an unimplemented optimal native noisy-adjoint is presented as the competitor's best possible algorithm. The Aer adapter's unmeasured runtime binding and job-batching options are explicitly documented.

IBM's term cap can truncate, whereas the native Quantum Circuit Simulator cap fails. Any nonzero reported discarded coefficient norm makes that IBM result ineligible for an exact speed comparison, even if its scalar happens to pass the accuracy tolerance.

The Aer timeout applies to a 40-second measurement worker containing validation, many timed evaluations and setup checks. It does not mean one gradient takes 40 seconds.

## Choosing graph or dense execution

Graph preparation stores structural paths. With the C++ graph compiler, every warm graph evaluation measured in the earlier single-thread runs was faster than native, by 1.5× on a 100-qubit single-observable cone and by 46–95× on the deep and wide noisy circuits. On those deep and wide circuits graph preparation costs about one native evaluation, so preparation plus the first evaluation already beats native. On small circuits graph preparation takes 4–11 ms against native's 2–10 ms and is repaid after 4–63 parameter vectors. Larger structural graphs can exceed their budget, and the native backend remains the one for one-off evaluations and truncated mode. The graph remains opt-in, and any fallback is reported. These figures are single-machine measurements on Apple silicon, one thread, warm medians of 11 fresh vectors.

The local dense backend is a bounded capacity alternative with analytic noisy gradients. It is not intended to beat tuned external density simulators on every tiny case. The explicit auto mode uses it after a sparse cap failure, subject to its own cone and array-work limits; it does not predict the fastest method.

## Coverage boundaries

CUDA cuPauliProp, MPO/MPDO and stochastic trajectories have not been timed. PauliPropagation.jl and quimb were timed only in the [independent harness](../benchmarks/independent/README.md), not in the CPU suite, which records their absence and does not infer a ranking. Accurate trajectory comparisons require a common variance/error target. Unitary-only surrogates cannot silently replace deterministic T1/T2 evolution.

Historical benchmark results belong to their exact revisions, machines, circuit families and timing scopes. In particular, the previous restricted 127-wire fresh-circuit constructor, the general package API, and hosted agent/network/UI latency are different measurements. No current result establishes deployment latency or arbitrary 256-wire tractability.

Read the [comparison methodology](comparison-methodology.md) for complete contracts and primary upstream references. Return to [README](../README.md).
