# Architecture

The Pauli-cone solver at the core of Quantum Circuit Simulator computes a weighted sum of noisy Pauli expectations and optional analytic derivatives with respect to shared gate angles. The public input state is the zero product state. Parameter vectors are additive shifts from the base circuit; every occurrence contributes to its shared derivative. Noise parameters remain fixed.

## Preparation and ownership

Preparation validates all observable terms, removes identity labels, combines identical Pauli words with `math.fsum`, and removes exactly cancelled terms. The input limit of 1,024 terms is checked before canonicalization. Extreme finite weights whose combined sum overflows float64 are rejected; accumulation is not arbitrary precision.

Each prepared object owns a deep circuit snapshot. A nonblocking lock rejects simultaneous evaluations of that object. Separate plans can serve parallel callers. The default native plan performs one reverse dependency pass for all unique observables using bitsets. Observables are then grouped greedily, in request order, while their combined cone fits the key width that the widest single cone needs (64 wires, or the next multiple of 64). A group is propagated jointly: the circuit is linear in the observable, so this returns the same sum while terms the observables share are merged and propagated once. Joint propagation of the benchmark workloads took about half the time of separate cones. If a group exceeds the term cap, it is split back into single cones for that and later evaluations, so grouping never refuses a request that separate cones would answer.

Each group retains ordered indices into one shared native stage array and its own wire numbering, so rebinding changes each local transfer once and any register width works. All groups run in one native call, optionally on several threads; results are summed in group order, so they do not depend on the thread count. Rotation angles are rebound by vectorised writes into a NumPy view of the stage array.

## Early contraction into the input state

Immediately after backward propagation crosses a wire's earliest forward operation, no remaining operation can touch that wire. Its Pauli factor can be contracted into the known product input.

For zero input, I and Z contribute one; X and Y contribute zero. Internal plus-input tests instead retain I and X. The surviving wire label is cleared before insertion into the next term map. Equal remaining keys then combine before the live term budget is checked. This applies to both values and derivative coefficients, including affine relaxation, where the adjoint of Z contains Z and I.

Idle wires are contracted during observable initialization. Their asymmetric readout factor is evaluated directly, avoiding an exponential expansion of otherwise idle Z readouts.

This optimization uses product-input factorization and operation order. It does not discard a nonzero input overlap. It reduces an independent affine-relaxation product fixture from exponential intermediate support to one live term. Entangled interiors can still grow exponentially.

## Native sparse execution

The scalar is `Tr[rho0 E1†(... EL†(O))]`. Sparse terms use X/Z bitmasks. The one-wire Pauli order is I, X, Z, Y, encoded as `x + 2*z`; the first wire of a two-wire stage is the low-order pair of bits.

For a local transfer entry T and derivative dT, propagation accumulates `c_new += T*c` and `g_new += T*g + dT*c`. The derivative contribution is added to the stage's shared parameter. A zero transfer entry with a nonzero derivative remains in gradient mode, including at an exact zero angle.

Local transfer rows are specialized lazily once per stage per evaluation. Clifford permutations and diagonal noise commonly need one transition per occupied row. Rows are rebuilt after binding, so changing angles cannot freeze obsolete support. RX, RY and RZ transfers and derivatives use closed-form sine/cosine matrices, avoiding repeated small complex matrix traces.

Terms live in one flat open-addressing table keyed by the X and Z masks, using one to four 64-bit words according to the group's wire count, with one coefficient per output: one for values, the parameter count plus one for gradients. Each stage updates the table in place. Terms with identity on the stage's wires are untouched, because every channel's adjoint maps the identity to itself. Terms whose row maps to itself alone, such as a Z under RZ or a Pauli under depolarizing noise, are rescaled in place. Only terms the stage moves or branches are removed, transformed and merged back. The roundoff filter then checks only the terms changed by that stage. The original C symbols and structure layouts remain compatible. The experimental runner includes the maintained kernel instead of retaining an outdated duplicate.

The extension is compiled at source installation, loaded through a small CPython shim, and called through ctypes. Import and evaluation do not invoke a compiler.

## Reusable graph and dense alternatives

The explicit `graph` backend compiles possible transitions into a bounded arithmetic graph. Parameter support is obtained analytically, not from nonzero entries at one angle. It groups and renumbers cones as the native solver does, records each group's propagation in C++ as edges between scalar nodes, and evaluates one pass over the edges, differentiating in reverse without inverting noisy channels. Wide registers are supported with the same per-cone limit. See [execution graphs](execution-graph.md) for preparation and memory tradeoffs.

The explicit `dense` backend uses local density matrices for small cones. It converts supported local channels to density superoperators, propagates states forward, and applies channel adjoints backward for analytic shared-angle gradients. Only states immediately before parameterized stages are saved. Observable terms with identical stage sets and wire sets share a density evolution and combined measurement operator. Unrelated cones are not merged.

Dense preparation defaults to at most six wires per cone and 64 MiB of estimated density-array work. The estimate includes local operators, saved states and working temporaries; it excludes circuit snapshots, rebound transfer matrices, global caches and Python bookkeeping. It is not a process RSS limit. Preparation budgets for gradients even when a later call requests only values. The public qubit limit can be raised to ten, subject to the byte budget.

`auto` is an explicit capacity fallback: native sparse execution is attempted first, and a `ConeOverflow` during preparation or evaluation triggers dense preparation. It is not a prediction of the fastest backend. All cones must fit the dense limits. Once selected, the dense plan is retained for subsequent vectors.

Graph preparation-budget failures fall back to native, recording the actual backend and reason. Other errors remain failures. Explicit native, Python and dense requests never silently switch representation.

## Numerical and failure contracts

Native and Python sparse execution retain absolute transfer-entry filtering at 5e-15. Coefficient filtering is relative: a term is dropped when its largest coefficient is at most 1e-16 times the l1 norm of the observable being propagated, so an observable of any scale gives the same relative accuracy. These are numerical filters, not an approximation budget or a global error certificate. Graph preparation retains fixed transfer filtering but preserves small coefficients, including during readout initialization. Dense evolution uses the available float64 local transfers and complex128 density arrays without sparse coefficient filtering.

`plan.numerics` reports the active policy. All backends report that no global floating-point error bound is certified.

## Truncated mode

`Truncation(min_coefficient, max_weight)` makes the native solver discard terms during propagation. After each stage, a changed term is discarded when its largest coefficient is below `min_coefficient` times the observable's l1 norm, or when it acts on more than `max_weight` wires that are still open. Value-only evaluations compare the value coefficient alone, so gradient evaluations keep at least as many terms and can return a slightly different value.

The reported bound is rigorous. A term `c*P` discarded after stage t would have contributed `c*Tr(rho_t P)` to the exact value, where `rho_t` is the true state before stage t, and `|Tr(rho_t P)| <= 1`. The value bound is therefore the sum of `|c|` over every discarded term, roundoff drops included. For gradients, a discarded term also loses `d Tr(rho_t P)/d theta_j`, bounded by the sum of `2*||G||` over parameter j's stages before t, where G is the gate generator: 1 for RX, RY and RZ, 2 for fSim. Both bounds are clipped to the trivial bound from `|exact| <= l1 norm`. The bound is a worst case; in the qualification runs the measured error was between 0.1% and 36% of it, so a convergence check over several tightening thresholds remains the practical accuracy test. It is empirical: on the 20-step kicked Ising circuit at θ = 0.4, three levels agreed at 0.836 while converged published simulations give 0.850. A truncated graph chooses its discarded terms during preparation instead; see [execution graphs](execution-graph.md#truncated-graphs). Truncation is refused by the Python and dense backends; `auto` with truncation runs the native solver and falls back to the exact dense backend, with a zero bound, only on a term-cap failure at 256 wires or fewer. Different representations can round differently; qualification uses independent tests and stated tolerances rather than bitwise identity.

`threads` parallelises across independent cone groups only. A circuit whose observables share one cone group, such as a magnetisation over a densely connected register, runs on one thread.

Term caps, graph node/edge limits, and dense array budgets fail visibly. A failed batch is never emitted as successful partial output. Structural graph support can exceed a budget even when one numeric binding is cheap. Diagnostics distinguish preparation refusal from an accuracy failure.

The public contract does not include arbitrary Kraus input, thermal populations, mid-circuit feedback, samples, noise-parameter gradients or entangled initial states.

## Validation and other methods

The suite covers analytic products, independent unitary/Kraus density evolution, shared parameters, zero-angle rebinding, reversed wire order, value-only calls and budget failures. The optional [comparison harness](../benchmarks/README.md) uses external CPU implementations and an independent dense adjoint-gradient oracle.

GPU Pauli propagation, Julia surrogates, MPO/MPDO and trajectories remain relevant alternatives. Different channel, accuracy and hardware contracts are recorded in the [comparison methodology](comparison-methodology.md). These implementations do not establish that every alternative representation has been exhausted.

Return to [README](../README.md).

## Wide-register extension (0.3.0a1)

The logical register may contain up to 16,384 qubits, while a native key holds at most four 64-bit words per Pauli mask. For the `native` backend, the cone groups above are renumbered to consecutive local wires, so wide registers use the same single native call and one shared stage array. Other backends keep the per-cone planner: a shared reverse pass determines each observable's cone, structurally identical cones share preparation, and each cone becomes its own prepared plan. Untouched product-state factors are contracted analytically in both. The per-cone 256-wire ceiling, term cap, stage budgets and independent expected-value accuracy checks remain essential. See [extensibility](extensibility.md).
