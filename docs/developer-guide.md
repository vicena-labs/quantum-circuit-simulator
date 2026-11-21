# Building quantum algorithms and exploring connectivity

Quantum Circuit Simulator's core is a **focused expectation-value and analytic-gradient engine**, the Pauli-cone solver; state simulation, sampling, algorithms and chip design are built around it. This page is for developers who want to adapt a parameterized circuit, compare connectivity, or extend the solver. Start with the [README](../README.md) and [architecture](architecture.md).

## Try it

Install the source package, including test dependencies:

```bash
python -m pip install '.[test]'
python -m pytest -q
python examples/architecture_playground.py --topology line --qubits 5 --layers 2
python examples/architecture_playground.py --topology grid --rows 2 --cols 3 --layers 2
python examples/architecture_playground.py --topology custom --file examples/custom_topology.json --layers 2
python examples/qaoa_maxcut.py --qubits 3 --gamma .35 --beta .25
python examples/variational_energy.py --steps 6
python examples/compare_backends.py
```

The playground defines entangling connectivity and a synthetic depolarizing channel. It **does not** contain a real device's coupling map, calibrated error rates, crosstalk, scheduling/timing, pulse model, placement or routing. A custom graph is useful to test *algorithmic sensitivity to an architecture*, not to predict fidelity on a deployed chip without a validated hardware model.

### Custom architecture

Supply the undirected coupling graph in [this example JSON](../examples/custom_topology.json):

```json
{"n_qubits": 5, "couplers": [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]}
```

`Topology.line(n)`, `Topology.grid(rows, cols)` and `Topology(n, couplers)` validate couplers. `Topology.matchings()` provides a deterministic greedy partition into layers of non-overlapping edges. **It is not an optimal hardware scheduler.** `topology.add_cz_layer(circuit, depolarizing_probability=...)` adds CZ gates and optional synthetic noise. The core `Circuit.gate` method does not enforce the topology, so call the topology helpers or `Topology.require_edge` for every two-qubit gate in architecture-aware experiments. ECR directionality and calibrated gate durations are not represented by this undirected helper.

### Build and test a new algorithm

The [MaxCut QAOA example](../examples/qaoa_maxcut.py) shows a simple variational circuit on a graph, expressed exclusively in supported primitive gates. It returns the expected cut and two analytic gradients; those can feed any optimizer that accepts a function and gradient. The [variational energy example](../examples/variational_energy.py) evaluates a toy three-wire Hamiltonian and takes a few explicit gradient-descent steps. Note that the package currently has no built-in production optimizer or a general Hamiltonian compiler. It is a numerical building block, not a turn-key algorithm suite.

An algorithm developer generally:

1. Defines a graph and an ansatz using supported one-/two-qubit gates.
2. Supplies up to 4,096 shared angle variables (`Circuit(..., n_parameters=...)`) and one or more weighted Pauli terms as the observable.
3. Prepares the circuit once, then evaluates different finite parameter shifts with `evaluate`/`evaluate_many` or `evaluate_values`.
4. Checks small cases against the `dense` backend or an independent reference.
5. Profiles full preparation plus evaluation; verifies any claimed speedup at equal accuracy and resources.

A single parameter shift is added to **every gate** that references the parameter index. It is *not* an absolute angle. Only the zero product initial state is available through the public API, but input states can be prepared from that state using supported gates.

## Which earlier optimizations are actually included?

| Capability | Public API? | Scope |
| --- | --- | --- |
| Sparse backward Heisenberg Pauli transfer | Yes (`native`) | Noisy expectation and shared gradients |
| Reverse causal-cone reduction and shared indexed stages | Yes (`native`) | Grouped local observables, reusable planning |
| Early initial-state contraction and specialized local transfer rows | Yes (`native`) | Avoids unnecessary expansion in eligible circuits |
| Fast analytic rotation transfer and separate value-only path | Yes (`native`) | Faster rebinding and no derivative work for values |
| Reusable structural execution graph + reverse differentiation | Yes (`graph`) | Experimental; preparation and resource costs charged |
| Small-cone analytic dense method and explicit overflow fallback | Yes (`dense`, `auto`) | Bounded by cone memory/wires; not full-density for large chips |
| 127-wire fresh calibrated-shape C++ constructor | **Experimental source only** | Restricted schema and gate schedule, not normal Circuit API |
| Full-state simulation, probabilities, shot sampling, mid-circuit feedback | **No** | Use other independently qualified engines |
| Generic QAOA/VQE optimization and molecular Hamiltonian builders | **No** | Examples demonstrate concepts but do not replace full solvers |
| Arbitrary gate/kernel plugins or automatic chip routing | **No** | Extending gate sets requires numerical implementation and tests |
| 4096-wire chip, local Pauli expectations | **Yes, narrow cones** | Native causal-cone key <=256 active wires, no full-state promise |
| Deep 4096-wire chip with global entanglement | **Not guaranteed** | Active cone, term and stage budgets still apply |

The current package is complete **for its documented specialized method**, but it does not package unrelated historical statevector/adjoint/Stim/Qulacs solver implementations or every experimental quantum simulation speedup. Do not transfer a timing claim from those other workloads to this package.

## How to extend the engine safely

For a new gate, noise channel, or parameterization:

1. Define its mathematical action and adjoint in the stated Pauli order `I, X, Z, Y`, including any nonunital identity terms.
2. Implement the local transfer and derivative with consistent wire order in the reference and native paths.
3. Ensure the dependency analysis includes the new operation; do not discard paths merely because a coefficient is zero at one angle.
4. Check parameter sharing, near-zero angles, nonunital noise, and numerical derivatives against independent small density-matrix calculations.
5. Update the versioned CLI schema, docs, caps, and regression suite if the operation should become public.
6. Benchmark cold and prepared modes against compatible external methods rather than a deliberately slower baseline.

Do not silently treat arbitrary hardware channels as depolarizing noise or claim an exact physical calibration from a synthetic graph. Where the representation is ineffective, report the limit or explicitly switch to a qualified backend.

## Integrating with research agents

The portable contract is a source-installed Python package or versioned JSON CLI. A coding agent can inspect `pyproject.toml`, run the included tests, invoke the `doctor` command and exercise the examples. Do not copy a development monorepo, run the experimental C++ runner on untrusted inputs, or use a solver output without validating the `backend` and error status. [Security boundaries](../.github/SECURITY.md) and [benchmark rules](comparison-methodology.md) apply.

Return to [README](../README.md).

## Many-qubit research and backend extensibility (0.3.0a1)

For a 4,096-qubit shallow test with 128 sampled local Z expectations:

```bash
python examples/large_qubit_grid.py --rows 64 --columns 64 --observables 128 --vertical
```

For 16,384 logical qubits (same narrow-cone limitation):

```bash
python examples/large_qubit_grid.py --rows 128 --columns 128 --observables 32
```

The maximum logical width is 16,384, whereas the bundled native Pauli key handles **at most 256 wires in each backwards structural cone**. Strongly mixing deep global circuits may fail this limit. All stage/term caps remain enforceable. See [wide-register and backend extension guide](extensibility.md) for the full contract and examples of registering a custom tensor or other future solver.
