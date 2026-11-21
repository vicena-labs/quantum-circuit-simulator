# Quantum Circuit Simulator

**A quantum circuit simulator by Vicena, built for AI agents and the scientists who work with them.** It answers the question that decides where quantum computing effort goes: *will this algorithm work on that hardware, at this size and noise level, and what would make it work?* In one package you can:
- simulate noisy circuits on realistic chips;
- develop algorithms with exact gradients;
- design new chips and compare them;
- get every answer as JSON that says how it was computed and whether it is exact.

**Release candidate 0.5.0rc1, by Mehdi Farza, licensed under Apache-2.0.** It is not yet published on PyPI; install it from a checkout with `pip install .`. Once released, `pip install quantum-circuit-simulator` will install it. The repository contains the source, tests, examples and benchmark evidence. See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY.md), [how to cite it](CITATION.cff) and the [related work](docs/related-work.md) it builds on.

## What it does

- **Simulates circuits on chips.** It compiles a circuit onto a superconducting, trapped-ion, neutral-atom or custom chip, then simulates it with gate errors, idle relaxation, crosstalk and readout error.
- **Develops algorithms.** It builds Hamiltonians, Trotter and QAOA circuits, runs VQE and QAOA with exact gradients in up to 4,096 shared parameters, applies zero-noise extrapolation, and estimates the shot budget a hardware run needs.
- **Designs hardware.** It builds chips (modules, defects, physics-derived errors, crosstalk), grows them to larger sizes with the same architecture, and ranks designs on real workloads.
- **Returns states and samples.** It gives amplitudes, density matrices and measured bitstrings with noise, up to 28 qubits, and any size for Clifford circuits.
- **Ships real chips and standard circuits.** It includes 65 IBM chips and 3 Google chips from their measured calibrations, D-Wave Advantage-style annealers, and over a hundred benchmark circuits from MQT Bench and Supermarq.
- **Draws chips as CAD files.** It writes any chip, real or designed, as a GDSII, DXF or SVG layout.

## How it solves: Pauli cones

The main solver propagates each observable backward through the circuit, in the Heisenberg picture, as a sum of Pauli operators. It touches only the observable's backward **light cone**, so cost depends on the cone, not the register: a local observable on a 1,000-qubit shallow circuit is cheap. The solver is written in C++, groups observables whose cones overlap, and differentiates every shared parameter. It can compile the propagation once into a reusable graph, so each new parameter vector costs one pass forward and one pass back. An optional truncation answers deeper circuits approximately, with a rigorous bound on what it discarded. We call this the Pauli-cone method.

A router chooses the solver for every question and says which one it used:
1. exact Pauli cones;
2. the state vector, or the density matrix for noisy circuits;
3. exact local density matrices;
4. truncated Pauli cones with a convergence check.

It can also cross-check one method against another. Pauli propagation itself returns expectation values; the state simulators add states and samples.

Supported operations include H, SX, X/Y/Z, RX/RY/RZ, CZ, ECR, fSim (theta derivative only), any fixed one- or two-qubit unitary, independent Pauli errors, one/two-qubit depolarization, zero-temperature T1/T2 relaxation, and asymmetric Z readout. Circuits address **1–16,384 logical qubits**; the Pauli-cone solver uses **at most 256 active wires per observable light cone**, and larger circuits are remapped into local cones. In exact mode, a cone too broad for its term budget is refused instead of answered partially. Nothing needs a hosted account, API key, internet connection at runtime, or proprietary application installation.

## Install and run locally

Use Python 3.10 or later. NumPy is the only runtime Python dependency. Binary wheels are published for Linux, macOS and Windows; on other platforms, installation compiles the extension and needs a C++17 compiler and Python development headers.

```bash
python -m pip install quantum-circuit-simulator      # or, from a checkout: python -m pip install .
quantum-circuit-simulator doctor
quantum-circuit-simulator example > problem.json
quantum-circuit-simulator run problem.json
```

## Use it from OpenQASM, Qiskit or an agent

Existing circuits run without rewriting. OpenQASM 2 is read with no dependencies; OpenQASM 3 and Qiskit circuits need `pip install "quantum-circuit-simulator[qiskit]"`. Every gate is decomposed exactly into the native set, and Qiskit Parameters become shared parameters, so one prepared plan serves a whole parameter sweep.

```bash
quantum-circuit-simulator plan circuit.qasm --observable "0.5*Z0*Z1 - X3"   # dry run: is exact feasible?
quantum-circuit-simulator run circuit.qasm --observable "0.5*Z0*Z1 - X3" --noise-2q 0.01
```

```python
from quantum_circuit_simulator.interop import expectation
from quantum_circuit_simulator.estimator import Estimator          # a Qiskit Estimator V2

print(expectation(qiskit_circuit, "Z0*Z1", values={"theta": 0.3}, gradients=True))
evs = Estimator().run([(qiskit_circuit, observables, parameter_values)]).result()[0].data.evs
```

## Algorithms: Hamiltonians, optimisation, mitigation and shot noise

`quantum_circuit_simulator.hamiltonians` builds Pauli-sum Hamiltonians (transverse-field Ising, Heisenberg/XXZ, Max-Cut and Fermi-Hubbard under Jordan–Wigner, on chains, grids and random regular graphs) or loads them from JSON, strings, Qiskit `SparsePauliOp` or OpenFermion `QubitOperator`. `trotter` and `qaoa` turn them into circuits whose angles are shared, weighted parameters, so time or the QAOA angles are differentiable. `quantum_circuit_simulator.algorithms` adds an `Objective` that reuses one plan per Hamiltonian, optimisers (Adam, gradient descent, SPSA, L-BFGS) that keep a full trace, `vqe`, `qaoa`, zero-noise extrapolation (`zne`) and the statistical error of a hardware run with a given number of shots (`shot_noise`).

```python
from quantum_circuit_simulator import algorithms as al, devices, hamiltonians as hm

h = hm.heisenberg(hm.chain(8))
result = al.vqe(h, al.hardware_efficient(8, layers=3), steps=300)      # exact gradients, 64 parameters
print(result.value, result.reference)                                     # versus exact diagonalisation
noisy = al.vqe(h, al.hardware_efficient(8, layers=3), device=devices.get("heavy-hex-127"))
cut = al.qaoa(hm.maxcut(hm.random_regular(12, 3)), depth=2)               # approximation ratio, best bitstring
```

```bash
quantum-circuit-simulator optimize --model heisenberg --lattice grid --size 3,3 --device heavy-hex-127 --progress
quantum-circuit-simulator zne circuit.qasm --observable "Z0*Z1" --device square-grid --fit richardson
quantum-circuit-simulator shot-noise circuit.qasm --observable h.json --shots 10000
```

With more than 16 parameters, gradients come from the graph backend's single reverse pass (the native solver computes them 16 per pass). On a 12-qubit, 120-parameter ansatz with a Heisenberg Hamiltonian the graph returns the value and all 120 derivatives in 5 ms, against 0.85 s for the native passes (an Apple M-series laptop, one thread).

## States and sampling

`quantum_circuit_simulator.statevector` simulates states directly: amplitudes of noise-free circuits up to 28 qubits, density matrices up to 12 qubits, and measured bitstrings with noise and readout error, sampled exactly from the density matrix or from one quantum trajectory per shot. Clifford circuits of any size are sampled through Stim (`pip install "quantum-circuit-simulator[stim]"`). A circuit compiled onto a large chip is sampled on the qubits it uses only. The solver router cross-checks Pauli propagation against these simulators on request (`solve(..., cross_check=True)`).

```bash
quantum-circuit-simulator sample circuit.qasm --shots 4000 --device heavy-hex-127 --seed 1
quantum-circuit-simulator statevector circuit.qasm
```

## Output contracts

Every command prints one JSON document tagged `schema: quantum-circuit-simulator-<kind>/1` with a `provenance` record (package and library versions, platform, SHA-256 of the canonical request, seed, UTC time), so a result can be reproduced and cited. Compiling reports a noise budget per qubit and per coupler (gate, idle, spectator and readout errors and ZZ phase) and, with `--schedule`, every gate's physical qubits, start time and duration; chips carry qubit coordinates in lattice units. Long tasks stream JSON-line progress events on standard error with `--progress`, and sweeps resume from a `--checkpoint` file. Fields are only added within a schema version.

## Real chips, benchmark circuits, CAD and annealing

**Real chips.** `devices.get("ibm_fez")` returns IBM's 156-qubit Heron r2 chip, and `devices.get("google_willow_pink")` Google's 105-qubit Willow chip. Each is built from a published calibration snapshot: every qubit's T1, T2, readout and gate error, every coupler's two-qubit error, gate durations, IBM's qubit frequencies and anharmonicities, and the chip's real coordinates. The package includes 65 IBM chips (Heron, Eagle, Falcon, Hummingbird and older) and Google's Rainbow, Weber and Willow Pink. `quantum-circuit-simulator devices` lists them with their snapshot dates. The compiler gives each gate T1/T2 relaxation for its duration and depolarizing noise for the rest of its measured error, as hardware errors arise. On IBM snapshots it agrees with Qiskit Aer's noise model to within 0.01, and it adds the relaxation of idle qubits, which Aer's backend model leaves out.

**Benchmark circuits.** `circuits.library()` lists over a hundred circuits from MQT Bench (QFT, Grover, QPE, amplitude estimation, arithmetic, QAOA, VQE ansätze, quantum walks and more) and Supermarq, at sizes up to 127 qubits, in this package's gate set. `circuits.load("mqt/qft", 16)` returns one as OpenQASM, and sweeps take library names like families. `family("kicked-ising")` is IBM's 127-qubit utility experiment.

**CAD.** `cad.write(chip, "chip.gds")` writes a chip as GDSII (also DXF and SVG): Xmon-style transmon crosses, coupler traces and readout pads at the chip's coordinates, or sites for ions, atoms and annealers, on separate layers in micrometres. It is a placement layout for drawing, documenting and starting a design, not a fabrication-ready mask; full electromagnetic layout is the job of tools such as Qiskit Metal or KQCircuits.

**Annealing.** `annealing.anneal(problem, devices.get("dwave-advantage"))` embeds an Ising or QUBO problem on a D-Wave Advantage-style Pegasus chip (or Advantage2 Zephyr, Chimera, Pegasus, Zephyr) with minorminer, rescales it to the chip's h and J ranges, can add control errors, and samples it with simulated quantum annealing (path-integral Monte Carlo), simulated annealing or tabu search. It reports the best energy, the exact ground energy and its probability on small problems, chain lengths and chain breaks. These are models of an annealer, not the device. Needs `pip install "quantum-circuit-simulator[dwave]"`.

```bash
quantum-circuit-simulator devices                                     # layouts and real chips
quantum-circuit-simulator run circuit.qasm --observable "Z0*Z1" --device ibm_fez --compile
quantum-circuit-simulator circuits --suite "MQT Bench" --search qft    # the benchmark library
quantum-circuit-simulator sweep --family mqt/qft --sizes 8,16 --device ibm_torino --device google_willow_pink
quantum-circuit-simulator cad ibm_fez --out fez.gds
quantum-circuit-simulator anneal --model maxcut --lattice regular --size 60,3 --device dwave-advantage
```

## Chips

`quantum_circuit_simulator.devices` describes chips: connectivity, native gates, gate and readout errors, coherence times and gate durations, in one JSON format that also serves for custom chips. The catalogue covers:
- superconducting heavy-hex (including the 127-qubit IBM Eagle-style layout), square-grid and line chips;
- all-to-all trapped ions;
- neutral-atom arrays;
- D-Wave-style Chimera, Pegasus and Zephyr annealer graphs (the last two need `pip install "quantum-circuit-simulator[dwave]"`).

Catalogue noise values are typical figures for each technology, not calibration data for a particular chip.

```python
from quantum_circuit_simulator import devices
from quantum_circuit_simulator.interop import expectation

chip = devices.get("heavy-hex-127")
for factor in (0.5, 1, 2):                       # a noise sweep on one chip
    print(factor, expectation(circuit_qasm, "Z0*Z1", device=chip.scaled_noise(factor)).value)
```

```bash
quantum-circuit-simulator devices                                   # the catalogue
quantum-circuit-simulator device square-grid --param rows=6 --param columns=6
quantum-circuit-simulator run circuit.qasm --observable "Z0*Z1" --device all-to-all-ions --device-param n=20
```

With `--device`, the response reports any two-qubit operations on qubits the chip does not couple. To run on the chip, add `--compile`.

`quantum_circuit_simulator.compiler` routes a circuit onto a chip:
- it places the qubits, inserting SWAPs where needed (built in, or Qiskit's SABRE with `router="sabre"`);
- it schedules gates with the chip's durations;
- it simulates gate errors, T1/T2 decay of idle qubits, and readout error on the measured Paulis.

Observables keep the circuit's qubit numbers, and `noise_scale` multiplies every error and decoherence rate for noise sweeps. Without noise, a compiled circuit reproduces the original expectation exactly.

`quantum_circuit_simulator.design` builds new chips. It can:
- tile modules joined by links with their own error rates;
- remove defective qubits and edit couplers;
- derive gate errors from T1, T2 and gate durations (the exact coherence limit, plus control error).

It can also give every qubit its own errors and coherence times, add crosstalk (spectator errors and ZZ phases around two-qubit gates), and grow any chip to more qubits with the same architecture. A sweep can grow the chip for each size with `--device fit:NAME`. `compare` ranks designs against catalogue chips on benchmark circuits. Both are also commands (`quantum-circuit-simulator design`, `quantum-circuit-simulator compare`), and a designed chip is an ordinary device file.

Sweeps put it together: compile and solve over chips, noise scales and sizes, with the ideal value and routing cost beside every point. The solver is chosen automatically and every row says whether it is exact.

```bash
quantum-circuit-simulator sweep --family qaoa-ring --sizes 8,12,16 --device heavy-hex-127 \
    --device "all-to-all-ions:n=16" --noise-scales 0,1,2 --format markdown
```

```python
from quantum_circuit_simulator.compiler import compile_circuit
compiled = compile_circuit(circuit_qasm, devices.get("heavy-hex-127"), noise_scale=2)
print(compiled.report["swaps"], compiled.report["duration_s"], compiled.expectation("Z0*Z1").value)
```

The command line is built to be driven by AI agents in a shell:
- every command prints one JSON document;
- every error carries a `hint` naming the next step;
- `quantum-circuit-simulator capabilities` and `quantum-circuit-simulator schema` describe the tool;
- `quantum-circuit-simulator guide` prints a usage guide written for agents.

[AGENTS.md](AGENTS.md) points agents to it.

No compiler runs when importing the package or evaluating a circuit. `--backend python` selects the reference implementation explicitly. The default native backend fails visibly if its term budget is exceeded.

```python
from quantum_circuit_simulator import Circuit

circuit = Circuit(2, n_parameters=1)
circuit.gate("RY", [0], theta=0.7, param=0)
circuit.gate("H", [1]).gate("CZ", [0, 1])
circuit.depolarize([0, 1], probability=0.01)
circuit.relax(0, duration=2e-7, t1=70e-6, t2=90e-6)
circuit.readout(0, p01=0.02, p10=0.01)

plan = circuit.prepare([(1.0, {0: "Z"})], backend="native")
for result in plan.evaluate_many([[0.0], [0.1], [-0.1]]):
    print(result.value, result.gradient)

# Skip derivative propagation when only values are needed.
print(plan.evaluate_value([0.1]))
print(plan.evaluate_values([[0.0], [0.1], [-0.1]]))
```

### Truncated mode

```python
from quantum_circuit_simulator import Truncation

approx = circuit.prepare([(1.0, {0: "Z"})], truncation=Truncation(min_coefficient=1e-4, max_weight=6))
result = approx.evaluate([0.1])
print(result.value, result.error_bound, result.gradient_error_bound)
```

`min_coefficient` discards a term once its coefficient falls below that fraction of the observable's l1 norm; gradient evaluations keep a term while any derivative coefficient is above it. `max_weight` discards terms acting on more than that many wires still open, after early contraction into the input state. `error_bound` is rigorous: a term `c*P` discarded at any point would have contributed `c*Tr(rho P)`, and `|Tr(rho P)| <= 1`. It is a worst case and is often much larger than the actual error, so check convergence by tightening the tolerance until the value stops changing. Truncation runs on `native`, `graph` and `auto`; the other backends refuse it. The native solver chooses the discarded terms afresh at every evaluation. A truncated `graph` chooses them when it compiles, at the angles bound then, and keeps evaluating that smaller graph; its bound is still computed at each evaluation's own angles, so it stays rigorous and grows when the parameters move to where the discarded terms matter. When the bound exceeds `reprepare_factor` (default 2) times its value at compilation, the graph compiles itself again at the current angles; `reprepare_factor=None` keeps the first graph.

`threads=N` in `prepare` evaluates independent cone groups in parallel. Results do not depend on the thread count.

Each vector contains **additive shifts from the original gate angles**, not absolute replacements. Every occurrence of parameter 0 receives the same shift. The returned derivative is with respect to that shared shift. Reusing a plan never reuses a previous numerical answer. Preparing a plan takes an owned snapshot; later edits to the original circuit do not change it.

## Choosing an execution method

| Backend | Use | Limits and behavior |
| --- | --- | --- |
| `native` (default) | Sparse local observables and shared gradients | C++ execution; hard live-term cap; early contraction into product inputs |
| `python` | Inspectable sparse reference | Same transfer conventions and numerical filters; slower Python maps |
| `graph` | Repeated parameter vectors on manageable structural graphs | Compiles the propagation once, then each evaluation is one pass over stored arithmetic with reverse differentiation; records native fallback if graph preparation exceeds its limits |
| `dense` | Small cones with broad Pauli support | Analytic channel-adjoint gradients; defaults to six cone wires and a 64 MiB density-array work budget |
| `auto` | Explicit sparse-to-dense capacity fallback | Falls back on a sparse cap failure only; does not predict the fastest method |

```python
graph = circuit.prepare([(1.0, {0: "Z"})], backend="graph",
                        graph_max_nodes=2_000_000, graph_max_edges=10_000_000)
result = graph.evaluate([0.1], gradients=False)  # gradient is an empty tuple
print(result.backend, graph.diagnostics, graph.numerics)
```

`evaluate_many(..., gradients=False)` returns `Evaluation` objects with empty gradients. The convenience `evaluate_values()` returns only floats. Duplicate observables are combined and exact cancellations are removed before cone construction. Detailed budget and fallback contracts are in [architecture](docs/architecture.md).

## Conventions and numerical boundaries

- `depolarize(..., probability=p)` means total probability of a uniformly chosen nonidentity Pauli error. It is not Aer’s depolarizing lambda; for k wires, `lambda = p * 4**k / (4**k - 1)`.
- Relaxation duration, T1 and T2 must use the same units; `0 < T2 <= 2*T1`. The model assumes zero temperature.
- Readout uses `p01=P(measure 1 | prepared 0)` and `p10=P(measure 0 | prepared 1)`. X/Y observables on a wire with this classical Z-readout model are rejected instead of inventing a measurement model.
- Sparse native/reference arrays are float64, with absolute transfer filtering at `5e-15` and coefficient filtering at `1e-16` times the observable's l1 norm, so results scale linearly with observable weights. Graphs do not prune coefficients; dense arrays are complex128. Structural cone reduction and early product-input contraction are exact algebraically, but no backend certifies an accumulated floating-point error bound. Truncated plans bound the truncation error, not floating-point roundoff. Read `plan.numerics` for the active policy.
- Exceeding a declared budget fails the request unless an explicitly selected fallback succeeds within its own limits. No incomplete sum is returned as success. A prepared object cannot be evaluated concurrently; use separate plans for parallel callers.

The command-line interface validates a versioned JSON request, limits input to 4 MiB and executes one worker with a wall deadline (`--timeout`, default 60 seconds). The deadline begins after request input has been read. This is not a secure sandbox or a hard RAM budget for untrusted services; see [security boundaries](SECURITY.md).

## Algorithm and architecture playground

The source includes [a developer guide](docs/developer-guide.md), examples for line/grid/custom connectivity, a small MaxCut QAOA circuit, a variational energy/gradient-descent example, and within-package backend comparisons. These examples let researchers build new ansätze on *supplied coupling graphs*; the graphs are not calibrated chips or hardware transpilation. The package remains an optimized **noisy Pauli observable/gradient backend**, not a complete full-state/sampling algorithm suite. See the [capability and speedup inventory](docs/developer-guide.md#which-earlier-optimizations-are-actually-included).

```bash
python examples/architecture_playground.py --topology grid --rows 2 --cols 2
python examples/architecture_playground.py --topology custom --file examples/custom_topology.json
python examples/qaoa_maxcut.py --qubits 3
python examples/variational_energy.py --steps 6
python examples/compare_backends.py
```

## Documentation and development

[Architecture](docs/architecture.md) explains the process and representation choices. [Execution graphs](docs/execution-graph.md) covers reusable topology and reverse differentiation. [JSON interface](docs/json-interface.md) specifies operations and error handling. [Measured performance](docs/performance.md) separates new measurements from historical prototype results. [Comparison methodology](docs/comparison-methodology.md) and the [benchmark harness](benchmarks/README.md) describe independent baselines and missing comparisons. [Release checklist](docs/release-checklist.md) covers source publication.

[Wide-register qualification](docs/large-register-validation.md) documents the 4,096/16,384-qubit local tests and limitations. [Validation](docs/validation.md) records the installed-package checks, numerical coverage, source scanning and remaining qualification limits.

```bash
python -m pip install '.[test]'
python -m pytest -q
python tools/check_source_hygiene.py  # Strict source-archive check.
# In a Git checkout, use: python tools/check_source_hygiene.py --tracked-only
python tools/check_release.py
```

Wheels are built by the [wheel workflow](.github/workflows/wheels.yml) and uploaded to PyPI only from a published GitHub release. The [contribution guide](CONTRIBUTING.md) and [third-party dependency review](THIRD_PARTY.md) explain what is included. Static source checks are safeguards, not proof of ownership or absence of unknown secrets.

The restricted fresh-circuit native executables are retained as **experimental sources**, not exposed as the general package CLI. Their narrower input contract and missing production isolation should not be confused with the public Circuit API.

## Related work and acknowledgements

Quantum Circuit Simulator implements Heisenberg-picture Pauli propagation, a method developed across a substantial literature, and reuses ideas from several open tools. [Related work](docs/related-work.md) lists the papers and software it builds on, what each contributed, and how Quantum Circuit Simulator differs.

## Many-qubit circuits and custom backends

Quantum Circuit Simulator **0.3.0a1** introduced a wide logical-register adapter: 1–16,384 indexed wires, connected line/grid/user-supplied topologies, and native evaluation of local observables even when their wire indices exceed 255. The backward dependency graph for all observables is traversed once. Observables whose combined cone is small enough are grouped, each group is renumbered to its own wires, and all groups run in one native call over one shared bound copy of the circuit. Independent idle wires contract analytically.

```bash
python examples/large_qubit_grid.py --rows 64 --columns 64 --observables 128 --vertical
python examples/large_qubit_grid.py --rows 128 --columns 128 --observables 32
```

These demos use a parameterized RY layer on every physical wire, independent horizontal CZ pairs, an optional `--vertical` CZ matching, depolarizing noise, and sampled Z expectations. **They do not simulate arbitrary strongly entangled 4,096- or 16,384-qubit states.** The public interface keeps the 16-shared-parameter and 1,024-observable limits. Each cone is bounded to 256 physical wires, with stage and Pauli-term budgets; an oversized cone is refused. The `max_cone_qubits` parameter sets a lower refusal threshold at every register width.

Developers can register a trusted backend for tensor networks, trajectory sampling, stabilizers, or another representation without modifying the native solver. `register_backend('ext-example', factory)` expects a factory returning a plan with `bind()` and `evaluate()`; see [the extensibility guide](docs/extensibility.md). External backends must provide their own numerical validation, and they execute in process, not a sandbox. A custom backend receives the full wide register rather than being forced into the 256-wire cone adapter.

