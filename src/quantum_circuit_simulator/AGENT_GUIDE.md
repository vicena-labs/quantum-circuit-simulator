# Quantum Circuit Simulator: guide for agents

Quantum Circuit Simulator, by Vicena, simulates noisy quantum circuits on realistic chips, develops algorithms with exact gradients, and designs chips. It computes expectation values and gradients in up to 4,096 shared parameters; simulates states and samples bitstrings; builds Hamiltonians; optimises VQE and QAOA; extrapolates to zero noise; estimates shot noise; and models, compiles onto and designs chips.

Its main solver is the **Pauli-cone method**: each observable is propagated backward through its light cone as a sum of Paulis, so cost depends on the cone, not the register. A local observable on a 1,000-qubit shallow circuit is cheap, while a deep, noiseless, highly entangling circuit may be refused or need truncation. The router falls back to state simulation where that fits, and every result names the method used.

## When to use it

- **Use it for:** expectation values and gradients of noisy or shallow circuits, local observables on many qubits, VQE and QAOA with exact gradients, Hamiltonian dynamics, zero-noise extrapolation, shot budgets, sampling (up to 28 qubits, any size for Clifford circuits), and chip comparison and design.
- **Use something else for:**
  - mid-circuit measurement, resets or classical control;
  - initial states other than |0...0>;
  - deep noiseless circuits beyond 28 qubits where `plan` shows wide cones and exact answers are required. A tensor-network simulator may handle those;
  - error-correction decoding, which is not built in yet.

## Five commands

Every command prints one JSON document. Exit code 0 is success, 2 an invalid request or failed evaluation, and 124 a timeout. On failure, read `error.hint`: it names the next thing to try.

```bash
quantum-circuit-simulator capabilities                                        # gates, noise, limits, backends
quantum-circuit-simulator plan circuit.qasm --observable "Z0*Z1"              # dry run: is exact feasible?
quantum-circuit-simulator run circuit.qasm --observable "0.5*Z0*Z1 - X3"      # value and gradients
quantum-circuit-simulator run circuit.qasm --observable "Z5" --min-coefficient 1e-4   # approximate, with a bound
quantum-circuit-simulator schema                                              # JSON request format
```

Observables are written with wire indices, such as `"0.5*Z0*Z1 - X3"`, or as Qiskit labels, such as `"0.5*IXZ"`, where the rightmost character is wire 0. OpenQASM 2 is read directly; OpenQASM 3 needs Qiskit. Add noise to an OpenQASM circuit with `--noise-1q 0.001 --noise-2q 0.01 --readout 0.02,0.03`.

## Chips

`quantum-circuit-simulator devices` lists the chip catalogue:
- superconducting heavy-hex (including the 127-qubit IBM Eagle-style layout), square-grid and line chips;
- all-to-all trapped ions;
- neutral-atom arrays;
- D-Wave-style Chimera, Pegasus and Zephyr annealer graphs.

`quantum-circuit-simulator device square-grid --param rows=6 --param columns=6` describes one chip, and `--edges` prints its full JSON description, which is also the format for a custom chip.

Add `--device heavy-hex-127` (or `--device my_chip.json`) to `run` or `plan` to apply the chip's gate and readout errors and check its connectivity. The response's `device.uncoupled_two_qubit_operations` counts gates on qubit pairs the chip does not couple. Catalogue noise values are typical figures for each technology, not calibration data; scale them for noise studies in Python with `chip.scaled_noise(2.0)`.

To run a circuit on a chip, add `--compile`:

```bash
quantum-circuit-simulator compile circuit.qasm --device heavy-hex-127                       # SWAPs, depth, duration, layouts
quantum-circuit-simulator run circuit.qasm --observable "Z0*Z5" --device heavy-hex-127 --compile --noise-scale 2
```

Compiling does four things:
1. Places the circuit's qubits on the chip.
2. Inserts SWAPs so every two-qubit gate acts on coupled qubits. `--router sabre` uses Qiskit and usually needs fewer.
3. Schedules gates with the chip's durations.
4. Simulates gate errors (a SWAP counts as three two-qubit gates), T1/T2 decay while qubits idle, and readout error on the measured Paulis.

The compile report's `layout` is `exact` when every interacting pair sits on a coupler, so no SWAPs are needed; otherwise it is `greedy`. Observables keep the circuit's own qubit numbers; they are moved to the final layout for you. `--noise-scale` multiplies every error and decoherence rate, so 0 is noise-free and repeating a run at several scales is a noise sweep. In Python: `from quantum_circuit_simulator.compiler import compile_circuit`, then `compile_circuit(circuit, chip).expectation("Z0*Z1")`.

## Sweeps

Compare chips, noise levels and sizes in one command; each row gives the value, the ideal noise-free value, the deviation, the method used, and the routing cost:

```bash
quantum-circuit-simulator sweep --family qaoa-ring --sizes 8,12,16 --family-param p=2 \
    --device heavy-hex-127 --device "all-to-all-ions:n=16" --noise-scales 0,0.5,1,2 --format markdown
quantum-circuit-simulator sweep circuit.qasm --observable "Z0*Z5" --device my_chip.json --noise-scales 1,2
```

Built-in families, each with its own observable:
- `ghz`;
- `hardware-efficient`;
- `qaoa-ring` (MaxCut);
- `ising-trotter`.

The solver is chosen per point: exact sparse propagation, then the state vector (noise-free, up to 24 qubits) or density matrix (noisy, up to 10 qubits), then exact dense for cones of at most 6 wires, then truncation at 1e-4 and 1e-5. A truncated row has `exact: false`, an `error_bound` with `bound_informative`, and the `convergence_change` between the two tolerances. A chip too small for a size gives a row with `error` instead of stopping the sweep. In Python: `from quantum_circuit_simulator.experiments import sweep, solve`.

## Designing chips

Build a new chip from a catalogue layout, then compare it with existing chips:

```bash
quantum-circuit-simulator design --base "heavy-hex:rows=3,row_length=7" --modules 4 --links ring --link-error 0.02 \
    --remove-qubits 5,17 --t1 3e-4 --t2 2e-4 --time-1q 3e-8 --time-2q 2e-7 --control-error-2q 2e-3 \
    --name modular-hh --out modular.json
quantum-circuit-simulator compare modular.json heavy-hex-127 "square-grid:rows=6,columns=6" --family qaoa-ring --sizes 8,12 \
    --format markdown
```

Design options:
- `--modules` tiles copies of the base and joins them in a line, ring or all-to-all. Links can have their own `--link-error`, usually worse than on-module couplers.
- `--remove-qubits` removes defective qubits and renumbers the rest.
- `--add-couplers` and `--remove-couplers` edit connectivity.
- Giving all of `--t1`, `--t2`, `--time-1q` and `--time-2q` derives gate errors from the exact coherence limit, plus `--control-error-1q/2q` for errors coherence does not capture.

More options:
- `--qubit-spread 0.2` gives each qubit its own one-qubit error, readout, T1 and T2, spread log-normally around the chip values as on a real chip.
- `--spectator-error` and `--zz-hz` add crosstalk. Neighbours of every two-qubit gate take a depolarizing error. Every coupled pair of active qubits picks up a ZZ conditional phase of 2π·ζ·t for as long as both are active, idle or not, except during their own two-qubit gate.
- Coherent ZZ multiplies the work of exact simulation. Sweeps and `solve` then switch to truncation and report its convergence; expect seconds rather than milliseconds per point.
- `--expand-to N` grows a chip to at least N qubits with the same architecture: heavy-hex stays heavy-hex, grids keep their aspect ratio, and modular chips add modules.

In a sweep, `--device fit:heavy-hex-127` grows the chip to fit each size.

The result is an ordinary device file: use it anywhere a device goes (`--device modular.json`). `compare` ranks chips by mean deviation from the ideal value on a benchmark family, with mean SWAPs, two-qubit gates and duration. In Python, `quantum_circuit_simulator.design` has the same operations.

## Real chips, benchmark circuits, CAD and annealing

- **Real chips:** `quantum-circuit-simulator devices` lists generated layouts (`kind: layout`) and real chips (`kind: real`): 65 IBM chips such as `ibm_fez`, `ibm_torino`, `ibm_sherbrooke` and `ibm_kyiv`, and Google's `google_willow_pink`, `google_weber` and `google_rainbow`. Each carries its measured per-qubit and per-coupler errors, T1, T2, readout, gate times and coordinates from a dated calibration snapshot. Use them anywhere a device goes: `--device ibm_fez --compile`. Say "the ibm_fez calibration of <snapshot date>", not "ibm_fez today".
- **Benchmark circuits:** `quantum-circuit-simulator circuits` lists families and the MQT Bench and Supermarq library; `circuits mqt/qft --size 16 --out qft.qasm` writes one. Sweeps accept library names (`--family mqt/grover --sizes 4,8`), with mean Z as the default observable. `kicked-ising` is IBM's 127-qubit utility experiment.
- **CAD:** `quantum-circuit-simulator cad ibm_fez --out fez.gds` (or `.dxf`, `.svg`) draws any chip. It is a placement layout, not a fabrication mask.
- **Annealing:** `quantum-circuit-simulator anneal --model maxcut --lattice regular --size 60,3 --device dwave-advantage` embeds and samples an Ising problem; `--problem p.json` takes any Ising or QUBO. Report `best_energy`, `ground_energy` when present, and `chain_break_fraction`. The samplers model annealing (path-integral Monte Carlo by default); they are not the D-Wave device. Needs the `dwave` extra.

## Hamiltonians and algorithms

```bash
quantum-circuit-simulator hamiltonian --model heisenberg --lattice grid --size 3,3 --model-param j=1 --out h.json
quantum-circuit-simulator optimize --hamiltonian h.json --layers 3 --steps 300 --progress          # VQE
quantum-circuit-simulator optimize --algorithm qaoa --model maxcut --lattice regular --size 12,3 --depth 2
quantum-circuit-simulator optimize --hamiltonian h.json --device heavy-hex-127 --method spsa       # on a noisy chip
quantum-circuit-simulator zne circuit.qasm --observable "Z0*Z1" --device heavy-hex-127 --fit richardson
quantum-circuit-simulator shot-noise circuit.qasm --observable h.json --shots 10000
```

- Models: `transverse-field-ising` (j, h), `heisenberg` (j, h), `maxcut` (weights), `fermi-hubbard` (t, u, mu; spin-up wires first). Lattices: `chain` (n), `grid` (rows, columns), `regular` (n, degree). `hamiltonian` also reports the exact ground energy up to 14 qubits.
- `optimize` returns the best value, the parameters, the exact optimum as `reference` when the system is small, and a `trace` with the value and gradient norm of every iteration. QAOA adds the approximation ratio and the best sampled bitstring. Methods: `adam` (default), `gd`, `lbfgs`, and `spsa`, which uses values only, as a hardware run would.
- `zne` amplifies every device error by each `--scales` factor, fits, and reports the estimate with the ideal and unmitigated values, so you can see how much mitigation recovers.
- `shot-noise` gives the standard error a hardware run with that many shots would have, measuring qubit-wise commuting groups, with exact variances.
- In Python: `from quantum_circuit_simulator import hamiltonians as hm, algorithms as al`. `hm.trotter(h, "t", steps=8)` makes time a differentiable parameter; `al.Objective(circuit, h, device=chip)` gives value and gradient at any parameter vector from one reused plan.

## States and samples

```bash
quantum-circuit-simulator sample circuit.qasm --shots 4000 --device heavy-hex-127 --seed 1   # noisy bitstrings, wire 0 rightmost
quantum-circuit-simulator statevector circuit.qasm                                          # amplitudes, top bitstrings
```

Sampling is exact from the density matrix up to 10 qubits, then uses one quantum trajectory per shot up to 28 qubits; larger Clifford circuits go to Stim. `exact_distribution` says whether the samples follow the noisy distribution exactly. On a chip, only the qubits the circuit uses are simulated.

## Results, progress and checkpoints

Every result has `schema` (`quantum-circuit-simulator-<kind>/1`) and `provenance` (versions, request SHA-256, seed, time); quote the provenance when you report a number. `compile` adds a `noise_budget` per qubit and per coupler, and `--schedule` lists every gate with its start time and duration. `--progress` streams one JSON line per completed unit on standard error. `sweep --checkpoint run.jsonl` records finished points, so a rerun after a timeout resumes instead of starting over.

## Recipe

1. **Plan.** `quantum-circuit-simulator plan`: if `exact_guaranteed` is true, exact evaluation cannot exceed the term cap.
2. **Run exactly.** `quantum-circuit-simulator run`. A `ConeOverflow` error means the exact expansion exceeded the cap (default 1,000,000 terms per cone; `--max-terms` raises it, at about 0.5 KB of memory per term). It never returns a wrong answer.
3. **Truncate if needed.** Rerun with `--min-coefficient 1e-4 --converge 0.01 --timeout 900`. The threshold tightens half a decade per level until two successive values agree within 0.01, the term cap stops it or the time runs out, and `convergence` reports every level. Each result's `error_bound` is a rigorous worst case on |exact − returned|. For deep circuits it is often no tighter than the trivial bound, and `bound_informative` is then false: the change between levels is the only evidence, and it is empirical, not a guarantee. Never stop at the first overflow: raise `--max-terms`, tighten the threshold, and report what converged.
4. **Sweep parameters.** For many parameter vectors of one circuit, use `--backend graph`. It compiles once and is much faster per vector; put up to 128 vectors in one JSON request's `parameter_vectors`.
5. **Small cones.** When every cone has at most 6 wires, `--backend dense` is exact regardless of depth.

## Reading a result

- `results[i].value`: the expectation of the whole weighted observable at parameter vector i.
- `results[i].gradient`: d value / d parameter, one entry per shared parameter.
- `results[i].error_bound` and `gradient_error_bound`: present only for truncated runs; `bound_informative` is false when the bound says nothing beyond |exact| ≤ ‖O‖₁.
- `convergence` (with `--converge`): each level's threshold, values, bounds, peak terms and seconds; `change` between the last two; `converged`; and why it `stopped`.
- `backend`: the method actually used. `diagnostics.fallback_reason` explains any fallback.
- `diagnostics`: cone sizes and term counts. `numerics`: the cutoffs in force.

## Python

```python
from quantum_circuit_simulator.interop import expectation
r = expectation(open("circuit.qasm").read(), "0.5*Z0*Z1 - X3")         # or a Qiskit QuantumCircuit
r = expectation(qiskit_circuit, "Z0", values={"theta": 0.3}, gradients=True)
print(r.value, r.gradient, r.error_bound)
```

As a drop-in Qiskit primitive: `from quantum_circuit_simulator.estimator import Estimator`, then use it wherever a Qiskit `EstimatorV2` is expected. To simulate on a chip, use `from quantum_circuit_simulator import devices`, then `chip = devices.get("all-to-all-ions", n=20)`, and pass `device=chip` to `expectation` or `Estimator`.

On a real chip, in a notebook:

```python
from quantum_circuit_simulator import devices
from quantum_circuit_simulator.compiler import compile_circuit
from quantum_circuit_simulator.statevector import sample_compiled
chip = devices.get("ibm_fez")                              # calibration date: chip.parameters["hardware"]
compiled = compile_circuit(open("circuit.qasm").read(), chip, layout="trivial", noise_scale=1.0)
print(compiled.report["swaps"], compiled.report["two_qubit_gates"])
print(compiled.expectation("Z0*Z1").value)                 # exact noisy expectation
counts = sample_compiled(compiled, shots=4096, seed=1)["counts"]
```

Truncating with a convergence ladder, from Python:

```python
from quantum_circuit_simulator.experiments import ladder, refine
results, report = refine(circuit, terms, [None], ladder(1e-4, levels=5), tolerance=0.01, seconds=900)
print(results[0].value, report["change"], report["converged"], report["stopped"])
```

For repeated evaluation, convert once and reuse the plan:

```python
from quantum_circuit_simulator.interop import convert, parse_observable
c = convert(qiskit_circuit)                       # Qiskit Parameters become shared parameters
plan = c.circuit.prepare(parse_observable("Z0*Z1"), backend="graph")
for values in sweep:
    print(plan.evaluate(c.shifts(values)).value)
```

## Verify before reporting

- **Cross-check small cases:** on a reduced instance (at most about 20 qubits), `solve(circuit, terms, cross_check=True)` compares Pauli propagation with state simulation and reports `cross_check.agrees`; or compare against `--backend python` or `dense`.
- **Report truncation fully:** for truncated results, state the threshold and the convergence check, not only the bound. When `bound_informative` is false, say the value rests on convergence alone.
- **Mind the noise convention:** noise probabilities are total non-identity Pauli error probabilities. Qiskit Aer's depolarizing parameter λ equals p·4^k/(4^k−1) for k wires.
