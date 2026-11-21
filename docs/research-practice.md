# Research practice and the package

What researchers in quantum simulation and algorithm design do with simulators, which of it the package covers, and what is still missing. Surveyed in November 2026 from the publications below and the tools researchers use (Qiskit and its add-ons, PennyLane, OpenFermion, Mitiq, Qualtran, the Azure Quantum Resource Estimator). Back to the [README](../README.md).

## Covered, with the exact check behind each

| Practice | Module | Publications | Checked against |
|---|---|---|---|
| Molecular Hamiltonians, active spaces, Jordan–Wigner, Bravyi–Kitaev and parity mappings, Z2 tapering, UCCSD | `chemistry` | [Seeley, Richard and Love](https://arxiv.org/abs/1208.5986); [Bravyi et al., tapering](https://arxiv.org/abs/1701.08213) | PySCF FCI and CASCI to 1e-8; identical spectra across mappings; UCCSD reaches FCI on H2 |
| Propagated noise absorption | `mitigation.pna` | [qiskit-addon-pna](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-pna) | the noise-free value to 1e-10 when the noise model is exact |
| Probabilistic error cancellation | `mitigation.pec`, `pec_overhead` | [van den Berg et al.](https://arxiv.org/abs/2201.09866) | the depolarizing closed form for gamma; unbiased within its standard error |
| Readout mitigation, TREX | `mitigation.readout_correct`, `mitigated_counts` | [van den Berg et al., TREX](https://arxiv.org/abs/2012.09738) | exact recovery under the readout model |
| Commuting measurement groups, shot budgets | `measurement.group`, `shot_budget` | [Gokhale et al.](https://arxiv.org/abs/1907.13623); [Crawford et al.](https://arxiv.org/abs/1908.06942) | every group commutes and is diagonalised; analytic variance against sampling |
| Classical shadows | `measurement.shadows` | [Huang, Kueng and Preskill](https://arxiv.org/abs/2002.08953) | exact values within their standard errors |
| Higher-order product formulas, qDRIFT, Trotter error bounds, multi-product formulas | `dynamics` | [Childs et al.](https://arxiv.org/abs/1912.08854); [Campbell](https://arxiv.org/abs/1811.08017); [Carrera Vazquez et al.](https://arxiv.org/abs/2207.07272) | `expm`: error slopes of order 2 and 4, bounds above the true error |
| Fault-tolerant resource estimates | `resources` | [Beverland et al.](https://arxiv.org/abs/2211.07629); [Ross and Selinger](https://arxiv.org/abs/1403.2975) | hand-worked estimates in the tests; not compared with Azure's tool |
| QAOA fixed angles, transfer and classical baselines | `baselines`, `algorithms.qaoa(init=...)` | [Wurtz and Lykov](https://arxiv.org/abs/2107.00677); [Zhou et al.](https://arxiv.org/abs/1812.01041) | the p=1 closed form exactly; p=2 table values on a girth-6 graph |
| Operator backpropagation and truncation with bounds | the Pauli-cone solver | [Fuller et al.](https://arxiv.org/abs/2502.01897); [Begušić, Gray and Chan](https://arxiv.org/abs/2308.05077) | IBM's 5-step exact utility values to 1e-14 |

## Still missing

- Benchmark protocols on the chip models: randomized benchmarking, layer fidelity, cross-entropy benchmarking and quantum volume ([McKay et al.](https://arxiv.org/abs/2311.05933), [Cross et al.](https://arxiv.org/abs/1811.12926)).
- ADAPT-VQE ([Grimsley et al.](https://arxiv.org/abs/1812.11173)) and sample-based quantum diagonalization ([Robledo-Moreno et al.](https://arxiv.org/abs/2405.05068)); both build on `chemistry`.
- Early fault-tolerant phase estimation such as QCELS ([Ding and Lin](https://arxiv.org/abs/2211.11973)), which needs mid-circuit measurement.
- Learned sparse Pauli–Lindblad noise models; `pna` and `pec` currently take the chip model as the learned noise.
- Command-line commands for the new modules; they are Python-only for now.

## Known limits of what is covered

- Goemans–Williamson runs only with cvxpy installed, and its test is skipped without it.
- The p=3 fixed angles are checked only approximately, at 0.7909 against the paper's 0.7924, because the girth-8 test graph needs 30 qubits.
- Tapering uses Z-type symmetries only, so the Hartree–Fock state fixes the sector.
- The order-2 Trotter bound bounds each nested commutator by its coefficients' l1 norm: valid, but looser than the operator norm.
- On tetron chips, the resource model assumes a syndrome round of four measurements.
