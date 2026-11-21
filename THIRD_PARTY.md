# Dependencies and third-party material

The standalone project includes the Pauli reference, native wrapper, indexed-cone wrapper, reusable graph, local density backend and C++ kernel. These sources use the project Apache-2.0 license. Preserve any contributor or third-party notices when extending them. A source inventory is not independent ownership or patent clearance.

NumPy is a runtime dependency, installed separately. Setuptools is a build dependency; pytest is a test dependency. They are not vendored in the source. The numerical extension uses the platform Python API and C++ standard library. Binary distribution review must account for any runtime libraries bundled into future platform wheels; the local wheel is not a portable manylinux release.

The optional experimental fresh-runner sources link to the system zlib library when compiled. They are not included in the installed Python wheel and are not compiled by the default package build.

## Bundled data

`src/quantum_circuit_simulator/data/` holds data converted from openly licensed packages by `tools/import_external.py`. No source code from these packages is copied; their data is converted into this package's own formats.

| File | Content | Source | License |
| --- | --- | --- | --- |
| `devices.json.gz` | IBM calibration snapshots: coupling maps, per-qubit T1, T2, readout and gate errors, per-coupler errors, gate durations, qubit frequencies, anharmonicities and coordinates | qiskit-ibm-runtime `fake_provider` (IBM); coordinates also from Qiskit's gate maps | Apache-2.0 |
| `devices.json.gz` | Google device specifications and median calibrations (Rainbow, Weber, Willow Pink) | cirq-google (Google) | Apache-2.0 |
| `circuits.json.gz` | Benchmark circuits, re-expressed in OpenQASM 2 with this package's gate set | MQT Bench (Technical University of Munich) | MIT |
| `circuits.json.gz` | Benchmark circuits | Supermarq (Infleqtion) | Apache-2.0 |

Each device and circuit records its source and version in its own `source` field. The snapshots are historical calibrations, not live device data.

The optional D-Wave extra installs dimod, dwave-networkx, dwave-samplers, dwave-system and minorminer (Apache-2.0) separately; the annealing module calls them and copies none of their code. The `dwave-advantage` and `dwave-advantage2` devices are built from dwave-networkx graphs with D-Wave's published h, J and annealing-time ranges.

No copied Qiskit, Aer, Stim, Qulacs or Ocean source, old CI artifacts, private application source, credentials, infrastructure settings or private Git history are intentionally included. KQCircuits (GPL-3.0) is not used, so its chip designs are not included.

The optional comparison tools install pauli-prop, Qulacs, Qiskit and Qiskit Aer separately. They are not runtime dependencies or vendored implementations. The recorded versions are pauli-prop 0.2.1 (Apache-2.0), Qulacs 0.6.14 (MIT), Qiskit 2.6.2 (Apache-2.0) and Aer 0.18.4 (Apache-2.0); consult each installed distribution's own notices. The adapters are original API integration code.

The complete Apache-2.0 text is in [LICENSE](LICENSE); the upstream source is https://www.apache.org/licenses/LICENSE-2.0.txt.
