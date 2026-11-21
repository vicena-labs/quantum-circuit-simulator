# Agents

**Using Quantum Circuit Simulator:** run `quantum-circuit-simulator guide`, or read [the agent guide](src/quantum_circuit_simulator/AGENT_GUIDE.md). It covers when to use the package, the five commands, a decision recipe, how to read results and how to verify them. `quantum-circuit-simulator capabilities` and `quantum-circuit-simulator schema` describe the supported gates, limits and request format as JSON.

**Changing Quantum Circuit Simulator:**
- Read the [README](README.md), [architecture](docs/architecture.md) and [contribution guide](.github/CONTRIBUTING.md).
- Run `python -m pytest -q` after building the extension with `python setup.py build_ext --inplace`.
- Keep the command line JSON-only, with a `hint` on every error.
- Keep the agent guide in step with any change to commands or outputs.
