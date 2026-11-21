"""Quantum annealing on D-Wave-style chips: Ising and QUBO problems, embedding, and simulated annealers.

    from quantum_circuit_simulator import annealing, devices, hamiltonians as hm
    problem = annealing.Problem.from_hamiltonian(hm.maxcut(hm.random_regular(40, 3)))
    result = annealing.anneal(problem, devices.get("dwave-advantage"), num_reads=200, seed=1)
    result["best_energy"], result["embedding"]["max_chain_length"], result["chain_break_fraction"]

An annealer does not run gate circuits: it minimises an Ising energy
E(s) = sum h_i s_i + sum J_ij s_i s_j over spins s = +1/-1 (+1 is |0>). On a chip
each problem variable becomes a chain of physical qubits found by minor
embedding (minorminer), held together by a chain coupling, rescaled to the
chip's h and J ranges, optionally perturbed by integrated control errors
(Gaussian noise on every h and J), sampled, and read back by majority vote over
each chain. Samplers (dwave-samplers): "sqa", path-integral Monte Carlo of
transverse-field annealing (simulated quantum annealing, the default);
"sa", classical simulated annealing; "tabu"; and "exact", every state, for up to
24 variables. These are models of an annealer, not the device: real anneals
also see the device's own schedule, temperature and noise.

Needs the D-Wave Ocean packages: pip install "quantum-circuit-simulator[dwave]".
"""
from __future__ import annotations

import math
import time

import numpy as np

# Published ranges of D-Wave's Advantage-generation solvers (dimensionless h and J, times in us).
ANNEALER_PROPERTIES = {"h_range": [-4.0, 4.0], "j_range": [-1.0, 1.0], "extended_j_range": [-2.0, 1.0],
                       "annealing_time_range_us": [0.5, 2000.0], "default_annealing_time_us": 20.0}


def _ocean():
    try:
        import dimod
        import dwave.embedding
        import dwave.samplers
        import minorminer
    except ImportError as exc:
        raise ImportError('annealing needs the D-Wave Ocean packages: pip install "quantum-circuit-simulator[dwave]"') from exc
    return dimod, dwave.embedding, dwave.samplers, minorminer


class Problem:
    """An Ising or QUBO problem: a dimod BinaryQuadraticModel with JSON and Hamiltonian conversions."""

    def __init__(self, bqm):
        self.bqm = bqm

    @classmethod
    def from_ising(cls, h, J, offset=0.0):
        dimod = _ocean()[0]
        return cls(dimod.BinaryQuadraticModel.from_ising(dict(h), {tuple(k): v for k, v in dict(J).items()}, offset))

    @classmethod
    def from_qubo(cls, Q, offset=0.0):
        dimod = _ocean()[0]
        return cls(dimod.BinaryQuadraticModel.from_qubo({tuple(k): v for k, v in dict(Q).items()}, offset))

    @classmethod
    def from_hamiltonian(cls, hamiltonian):
        """An Ising problem from a Z-only Hamiltonian of one- and two-body terms (spin s_i = <Z_i>)."""
        h, J, offset = {}, {}, 0.0
        for c, p in hamiltonian.terms:
            if set(p.values()) - {"Z"} or len(p) > 2:
                raise ValueError("an annealer takes Z-only terms on one or two qubits")
            if not p:
                offset += c
            elif len(p) == 1:
                (q,) = p
                h[q] = h.get(q, 0.0) + c
            else:
                a, b = sorted(p)
                J[(a, b)] = J.get((a, b), 0.0) + c
        problem = cls.from_ising(h, J, offset)
        for q in range(hamiltonian.n_qubits):
            problem.bqm.add_variable(q, 0.0)
        return problem

    @classmethod
    def from_dict(cls, data):
        """From {"vartype": "SPIN" | "BINARY", "linear": {var: bias}, "quadratic": [[u, v, bias]], "offset"}."""
        dimod = _ocean()[0]
        linear = {_key(k): float(v) for k, v in data.get("linear", {}).items()}
        quadratic = {(_key(u), _key(v)): float(b) for u, v, b in data.get("quadratic", [])}
        return cls(dimod.BinaryQuadraticModel(linear, quadratic, float(data.get("offset", 0.0)),
                                              data.get("vartype", "SPIN")))

    def to_dict(self):
        return {"vartype": self.bqm.vartype.name, "linear": {str(k): v for k, v in self.bqm.linear.items()},
                "quadratic": [[u, v, b] for (u, v), b in self.bqm.quadratic.items()], "offset": self.bqm.offset}

    @property
    def n_variables(self):
        return self.bqm.num_variables

    def ground_state(self):
        """(energy, sample) by checking every state; at most 24 variables."""
        variables = list(self.bqm.variables)
        n = len(variables)
        if n > 24:
            raise ValueError("exact solution is limited to 24 variables")
        spin = self.bqm.vartype.name == "SPIN"
        index = np.arange(2 ** n, dtype=np.uint32)

        def value(i):   # bit i of the state index is variable i: 0 means spin +1 (or binary 0)
            bit = ((index >> i) & 1).astype(np.int8)
            return 1 - 2 * bit if spin else bit
        position = {v: i for i, v in enumerate(variables)}
        cache = {}

        def column(v):
            if v not in cache:
                cache[v] = value(position[v])
            return cache[v]
        energy = np.full(2 ** n, self.bqm.offset, dtype=float)
        for v, b in self.bqm.linear.items():
            energy += b * column(v)
        for (u, v), b in self.bqm.quadratic.items():
            energy += b * (column(u) * column(v))
        best = int(np.argmin(energy))
        return float(energy[best]), {v: int(column(v)[best]) for v in variables}


def _key(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return text


def anneal(problem, device=None, *, sampler="sqa", num_reads=100, chain_strength=None, control_noise=0.0,
           seed=None, embedding=None, num_sweeps=None):
    """Sample low-energy states of `problem`, on `device` (embedded) or directly.

    control_noise is the standard deviation of Gaussian integrated control
    errors added to every rescaled h and J on the chip (D-Wave quotes a few
    hundredths). Returns plain data: best energy and sample, energy statistics,
    the ground energy and the probability of reaching it when the problem has
    at most 24 variables, and for a chip the embedding (qubits used, chain
    lengths) and the mean fraction of broken chains.
    """
    dimod, de, samplers, minorminer = _ocean()
    if isinstance(problem, Problem):
        bqm = problem.bqm.change_vartype("SPIN", inplace=False)
    else:
        bqm = problem.change_vartype("SPIN", inplace=False)
    rng = np.random.default_rng(seed)
    chosen = {"sqa": samplers.PathIntegralAnnealingSampler, "sa": samplers.SimulatedAnnealingSampler,
              "tabu": samplers.TabuSampler, "exact": dimod.ExactSolver}.get(sampler)
    if chosen is None:
        raise ValueError("sampler must be sqa, sa, tabu or exact")
    options = {} if sampler == "exact" else {"num_reads": int(num_reads), "seed": int(rng.integers(2 ** 31))}
    if sampler in ("sqa", "sa") and num_sweeps is not None:
        options["num_sweeps"] = int(num_sweeps)
    if sampler == "exact" and bqm.num_variables > 24:
        raise ValueError("the exact sampler is limited to 24 variables")
    start = time.perf_counter()
    report = {"sampler": sampler, "num_reads": int(num_reads) if sampler != "exact" else None,
              "variables": bqm.num_variables, "interactions": bqm.num_interactions}
    if device is None:
        samples = chosen().sample(bqm, **options)
    else:
        if device.family != "annealer":
            raise ValueError(f"{device.name} is a gate-model chip; anneal on dwave-advantage, dwave-advantage2, "
                             "chimera, pegasus or zephyr")
        source = list(bqm.quadratic) or []
        target = [tuple(e) for e in device.edges]
        if embedding is None:
            embedding = minorminer.find_embedding(source + [(v, v) for v in bqm.variables if not bqm.adj[v]],
                                                  target, random_seed=int(rng.integers(2 ** 31)))
            # Isolated variables need one qubit each; find_embedding ignores self-loops, so place them.
            used = {q for chain in embedding.values() for q in chain}
            free = iter(q for q in range(device.n_qubits) if q not in used)
            for v in bqm.variables:
                if v not in embedding:
                    embedding[v] = [next(free)]
        if not embedding or len(embedding) < bqm.num_variables:
            raise ValueError(f"no embedding of {bqm.num_variables} variables into {device.name} was found; "
                             "use a larger chip or a sparser problem")
        adjacency = {q: set() for q in range(device.n_qubits)}
        for a, b in target:
            adjacency[a].add(b)
            adjacency[b].add(a)
        strength = chain_strength if chain_strength is not None else de.chain_strength.uniform_torque_compensation(
            bqm, embedding)
        physical = de.embed_bqm(bqm, embedding, adjacency, chain_strength=strength)
        properties = device.parameters.get("annealer", ANNEALER_PROPERTIES)
        h_max = max(abs(x) for x in properties["h_range"])
        j_max = max(abs(x) for x in properties["j_range"])
        scale = max([abs(b) / h_max for b in physical.linear.values()] +
                    [abs(b) / j_max for b in physical.quadratic.values()] + [1e-12])
        physical.scale(1 / scale)
        if control_noise:
            for v in physical.variables:
                physical.add_linear(v, rng.normal(0, control_noise))
            for u, v in list(physical.quadratic):
                physical.add_quadratic(u, v, rng.normal(0, control_noise))
        raw = chosen().sample(physical, **options)
        samples = de.unembed_sampleset(raw, embedding, bqm, chain_break_method=de.majority_vote,
                                       chain_break_fraction=True)
        lengths = [len(chain) for chain in embedding.values()]
        report["embedding"] = {"physical_qubits": sum(lengths), "chip_qubits": device.n_qubits,
                               "max_chain_length": max(lengths), "mean_chain_length": sum(lengths) / len(lengths),
                               "chain_strength": float(strength), "rescaled_by": 1 / scale}
        report["chain_break_fraction"] = float(np.mean(samples.record.chain_break_fraction))
        report["device"] = device.name
        report["control_noise"] = control_noise
    energies = np.repeat(samples.record.energy, samples.record.num_occurrences)
    best = samples.first
    report.update(best_energy=float(best.energy), best_sample={str(k): int(v) for k, v in best.sample.items()},
                  energy={"min": float(energies.min()), "mean": float(energies.mean()),
                          "median": float(np.median(energies)), "max": float(energies.max())},
                  seconds=time.perf_counter() - start,
                  note="a model of annealing (path-integral Monte Carlo, simulated annealing or tabu), not the device")
    if bqm.num_variables <= 24:
        ground, _ = Problem(bqm).ground_state()
        report["ground_energy"] = ground
        report["ground_state_probability"] = float(np.mean(np.isclose(energies, ground, atol=1e-9)))
    return report
