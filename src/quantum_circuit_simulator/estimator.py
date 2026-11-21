"""A Qiskit Estimator V2 primitive backed by Quantum Circuit Simulator.

    from quantum_circuit_simulator.estimator import Estimator
    job = Estimator().run([(circuit, observables, parameter_values)])
    evs = job.result()[0].data.evs

It follows Qiskit's StatevectorEstimator: each PUB broadcasts its parameter
values against its observables, and each element is one expectation value.
Results are exact, so `stds` is zero; with `truncation`, each result also
carries `error_bounds`, a rigorous bound on |exact - returned|. `noise` adds a
uniform depolarizing and readout model (see quantum_circuit_simulator.interop.Noise), which
Qiskit circuits do not carry.

When every circuit parameter enters its angles linearly (k*p + b) and there
are at most 4096 of them, one plan per observable serves every parameter value
of the PUB. Otherwise the circuit is rebuilt for each binding.
"""
from __future__ import annotations

import numpy as np
from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob

from ._core._reference import DEFAULT_MAX_TERMS
from .interop import ConversionError, Noise, _without_readout, from_qiskit, parse_observable, readout_expand


class Estimator(BaseEstimatorV2):
    """Quantum Circuit Simulator as a Qiskit Estimator V2.

    backend, truncation and threads are passed to Circuit.prepare; noise is
    a quantum_circuit_simulator.interop.Noise or mapping; max_terms caps each cone.
    """

    def __init__(self, *, backend="native", truncation=None, threads=1, noise=None, device=None,
                 max_terms=DEFAULT_MAX_TERMS, default_precision=0.0):
        if device is not None and noise is None:
            noise = device.noise_model()  # a quantum_circuit_simulator.devices Device's gate and readout errors
        # Readout applies to the measured Paulis (any axis), not as a circuit channel.
        noise, self.readout = _without_readout(Noise.coerce(noise))
        self.backend = backend
        self.truncation = truncation
        self.threads = threads
        self.noise = noise
        self.max_terms = max_terms
        self._default_precision = default_precision

    def run(self, pubs, *, precision=None):
        if precision is None:
            precision = self._default_precision
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs):
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub):
        circuit = pub.circuit
        source = list(circuit.parameters)
        values = pub.parameter_values.as_array(source) if source else np.zeros(pub.parameter_values.shape + (0,))
        shape = np.broadcast_shapes(pub.parameter_values.shape, pub.observables.shape)
        binding_index = np.broadcast_to(np.arange(int(np.prod(pub.parameter_values.shape, dtype=int)))
                                        .reshape(pub.parameter_values.shape), shape)
        _, observables = np.broadcast_arrays(np.empty(pub.parameter_values.shape), pub.observables)
        flat_values = values.reshape(int(np.prod(pub.parameter_values.shape, dtype=int)), len(source))
        try:
            shared = from_qiskit(circuit, noise=self.noise, max_terms=self.max_terms)
        except ConversionError:
            shared = None  # rebuild per binding below
        evs = np.zeros(shape)
        bounds = np.zeros(shape)
        plans = {}
        for index in np.ndindex(*shape):
            b = int(binding_index[index])
            terms = readout_expand(parse_observable(observables[index], circuit.num_qubits), self.readout)
            key = (None if shared else b, tuple((c, tuple(sorted(p.items()))) for c, p in terms))
            if key not in plans:
                converted = shared or from_qiskit(circuit, values=flat_values[b], noise=self.noise,
                                                  max_terms=self.max_terms)
                plans[key] = (converted, converted.circuit.prepare(
                    terms, backend=self.backend, truncation=self.truncation, threads=self.threads))
            converted, plan = plans[key]
            shifts = converted.shifts(flat_values[b]) if converted.parameters else None
            result = plan.evaluate(shifts, gradients=False)
            evs[index] = result.value
            bounds[index] = result.error_bound or 0.
        fields = dict(evs=evs, stds=np.zeros(shape), shape=shape)
        if self.truncation is not None:
            fields["error_bounds"] = bounds
        return PubResult(DataBin(**fields), metadata={
            "target_precision": pub.precision, "circuit_metadata": circuit.metadata,
            "quantum_circuit_simulator": {"backend": self.backend, "shared_parameters": shared is not None,
                            "plans": len(plans)}})
