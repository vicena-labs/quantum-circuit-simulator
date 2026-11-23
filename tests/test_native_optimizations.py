"""Qualify product-input contraction, sparse rows, and the value-only C ABI.

The small density oracle below evolves explicit matrices and Kraus operators;
it does not reuse Pauli-transfer propagation or its derivative recurrence.
"""
import ctypes as ct
import math

import numpy as np
import pytest

from quantum_circuit_simulator._core._native import CStage, CStats, CTerm, NativePrepared, library
from quantum_circuit_simulator._core._reference import ConeOverflow, NoisyPauliCircuit


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / math.sqrt(2)
SX = (I + 1j * I + X - 1j * X) / 2
CZ = np.diag([1, 1, 1, -1]).astype(complex)
ECR = np.array([[0, 0, 1, 1j], [0, 0, 1j, 1],
                [1, -1j, 0, 0], [-1j, 1, 0, 0]], complex) / math.sqrt(2)


def _embed(operator, wires, nqubits):
    dimension = 1 << nqubits
    result = np.zeros((dimension, dimension), complex)
    for column in range(dimension):
        local_column = sum(((column >> q) & 1) << i for i, q in enumerate(wires))
        for local_row in range(1 << len(wires)):
            row = column
            for i, q in enumerate(wires):
                row = (row & ~(1 << q)) | (((local_row >> i) & 1) << q)
            result[row, column] = operator[local_row, local_column]
    return result


def _density(n, operations, observables, shifts, initial):
    v = np.ones(1 << n, complex) / math.sqrt(1 << n) if initial == 'plus' else np.eye(1 << n, dtype=complex)[:, 0]
    rho = np.outer(v, v.conj())
    for name, wires, payload, param in operations:
        if name == 'D2':
            noisy = (1 - payload) * rho
            for a in (I, X, Y, Z):
                for b in (I, X, Y, Z):
                    if a is I and b is I:
                        continue
                    u = _embed(np.kron(b, a), wires, n)
                    noisy += (payload / 15) * (u @ rho @ u.conj().T)
            rho = noisy
            continue
        if name == 'RELAX':
            duration, t1, t2 = payload
            gamma = -math.expm1(-duration / t1)
            a0 = _embed(np.diag([1, math.sqrt(1 - gamma)]), wires, n)
            a1 = _embed(np.array([[0, math.sqrt(gamma)], [0, 0]], complex), wires, n)
            rho = a0 @ rho @ a0.conj().T + a1 @ rho @ a1.conj().T
            coherence = math.exp(-duration / t2) / math.sqrt(1 - gamma)
            u = _embed(Z, wires, n)
            rho = (1 + coherence) / 2 * rho + (1 - coherence) / 2 * (u @ rho @ u)
            continue
        if name in ('RX', 'RY', 'RZ'):
            theta = payload + (shifts[param] if param >= 0 else 0)
            axis = {'RX': X, 'RY': Y, 'RZ': Z}[name]
            local = math.cos(theta / 2) * I - 1j * math.sin(theta / 2) * axis
        elif name == 'FSIM':
            theta, phi = payload
            theta += shifts[param] if param >= 0 else 0
            c, s = math.cos(theta), math.sin(theta)
            local = np.array([[1, 0, 0, 0], [0, c, -1j * s, 0],
                              [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]], complex)
        else:
            local = {'H': H, 'SX': SX, 'CZ': CZ, 'ECR': ECR}[name]
        u = _embed(local, wires, n)
        rho = u @ rho @ u.conj().T
    answer = 0.
    for coefficient, ops in observables:
        measured = np.eye(1 << n, dtype=complex)
        for q, axis in ops.items():
            measured = measured @ _embed({'X': X, 'Y': Y, 'Z': Z}[axis], (q,), n)
        answer += coefficient * np.trace(measured @ rho).real
    return float(answer)


def _mixed_case(seed, n):
    rng = np.random.default_rng(seed)
    circuit = NoisyPauliCircuit(n, 3, max_terms=4096)
    operations = []

    def gate(name, wires, theta=0., param=-1):
        circuit.gate(name, wires, theta, param)
        operations.append((name, tuple(wires), theta, param))

    # Stagger first touches and revisit a wire across another wire's boundary.
    for q in range(n):
        gate('RY', (q,), float(rng.uniform(-.8, .8)), q % 3)
        if q:
            gate('ECR' if q % 2 else 'CZ', (q - 1, q))
        gate('RZ', (0,), float(rng.uniform(-.8, .8)), 2)
    gate('SX', (n - 1,))
    circuit.fsim((n - 1, 0), .31, -.27, param=1)
    operations.append(('FSIM', (n - 1, 0), (.31, -.27), 1))
    circuit.channel_depolarize((0, n - 1), .023)
    operations.append(('D2', (0, n - 1), .023, -1))
    for q in range(n):
        circuit.channel_relax(q, .17 + q * .03, 1.3, 1.7)
        operations.append(('RELAX', (q,), (.17 + q * .03, 1.3, 1.7), -1))
        gate('RX', (q,), float(rng.uniform(-.8, .8)), q % 3)
    observables = [(.7, {0: 'Z'}), (-.2, {0: 'X', n - 1: 'Y'}), (.13, {})]
    return circuit, operations, observables


@pytest.mark.parametrize('initial', ['zero', 'plus'])
@pytest.mark.parametrize('seed,n', [(0, 2), (1, 3), (2, 4), (3, 5)])
def test_contracted_native_against_independent_density(seed, n, initial):
    circuit, operations, observables = _mixed_case(seed, n)
    prepared = NativePrepared(circuit)
    for shifts in ((0., 0., 0.), (.17, -.09, .22)):
        prepared.bind(shifts)
        value, gradient = prepared.evaluate(observables, initial)
        expected = _density(n, operations, observables, shifts, initial)
        assert value == pytest.approx(expected, abs=3e-12)
        eps = 2e-5
        for parameter in range(3):
            plus, minus = list(shifts), list(shifts)
            plus[parameter] += eps
            minus[parameter] -= eps
            derivative = (_density(n, operations, observables, plus, initial)
                          - _density(n, operations, observables, minus, initial)) / (2 * eps)
            assert gradient[parameter] == pytest.approx(derivative, abs=3e-8)


@pytest.mark.parametrize('n', [16, 64, 256])
def test_affine_relaxation_contracts_before_term_limit(n):
    circuit = NoisyPauliCircuit(n, max_terms=1)
    for q in range(n):
        circuit.channel_relax(q, -math.log(.75), 1., 1.)
    value, gradient, stats = NativePrepared(circuit).evaluate(
        [(1., {q: 'Z' for q in range(n)})], include_stats=True)
    assert value == pytest.approx(1., abs=2e-14)
    assert gradient.size == 0
    assert stats['max_terms'] == stats['final_terms'] == 1


@pytest.mark.parametrize('initial,axis,gate', [('zero', 'Z', 'RY'), ('plus', 'X', 'RZ')])
def test_large_product_parity_value_and_shared_gradient(initial, axis, gate):
    n, theta, probability = 128, .17, .01
    circuit = NoisyPauliCircuit(n, 16, max_terms=1)
    for q in range(n):
        circuit.gate(gate, (q,), theta, param=15)
        circuit.channel_depolarize((q,), probability)
    value, gradient, stats = NativePrepared(circuit).evaluate(
        [(1., {q: axis for q in range(n)})], initial, include_stats=True)
    scale = 1 - 4 * probability / 3
    expected = (scale * math.cos(theta)) ** n
    assert value == pytest.approx(expected, rel=3e-13, abs=1e-15)
    expected_gradient = -n * math.sin(theta) * scale ** n * math.cos(theta) ** (n - 1)
    assert gradient[15] == pytest.approx(expected_gradient, rel=3e-13, abs=1e-14)
    assert np.array_equal(gradient[:15], np.zeros(15))
    assert stats['max_terms'] == 1


@pytest.mark.parametrize('initial', ['zero', 'plus'])
def test_untouched_wires_and_empty_circuit(initial):
    circuit = NoisyPauliCircuit(256, 2, max_terms=1)
    surviving = 'Z' if initial == 'zero' else 'X'
    killed = 'X' if initial == 'zero' else 'Z'
    # Several distinct masks collapse into the same scalar at the input boundary.
    observables = [(2., {255: surviving}), (.5, {63: surviving, 64: surviving}),
                   (-.3, {}), (8., {128: killed})]
    # Exercise the kernel's actual live-term cap independently of Python
    # observable normalization and query splitting.
    code, out, stats = _call(circuit, observables, initial=initial)
    assert code == 0
    assert out[0] == pytest.approx(2.2)
    assert np.array_equal(out[1:], [0., 0.])
    assert stats.peak_terms == stats.final_terms == 1


def test_derivative_only_transition_and_rebinding_through_zeros():
    circuit = NoisyPauliCircuit(1, 1, max_terms=1).gate('RY', (0,), 0., param=0)
    prepared = NativePrepared(circuit)
    for shift in (0., .41, -math.pi / 2, -.52, 0., math.pi / 2):
        prepared.bind([shift])
        value, gradient = prepared.evaluate([(1., {0: 'X'})])
        assert value == pytest.approx(math.sin(shift), abs=2e-14)
        assert gradient[0] == pytest.approx(math.cos(shift), abs=2e-14)


def _brickwork(max_terms, angle=None):
    # Wide enough that merged stages keep several terms open; angle=0 makes
    # every rotation the identity, so only derivatives branch.
    circuit = NoisyPauliCircuit(4, 1, max_terms=max_terms)
    for layer in range(3):
        for q in range(4):
            if angle is None:
                circuit.gate('RY', (q,), .3 + .1 * q + .05 * layer, param=0 if q == 0 else -1)
            else:
                circuit.gate('RX', (q,), angle, param=0)
        for q in range(layer % 2, 3, 2):
            circuit.gate('CZ', (q, q + 1))
    return circuit


def test_uncontracted_expansion_still_fails_closed():
    circuit = _brickwork(2)
    with pytest.raises(ConeOverflow):
        NativePrepared(circuit).evaluate([(1., {1: 'X'})])
    with pytest.raises(ConeOverflow):
        circuit.evaluate([(1., {1: 'X'})])


def _call(circuit, observables, initial='zero', value_only=False, indices=None):
    prepared = NativePrepared(circuit)
    terms = (CTerm * len(observables))()
    for i, (coefficient, ops) in enumerate(observables):
        x, z = circuit.mask(tuple(ops.items()))
        for word in range(4):
            terms[i].x[word] = (x >> (64 * word)) & ((1 << 64) - 1)
            terms[i].z[word] = (z >> (64 * word)) & ((1 << 64) - 1)
        terms[i].coef = coefficient
    # Sentinels verify that the scalar ABI writes exactly one output double.
    length = 3 if value_only else circuit.nparams + 1
    output = (ct.c_double * length)(*([12345.] * length))
    stats = CStats()
    name = 'run_pauli_indexed' if indices is not None else 'run_pauli'
    if value_only:
        name += '_value'
    function = getattr(library(), name)
    prefix = [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(CStage)]
    args = [circuit.nqubits, circuit.nparams,
            len(indices) if indices is not None else len(prepared.stages), prepared.stages]
    if indices is not None:
        prefix += [ct.c_int, ct.POINTER(ct.c_int32)]
        index_array = (ct.c_int32 * len(indices))(*indices)
        args += [len(prepared.stages), index_array]
    function.argtypes = prefix + [ct.c_int, ct.POINTER(CTerm), ct.c_int, ct.c_uint64,
                                 ct.POINTER(ct.c_double), ct.POINTER(CStats)]
    function.restype = ct.c_int
    args += [len(terms), terms, 0 if initial == 'zero' else 1, circuit.max_terms, output, ct.byref(stats)]
    code = function(*args)
    return code, np.array(list(output)), stats


@pytest.mark.parametrize('indexed', [False, True])
def test_value_only_abi_keeps_parameters_and_writes_one_double(indexed):
    circuit = NoisyPauliCircuit(2, 16, max_terms=1)
    circuit.gate('H', (1,)).gate('RY', (0,), .37, param=15)
    indices = [1] if indexed else None
    code, value, _ = _call(circuit, [(1., {0: 'X'})], value_only=True, indices=indices)
    assert code == 0
    assert value[0] == pytest.approx(math.sin(.37), abs=2e-14)
    assert np.array_equal(value[1:], [12345., 12345.])
    code, tangent, _ = _call(circuit, [(1., {0: 'X'})], indices=indices)
    assert code == 0
    assert tangent[16] == pytest.approx(math.cos(.37), abs=2e-14)
    assert np.array_equal(tangent[1:16], np.zeros(15))


def test_value_only_drops_derivative_only_expansion():
    circuit = _brickwork(1, angle=0.)
    # Z1 -> Z1 through identity rotations and CZs, but d/dtheta of each RX
    # branches into Y. With a term cap of one, the value fits and the full
    # tangent propagation must fail honestly.
    code, value, _ = _call(circuit, [(1., {1: 'Z'})], value_only=True)
    assert code == 0
    assert value[0] == pytest.approx(1., abs=2e-14)
    code, _, _ = _call(circuit, [(1., {1: 'Z'})])
    assert code == -1


def test_c_abi_rejects_out_of_register_masks():
    circuit = NoisyPauliCircuit(1)
    code, _, _ = _call(circuit, [(1., {255: 'Z'})])
    assert code == -2
