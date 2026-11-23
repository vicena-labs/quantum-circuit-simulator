"""The command line as an agent uses it: JSON everywhere, OpenQASM input, dry-run plans and error hints."""
import json
import math
import subprocess
import sys

import pytest

from quantum_circuit_simulator import Circuit
from quantum_circuit_simulator.agent import plan
from quantum_circuit_simulator.interop import expectation

BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry(0.7) q[0];
cx q[0], q[1];
"""


def cli(args, stdin=None):
    done = subprocess.run([sys.executable, '-m', 'quantum_circuit_simulator', *args], input=stdin, text=True, encoding="utf-8",
                          capture_output=True, timeout=60)
    return done.returncode, done.stdout


@pytest.fixture
def bell(tmp_path):
    path = tmp_path / 'bell.qasm'
    path.write_text(BELL)
    return str(path)


def test_self_description_commands_are_machine_readable():
    code, out = cli(['capabilities'])
    caps = json.loads(out)
    assert code == 0 and 'ECR' in caps['native_gates'] and 'cx' in caps['converted_gates']
    assert caps['exit_codes']['124'] and caps['limits']['shared_parameters'] == 4096
    code, out = cli(['schema'])
    assert code == 0 and {'circuit', 'openqasm'} == {s['title'] for s in json.loads(out)['request']['oneOf']}
    code, out = cli(['guide'])
    assert code == 0 and out.startswith('# Quantum Circuit Simulator: guide for agents') and 'quantum-circuit-simulator plan' in out


def test_run_openqasm_with_observable_noise_and_truncation(bell):
    code, out = cli(['run', bell, '--observable', 'Z0*Z1 + 0.5*Z1'])
    result = json.loads(out)
    assert code == 0 and result['ok']
    assert result['results'][0]['value'] == pytest.approx(1 + 0.5 * math.cos(0.7), abs=1e-12)
    assert result['results'][0]['value'] == pytest.approx(expectation(BELL, 'Z0*Z1 + 0.5*Z1').value, abs=1e-14)
    code, out = cli(['run', bell, '--observable', 'Z0*Z1', '--noise-2q', '0.05', '--readout', '0.02,0.03'])
    assert code == 0 and json.loads(out)['results'][0]['value'] < 1
    code, out = cli(['run', bell, '--observable', 'Z1', '--min-coefficient', '1e-3', '--backend', 'graph'])
    row = json.loads(out)['results'][0]
    assert code == 0 and 'error_bound' in row


def test_openqasm_request_form_on_standard_input():
    request = {'schema_version': 1, 'qasm': BELL, 'observables': '0.5*IZ + 0.5*ZI', 'gradients': False}
    code, out = cli(['run', '-'], json.dumps(request))
    assert code == 0 and json.loads(out)['results'][0]['value'] == pytest.approx(math.cos(0.7), abs=1e-12)


def test_errors_carry_actionable_hints(bell):
    code, out = cli(['run', bell])
    error = json.loads(out)['error']
    assert code == 2 and error['type'] == 'UsageError' and '--observable' in error['hint']
    deep = {'schema_version': 1, 'n_qubits': 6, 'max_terms': 4,
            'operations': [{'op': op, 'wires': [q], 'theta': 0.3 * (q + 1)} for q in range(6) for op in ('RX', 'RY')]
                          + [gate for layer in range(3) for gate in
                             [{'op': 'CZ', 'wires': [q, q + 1]} for q in range(layer % 2, 5, 2)]
                             + [{'op': 'RY', 'wires': [q], 'theta': 0.2 + 0.1 * layer} for q in range(6)]],
            'observables': [{'coefficient': 1.0, 'pauli': {'2': 'X', '3': 'Z'}}]}
    code, out = cli(['run', '-'], json.dumps(deep))
    error = json.loads(out)['error']
    assert code == 2 and error['type'] == 'ConeOverflow' and '--min-coefficient' in error['hint']


def test_plan_reports_feasibility_and_its_guarantee_holds(bell):
    code, out = cli(['plan', bell, '--observable', 'Z0*Z1'])
    facts = json.loads(out)['plan']
    assert code == 0 and facts['exact_guaranteed'] and facts['widest_cone_wires'] == 2
    # Whenever the plan guarantees exact feasibility, exact evaluation must succeed.
    for depth in range(1, 6):
        c = Circuit(5, 1, max_terms=2000)
        for layer in range(depth):
            for q in range(5):
                c.gate('RY', [q], 0.3 + 0.1 * q + layer, param=0)
            for q in range(layer % 2, 4, 2):
                c.gate('CZ', [q, q + 1])
        obs = [(1.0, {2: 'Z'}), (0.5, {0: 'X', 1: 'X'})]
        facts = plan(c, obs)
        if facts['exact_guaranteed']:
            c.prepare(obs).evaluate([0.1])
        assert facts['recommendations']
