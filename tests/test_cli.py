import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from quantum_circuit_simulator._example import EXAMPLE


def cli(args,raw=None):
    return subprocess.run([sys.executable,'-m','quantum_circuit_simulator',*args],input=raw,text=True, encoding="utf-8",capture_output=True,timeout=30)


def test_help_and_example_and_doctor():
    assert cli(['--help']).returncode==0
    assert json.loads(cli(['example']).stdout)==EXAMPLE
    diagnostic=json.loads(cli(['doctor']).stdout)
    assert diagnostic['native_available'] is True


@pytest.mark.parametrize('backend',['python','native','graph','dense','auto'])
def test_installed_cli_pipeline(backend,tmp_path):
    path=tmp_path/'input.json';path.write_text(json.dumps(EXAMPLE))
    completed=cli(['run',str(path),'--backend',backend])
    assert completed.returncode==0,completed.stderr+completed.stdout
    out=json.loads(completed.stdout)
    assert out['ok'] and out['backend']==('native' if backend=='auto' else backend)
    assert out['requested_backend']==backend
    assert len(out['results'])==3
    assert len(out['results'][0]['gradient'])==1
    assert out['timings']['parent_worker_wall_seconds']>0
    assert out['numerics']['certified_global_error_bound'] is False


@pytest.mark.parametrize('args,data',[
    (['--value-only'], EXAMPLE),
    (['--backend','graph'], dict(EXAMPLE, gradients=False)),
    (['--backend','dense'], dict(EXAMPLE, gradients=False)),
])
def test_value_only_cli_preserves_parameter_rebinding(args,data):
    completed=cli(['run','-',*args],json.dumps(data))
    assert completed.returncode==0, completed.stderr+completed.stdout
    output=json.loads(completed.stdout)
    assert output['gradients'] is False
    assert all(row['gradient']==[] for row in output['results'])
    assert len({row['value'] for row in output['results']})==3


@pytest.mark.parametrize('raw',[
    '{"schema_version":1,"schema_version":1}', 'not json', '{}',
    json.dumps(dict(EXAMPLE,unknown=1)),
    json.dumps(dict(EXAMPLE,n_qubits=True)),
    json.dumps(dict(EXAMPLE,parameter_vectors=[[float('nan')]])),
    json.dumps(dict(EXAMPLE,observables=[{'coefficient':1,'pauli':{'01':'Z'}}])),
    json.dumps(dict(EXAMPLE,operations=[{'op':'H','wires':[0],'junk':3}])),
    json.dumps(dict(EXAMPLE,parameter_vectors=[[0,0]])),
    json.dumps(dict(EXAMPLE,gradients='false')),
])
def test_invalid_json_is_failed_json_and_nonzero(raw):
    out=cli(['run','-'],raw)
    assert out.returncode!=0
    payload=json.loads(out.stdout)
    assert payload['ok'] is False
    assert 'results' not in payload


def test_input_size_is_bounded():
    out=cli(['run','-'],' '*(4*1024*1024+1))
    assert out.returncode==2
    assert json.loads(out.stdout)['error']['type']=='InputTooLarge'


def test_timeout_kills_worker():
    out=cli(['run','-','--timeout','0.000001'],json.dumps(EXAMPLE))
    assert out.returncode==124
    assert json.loads(out.stdout)['error']['type']=='Timeout'


def test_cli_truncation_and_threads_report_bounds():
    exact=json.loads(cli(['run','-'],json.dumps(EXAMPLE)).stdout)
    for backend in ('native','graph'):
        completed=cli(['run','-','--backend',backend,'--min-coefficient','1e-3','--max-weight','1','--threads','2'],
                      json.dumps(EXAMPLE))
        assert completed.returncode==0, completed.stderr+completed.stdout
        out=json.loads(completed.stdout)
        assert out['numerics']['truncation']['max_weight']==1
        for got,want in zip(out['results'],exact['results']):
            assert abs(got['value']-want['value'])<=got['error_bound']*(1+1e-12)+1e-14
            assert len(got['gradient_error_bound'])==1


def test_cli_rejects_truncation_on_exact_only_backends():
    completed=cli(['run','-','--backend','python','--min-coefficient','1e-3'],json.dumps(EXAMPLE))
    assert completed.returncode==2
    assert 'truncation' in json.loads(completed.stdout)['error']['message']
