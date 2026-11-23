import math
import numpy as np
import pytest
from quantum_circuit_simulator import Circuit, ConeOverflow


@pytest.mark.parametrize('backend',['native','python'])
@pytest.mark.parametrize('angle',[-1.2,-.4,0.,.3,1.1])
def test_relaxation_and_gradient_against_analytic_answer(backend,angle):
    t,t1,t2=2e-6,70e-6,80e-6
    a=math.exp(-t/t1)
    c=Circuit(1,1).gate('RY',[0],theta=angle,param=0).relax(0,t,t1,t2).readout(0,.03,.02)
    p=c.prepare([(1.,{0:'Z'})],backend=backend)
    for shift in (0.,.13,-.21):
        result=p.evaluate([shift])
        expected=-.01+.95*(a*math.cos(angle+shift)+(1-a))
        derivative=-.95*a*math.sin(angle+shift)
        assert result.value==pytest.approx(expected,abs=2e-12)
        assert result.gradient[0]==pytest.approx(derivative,abs=2e-12)


@pytest.mark.parametrize('backend',['native','python'])
def test_repeated_parameter_chain_rule_and_snapshot(backend):
    c=Circuit(1,1).gate('RY',[0],.1,0).gate('RY',[0],.2,0)
    p=c.prepare([(1.,{0:'Z'})],backend=backend)
    c.gate('X',[0])  # Later edits must not change a prepared snapshot.
    rows=p.evaluate_many([[.03],[.21]])
    for d,out in zip((.03,.21),rows):
        assert out.value==pytest.approx(math.cos(.3+2*d),abs=2e-12)
        assert out.gradient[0]==pytest.approx(-2*math.sin(.3+2*d),abs=2e-12)


@pytest.mark.parametrize('seed',range(6))
def test_mixed_pauli_observables_gradient_native_reference(seed):
    rng=np.random.default_rng(seed)
    c=Circuit(3,2,max_terms=4096)
    for layer in range(2):
        for wire in range(3):c.gate('RY',[wire],float(rng.uniform(-1,1)),layer)
        c.gate('CZ',[0,1]).fsim([1,2],.21,.33,param=1)
        c.depolarize([0,1],.02).pauli_error(2,'X',.01).relax(1,1e-6,65e-6,81e-6)
    obs=[(.4,{0:'X'}),(-.3,{1:'Y'}),(.6,{0:'Z',2:'Z'}),(0.1,{})]
    native=c.prepare(obs); ref=c.prepare(obs,backend='python')
    for shift in ([.03,-.07],[-.04,.11]):
        n,p=native.evaluate(shift),ref.evaluate(shift)
        assert n.value==pytest.approx(p.value,abs=2e-11)
        assert np.allclose(n.gradient,p.gradient,atol=2e-10,rtol=0)
        h=2e-5
        for index in range(2):
            lo=list(shift);hi=list(shift);lo[index]-=h;hi[index]+=h
            finite_diff=(ref.evaluate(hi).value-ref.evaluate(lo).value)/(2*h)
            assert n.gradient[index]==pytest.approx(finite_diff,abs=3e-7)


@pytest.mark.parametrize('n',[1,2,25,127,256])
def test_large_wire_index_local_analytic_answer(n):
    c=Circuit(n,1,max_terms=8).gate('RY',[n-1],.4,0)
    out=c.prepare([(1.,{n-1:'Z'})]).evaluate([.1])
    assert out.value==pytest.approx(math.cos(.5),abs=2e-12)
    assert out.gradient[0]==pytest.approx(-math.sin(.5),abs=2e-12)


@pytest.mark.parametrize('backend',['native','python'])
def test_overflow_returns_no_partial_success(backend):
    # Gates on one or two wires merge into a stage that closes at once, so the
    # overflow needs a brickwork wide enough to keep several terms open.
    c=Circuit(4,max_terms=1)
    for layer in range(3):
        for q in range(4): c.gate('RY',[q],.3+.1*q+.05*layer)
        for q in range(layer%2,3,2): c.gate('CZ',[q,q+1])
    with pytest.raises(ConeOverflow):c.prepare([(1.,{1:'X'})],backend=backend).evaluate()


@pytest.mark.parametrize('kwargs',[{'n_qubits':True},{'n_qubits':2.0},{'n_qubits':16385},{'n_qubits':0},
                                  {'n_qubits':1,'n_parameters':4097},{'n_qubits':1,'max_terms':True}])
def test_bad_circuit_sizes(kwargs):
    with pytest.raises(ValueError):Circuit(**kwargs)


@pytest.mark.parametrize('action',[
    lambda c:c.gate('RY',[False],.2),lambda c:c.gate('CZ',[0,0]),lambda c:c.gate('CNOT',[0,1]),
    lambda c:c.gate('RY',[2],.2),lambda c:c.gate('RY',[0],float('nan')),
    lambda c:c.gate('H',[0],param=0),lambda c:c.depolarize([0],1.1),
    lambda c:c.relax(0,1,1,3),lambda c:c.readout(0,.2,-.1),
    lambda c:c.prepare([(1.,{2:'Z'})]),lambda c:c.prepare([(float('inf'),{0:'Z'})]),
    lambda c:c.prepare([(1.,{False:'Z'})]),lambda c:c.prepare([(1.,{0:'W'})]),
])
def test_invalid_inputs_fail_before_native_execution(action):
    with pytest.raises(ValueError):action(Circuit(2,1))


def test_readout_semantics_are_explicit():
    c=Circuit(1).readout(0,.03,.02)
    with pytest.raises(ValueError,match='X/Y'):c.prepare([(1.,{0:'X'})])


def test_batch_and_parameter_validation():
    p=Circuit(1,1).gate('RY',[0],.3,0).prepare([(1.,{0:'Z'})])
    for row in ([True],[float('nan')],[],[1,2]):
        with pytest.raises(ValueError):p.evaluate(row)
    with pytest.raises(ValueError):p.evaluate_many([])
    with pytest.raises(ValueError):p.evaluate_many([[0.]]*129)
    p._lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='concurrently'):p.evaluate([0.])
    finally:p._lock.release()
