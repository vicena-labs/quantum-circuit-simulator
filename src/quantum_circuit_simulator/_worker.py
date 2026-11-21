"""One finite request per process. The CLI parent enforces a wall deadline."""
from __future__ import annotations
import json
import sys
import time


def strict_json(text):
    def pairs(items):
        obj={}
        for k,v in items:
            if k in obj: raise ValueError(f"duplicate JSON key: {k}")
            obj[k]=v
        return obj
    def constant(value): raise ValueError(f"nonfinite JSON value {value}")
    return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)


def emit(kind, body, data, start, **facts):
    """Print one result document: the envelope (schema, provenance) around the body and timings."""
    from .contracts import envelope
    timings = body.pop('timings', {})
    timings['worker_seconds'] = time.perf_counter() - start
    print(json.dumps({**envelope(kind, body, data, **facts), 'timings': timings},
                     allow_nan=False, separators=(',', ':')), flush=True)
    return 0


def main():
    try:
        backend=sys.argv[1]
        start=time.perf_counter()
        data=strict_json(sys.stdin.buffer.read(4*1024*1024+1).decode('utf-8'))
        options=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
        if isinstance(data, dict) and 'task' in data:
            from . import tasks
            kind, body = tasks.run(data['task'], progress=options.get('progress', False))
            return emit(kind, body, data, start, seed=data['task'].get('seed'))
        if isinstance(data, dict) and 'compare' in data:
            from .design import compare
            from .devices import Device
            spec = data['compare']
            result = compare([Device.from_dict(c) for c in spec['chips']], family=spec.get('family', 'qaoa-ring'),
                             sizes=spec.get('sizes', [8, 12]), noise_scale=spec.get('noise_scale', 1.0),
                             router=spec.get('router', 'greedy'), family_parameters=spec.get('family_parameters'))
            return emit('compare', {'rows': result.rows}, data, start)
        if isinstance(data, dict) and 'sweep' in data:
            from .experiments import sweep
            spec = data['sweep']
            allowed = {'circuit', 'family', 'family_parameters', 'sizes', 'observable', 'devices', 'noise_scales',
                       'router', 'layout', 'idle_noise', 'checkpoint'}
            unknown = set(spec) - allowed
            if unknown:
                raise ValueError(f"unknown sweep fields {sorted(unknown)}")
            from .contracts import Progress
            points = len(spec.get('sizes') or [None]) * (1 + len(spec.get('devices') or []) * len(spec.get('noise_scales', [1.0])))
            result = sweep(spec.get('family') or spec['circuit'], spec.get('observable'),
                           devices=spec.get('devices'), noise_scales=spec.get('noise_scales', [1.0]),
                           sizes=spec.get('sizes'), router=spec.get('router', 'greedy'),
                           layout=spec.get('layout', 'greedy'), idle_noise=spec.get('idle_noise', True),
                           family_parameters=spec.get('family_parameters'), checkpoint=spec.get('checkpoint'),
                           progress=Progress('sweep', points, enabled=options.get('progress', False)))
            return emit('sweep', {'rows': result.rows}, data, start)
        from .schema import decode_request
        decoded = decode_request(data)
        c, obs, vecs = decoded[:3]
        compile_report = decoded[3] if len(decoded) > 3 else None
        device_check = None
        if compile_report is not None and len(sys.argv) > 2 and sys.argv[2] == 'compile':
            if not options.get('schedule'):
                compile_report = {k: v for k, v in compile_report.items() if k != 'schedule'}
            return emit('compile', {'compile': compile_report}, data, start)
        if isinstance(data, dict) and 'device' in data and compile_report is None:
            from .devices import Device
            device = Device.from_dict(data['device'])
            pairs = device.uncoupled_pairs(c)
            device_check = {'name': device.name, 'n_qubits': device.n_qubits,
                            'noise_applied': 'noise' not in data,
                            'uncoupled_two_qubit_operations': len(pairs),
                            'first_uncoupled_pairs': [list(p) for p in pairs[:5]],
                            'note': 'two-qubit operations must act on coupled qubits to run on this chip; '
                                    'route the circuit onto the device first' if pairs else
                                    'every two-qubit operation acts on coupled qubits'}
        if len(sys.argv) > 2 and sys.argv[2] == 'plan':
            from .agent import plan
            return emit('plan', {'plan': plan(c, obs), **({'device': device_check} if device_check else {})},
                        data, start)
        gradients = data.get('gradients', True) and not (len(sys.argv)>2 and sys.argv[2]=='value')
        # Options from the CLI parent: threads and an optional truncation.
        from .api import Truncation
        truncation=Truncation(**options['truncation']) if 'truncation' in options else None
        if 'converge' in options:
            return _converge(c, obs, vecs, gradients, options, compile_report, device_check, data, start)
        t=time.perf_counter()
        plan=c.prepare(obs,backend=backend,truncation=truncation,threads=options.get('threads',1))
        preparation=time.perf_counter()-t
        t=time.perf_counter(); results=plan.evaluate_many(vecs,gradients=gradients); computation=time.perf_counter()-t
        if compile_report:
            compile_report = {k: v for k, v in compile_report.items() if k != 'schedule'}
        output={'backend':plan.backend,'requested_backend':backend,'n_qubits':c.n_qubits,
            'n_parameters':c.n_parameters,'results':[x.as_dict() for x in results],
            'gradients':gradients, 'diagnostics':plan.diagnostics,
            **({'device': device_check} if device_check else {}),
            **({'compile': compile_report} if compile_report else {}),
            'numerics':plan.numerics,
            'timings':{'preparation_seconds':preparation,'evaluate_batch_seconds':computation}}
        return emit('result', output, data, start, solver=plan.backend)
    except Exception as exc:
        from .agent import hint_for
        print(json.dumps({'ok':False,'error':{'type':type(exc).__name__,'message':str(exc),
                                              'hint':hint_for(exc)}}),flush=True)
        return 2

def _converge(c, obs, vecs, gradients, options, compile_report, device_check, data, start):
    """Tighten the truncation level by level until successive values agree."""
    from .experiments import ladder, refine
    settings = options['converge']
    t = time.perf_counter()
    results, report = refine(c, obs, vecs, ladder(options['truncation']['min_coefficient'], settings['levels']),
                             tolerance=settings['tolerance'], seconds=settings['seconds'],
                             gradients=gradients, threads=options.get('threads', 1))
    if results is None:
        from .api import ConeOverflow
        raise ConeOverflow(report['stopped'])
    output = {'backend': 'native', 'requested_backend': 'native', 'n_qubits': c.n_qubits,
              'n_parameters': c.n_parameters, 'results': [x.as_dict() for x in results],
              'gradients': gradients, 'convergence': report,
              **({'device': device_check} if device_check else {}),
              **({'compile': {k: v for k, v in compile_report.items() if k != 'schedule'}} if compile_report else {}),
              'timings': {'evaluate_batch_seconds': time.perf_counter() - t}}
    return emit('result', output, data, start, solver='native')


if __name__=='__main__':raise SystemExit(main())
