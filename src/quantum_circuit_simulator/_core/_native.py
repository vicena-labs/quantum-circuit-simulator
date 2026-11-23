"""Native wrapper for C++ sparse Pauli-transfer engine (research prototype)."""
from __future__ import annotations
import ctypes as ct
import importlib.util
import math
from functools import lru_cache
import numpy as np
from ._reference import ConeOverflow


class CStage(ct.Structure):
    _fields_=[('q0',ct.c_int32),('q1',ct.c_int32),('arity',ct.c_int32),('param',ct.c_int32),
              ('ptm',ct.c_double*256),('dptm',ct.c_double*256)]
class CTerm(ct.Structure):
    _fields_=[('x',ct.c_uint64*4),('z',ct.c_uint64*4),('coef',ct.c_double)]
class CStats(ct.Structure):
    _fields_=[('peak_terms',ct.c_uint64),('final_terms',ct.c_uint64)]
class CGroup(ct.Structure):
    _fields_=[('nqubits',ct.c_int32),('nstages',ct.c_int32),('nterms',ct.c_int32),('reserved',ct.c_int32),
              ('indices',ct.POINTER(ct.c_int32)),('local_wires',ct.POINTER(ct.c_int32)),
              ('terms',ct.POINTER(CTerm))]
class CBatchStats(ct.Structure):
    _fields_=[('peak_terms',ct.c_uint64),('final_terms',ct.c_uint64),('truncated_terms',ct.c_uint64),
              ('failed_group',ct.c_int32),('reserved',ct.c_int32),
              ('value_bound',ct.c_double),('gradient_bound',ct.c_double*16)]
class CGraphSupport(ct.Structure):
    _fields_=[('rows',ct.c_uint16*16),('identity',ct.c_uint16),('reserved',ct.c_uint16*3)]
class CGraphStats(ct.Structure):
    _fields_=[('nodes',ct.c_uint64),('edges',ct.c_uint64),('peak_terms',ct.c_uint64),('bytes',ct.c_uint64),
              ('truncated_terms',ct.c_uint64),('failed_group',ct.c_int32),('reserved',ct.c_int32)]

# A NumPy view of the CStage array, so rebinding writes PTMs without a Python
# loop over stages.
_STAGE_DTYPE = np.dtype([('q0','<i4'),('q1','<i4'),('arity','<i4'),('param','<i4'),
                         ('ptm','<f8',(256,)),('dptm','<f8',(256,))])
assert _STAGE_DTYPE.itemsize == ct.sizeof(CStage)

# The native kernel carries at most this many derivatives per pass.
CHUNK = 16

# Rotation PTMs in I, X, Z, Y order: the two moving rows/columns and the sign
# of the sine term, matching _reference.cached_gate_ptm.
_ROTATIONS = {'RX': (2, 3, 1.), 'RY': (1, 2, 1.), 'RZ': (1, 3, -1.)}


@lru_cache(maxsize=1)
def library():
    spec = importlib.util.find_spec('quantum_circuit_simulator._core._native_kernel')
    if spec is None or not spec.origin:
        raise RuntimeError('Native extension is absent. Install a built wheel or select backend=python.')
    lib=ct.CDLL(spec.origin)
    lib.run_pauli.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(CStage),ct.c_int,
                            ct.POINTER(CTerm),ct.c_int,ct.c_uint64,ct.POINTER(ct.c_double),ct.POINTER(CStats)]
    lib.run_pauli.restype=ct.c_int
    lib.run_pauli_value.argtypes=lib.run_pauli.argtypes
    lib.run_pauli_value.restype=ct.c_int
    lib.run_pauli_batch.argtypes=[ct.c_int,ct.POINTER(CStage),ct.c_int,ct.POINTER(ct.c_double),
                                  ct.c_int,ct.POINTER(CGroup),ct.c_int,ct.c_uint64,
                                  ct.c_double,ct.c_int32,ct.c_int,ct.c_int,
                                  ct.POINTER(ct.c_double),ct.POINTER(CBatchStats),ct.POINTER(ct.c_int32)]
    lib.run_pauli_batch.restype=ct.c_int
    lib.pauli_graph_compile.argtypes=[ct.c_int,ct.POINTER(CStage),ct.c_int,ct.POINTER(CGraphSupport),
                                      ct.c_int,ct.POINTER(CGroup),ct.c_int,ct.c_uint64,ct.c_uint64,
                                      ct.c_uint64,ct.c_double,ct.c_int32,ct.POINTER(ct.c_double),
                                      ct.c_int,ct.POINTER(ct.c_void_p),ct.POINTER(CGraphStats),
                                      ct.POINTER(ct.c_int32)]
    lib.pauli_graph_compile.restype=ct.c_int
    lib.pauli_graph_evaluate.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(CStage),ct.c_int,ct.c_int,ct.c_int,
                                       ct.POINTER(ct.c_double),ct.POINTER(ct.c_double)]
    lib.pauli_graph_evaluate.restype=ct.c_int
    lib.pauli_graph_free.argtypes=[ct.c_void_p]
    lib.pauli_graph_free.restype=None
    lib.pauli_bind_rotations.argtypes=[ct.POINTER(CStage),ct.c_int]+[ct.POINTER(ct.c_int32)]*4+[ct.POINTER(ct.c_double)]*4
    lib.pauli_bind_rotations.restype=None
    return lib


class NativePrepared:
    def __init__(self,circuit):
        self.c=circuit
        self.stages=(CStage*len(circuit.stages))()
        for i,s in enumerate(circuit.stages):
            a=self.stages[i]
            a.q0=s.wires[0];a.q1=s.wires[1] if len(s.wires)==2 else -1
            a.arity=len(s.wires);a.param=s.param
            d=len(s.matrix)
            t=np.ascontiguousarray(s.matrix,dtype=np.float64)
            dt=np.ascontiguousarray(s.derivative,dtype=np.float64)
            ct.memmove(ct.addressof(a)+CStage.ptm.offset,t.ctypes.data,d*d*8)
            ct.memmove(ct.addressof(a)+CStage.dptm.offset,dt.ctypes.data,d*d*8)
        view = np.frombuffer(self.stages, dtype=_STAGE_DTYPE) if circuit.stages else None
        self._ptm = view['ptm'] if view is not None else None
        self._dptm = view['dptm'] if view is not None else None
        self.params = view['param'] if view is not None else np.zeros(0, dtype=np.int32)
        # Each parameterised rotation is rewritten in four PTM and four
        # derivative entries; everything else in its PTM is fixed. One native
        # call rebinds them all, since on small circuits rebinding in Python
        # cost more than the propagation itself.
        chosen = [(i, s) for i, s in enumerate(circuit.stages) if s.param >= 0 and s.name in _ROTATIONS]
        def column(values, kind):
            return (kind * len(chosen))(*values)
        self._rotations = len(chosen), (
            column([i for i, _ in chosen], ct.c_int32),
            column([_ROTATIONS[s.name][0] for _, s in chosen], ct.c_int32),
            column([_ROTATIONS[s.name][1] for _, s in chosen], ct.c_int32),
            column([s.param for _, s in chosen], ct.c_int32),
            column([s.angle for _, s in chosen], ct.c_double),
            column([s.scale for _, s in chosen], ct.c_double),
            column([_ROTATIONS[s.name][2] for _, s in chosen], ct.c_double))
        self._shift_buffer = (ct.c_double * max(1, circuit.nparams))()
        self._other = [i for i, s in enumerate(circuit.stages)
                       if s.param >= 0 and s.name not in _ROTATIONS]

    def bind(self,deltas):
        """Bind a distinct shared-parameter vector without rebuilding static noise.

        The supplied deltas are relative to base angles captured at construction.
        Rebinding does not change the original circuit or the shared-stage layout.
        """
        if len(deltas)!=self.c.nparams or not all(math.isfinite(x) for x in deltas):
            raise ValueError('invalid shared-parameter vector')
        count, columns = self._rotations
        if count:
            self._shift_buffer[:len(deltas)] = deltas
            library().pauli_bind_rotations(self.stages, count, *columns, self._shift_buffer)
        from ._reference import cached_gate_ptm, transfer, fsim
        for i in self._other:
            s = self.c.stages[i]
            theta = s.angle + s.scale * float(deltas[s.param])
            if s.name == 'RZZ':
                t, dt = cached_gate_ptm('RZZ', theta, True)
            elif s.name == 'FSIM':
                t, dt = transfer(*fsim(theta, s.phi)[:2])
            else:
                raise ValueError('unsupported parameter rebind')
            self._ptm[i, :t.size] = t.ravel()
            self._dptm[i, :dt.size] = s.scale * dt.ravel()
        return self

    def in_chunks(self, run, with_gradients=True):
        """Value and full gradient from kernels that carry at most 16 derivatives.

        run(n) evaluates with stage parameters renumbered to 0..n-1 and returns
        (out, extra) where out is [value, d_0 .. d_{n-1}]. Gradients come in
        passes of 16 parameters over the same bound PTMs: the stages of other
        parameters keep their bound matrices and act as fixed gates. Returns the
        value and extra of the first pass, and the concatenated gradient and
        each pass's extra.
        """
        n = self.c.nparams
        if n <= CHUNK:
            out, extra = run(n)
            return out, [extra]
        original = self.params.copy()
        values, extras = None, []
        try:
            # A value alone needs one pass with every parameterised stage fixed.
            for start in range(0, n, CHUNK) if with_gradients else [n]:
                width = min(CHUNK, n - start)
                inside = (original >= start) & (original < start + width)
                self.params[:] = np.where(inside, original - start, -1)
                out, extra = run(width)
                extras.append(extra)
                values = [out[0]] if values is None else values
                values.extend(out[1:1 + width] if with_gradients else [])
        finally:
            self.params[:] = original
        return values, extras

    def evaluate(self,observables,initial='zero',include_stats=False, *, gradients=True):
        if initial not in ('zero','plus'):
            raise ValueError('native engine currently supports zero/plus product state')
        obs=self.c.initial_terms(observables,gradients=gradients,initial=initial)
        t=(CTerm*len(obs))()
        for j,((x,z),coef) in enumerate(obs.items()):
            for i in range(4):
                t[j].x[i]=(x>>(64*i))&((1<<64)-1)
                t[j].z[i]=(z>>(64*i))&((1<<64)-1)
            t[j].coef=float(coef[0])
        kernel = library().run_pauli if gradients else library().run_pauli_value
        def run(n):
            res=(ct.c_double*(n+1 if gradients else 1))()
            stats=CStats()
            code=kernel(self.c.nqubits,n,len(self.stages),self.stages,
                        len(t),t,0 if initial=='zero' else 1,self.c.max_terms,res,ct.byref(stats))
            if code==-1:raise ConeOverflow(f'native exact Pauli expansion exceeds {self.c.max_terms} terms')
            if code:raise RuntimeError(f'native evaluation failed with code {code}')
            return list(res), stats
        res, extras = self.in_chunks(run, gradients)
        stats = extras[0]
        res = np.asarray(res, dtype=float)
        if include_stats:
            return float(res[0]),np.array(list(res)[1:]),{'max_terms':int(stats.peak_terms),
                     'final_terms':int(stats.final_terms),'stages':len(self.stages)}
        return float(res[0]),np.array(list(res)[1:])
