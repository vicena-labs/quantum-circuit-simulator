"""Bounded, exact (floating-point) backward Pauli propagation of local noisy circuits.

Requires NumPy for small local Pauli transfer matrices only. For n>64 uses
Python arbitrary-precision masks; no 2**n state allocation. The observable
and its complete shared-parameter gradient are propagated in one sweep.

Numerical zero cutoffs: PTM rows <= 5e-15 (absolute); coefficients <= 1e-16 times
the observable's l1 norm (relative, so results scale with the observable).
This is a research candidate, *not* a general-purpose noisy state simulator.
Allowed outputs: Pauli-sum expectations and gradients, not density matrices,
samples or arbitrary initial entangled states. Memory is bounded by max_terms.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import math
import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.diag([1.,-1.]).astype(complex)
P = (I,X,Z,Y)  # bit label x+2z: I,X,Z,Y
P2 = tuple(np.kron(P[b], P[a]) for b in range(4) for a in range(4))
GATE2 = {
    'CZ': np.diag([1.,1.,1.,-1.]).astype(complex),
    # Orientation follows the 4x4 matrix convention in this prototype; wire a is LSB.
    'ECR': np.array([[0,0,1,1j],[0,0,1j,1],[1,-1j,0,0],[-1j,1,0,0]], complex)/np.sqrt(2),
}
GATE1 = {'H': (X+Z)/np.sqrt(2), 'SX': (I+1j*I+X-1j*X)/2, 'X': X, 'Y': Y, 'Z': Z}


def transfer(unitary: np.ndarray, dunitary: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dual Pauli transfer matrix: row=output observable before conjugation."""
    d = unitary.shape[0]
    if unitary.shape != (d,d) or d not in (2,4):
        raise ValueError('only 1- and 2-qubit matrices supported')
    if not np.allclose(unitary.conj().T @ unitary, np.eye(d), atol=2e-12):
        raise ValueError('matrix is not unitary')
    basis = P if d == 2 else P2
    n = len(basis)
    out = np.empty((n,n), float)
    derivative = np.zeros((n,n), float)
    for j, p in enumerate(basis):
        o = unitary.conj().T @ p @ unitary
        dp = (dunitary.conj().T @ p @ unitary + unitary.conj().T @ p @ dunitary) if dunitary is not None else None
        for k, q in enumerate(basis):
            out[j,k] = np.trace(q @ o).real/d
            if dp is not None:
                derivative[j,k] = np.trace(q @ dp).real/d
    return out, derivative


def rotation(axis: str, theta: float) -> tuple[np.ndarray,np.ndarray]:
    if axis not in ('X','Y','Z'):
        raise ValueError('rotation axis must be X, Y or Z')
    p = {'X':X,'Y':Y,'Z':Z}[axis]
    c,s = math.cos(theta/2),math.sin(theta/2)
    return c*I - 1j*s*p, -0.5*s*I - 0.5j*c*p


def rzz(theta: float) -> tuple[np.ndarray,np.ndarray]:
    """exp(-i theta Z.Z / 2) and its derivative in theta; diagonal in the computational basis."""
    zz = np.array([1., -1., -1., 1.])
    return np.diag(np.exp(-0.5j*theta*zz)), np.diag(-0.5j*zz*np.exp(-0.5j*theta*zz))


def fsim(theta: float, phi: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    c,s=math.cos(theta),math.sin(theta)
    phase=np.exp(-1j*phi)
    u=np.array([[1,0,0,0],[0,c,-1j*s,0],[0,-1j*s,c,0],[0,0,0,phase]],complex)
    dt=np.array([[0,0,0,0],[0,-s,-1j*c,0],[0,-1j*c,-s,0],[0,0,0,0]],complex)
    df=np.zeros((4,4),complex);df[3,3]=-1j*phase
    return u,dt,df


@lru_cache(maxsize=2048)
def cached_gate_ptm(name:str,theta:float,parametric:bool):
    if name in GATE1: return transfer(GATE1[name])
    if name in GATE2: return transfer(GATE2[name])
    if name == 'RZZ':
        u, du = rzz(theta)
        matrix, derivative = transfer(u, du if parametric else None)
        matrix.flags.writeable = derivative.flags.writeable = False
        return matrix, derivative
    if name in ('RX','RY','RZ'):
        # Closed-form adjoint rotations avoid sixteen complex traces at every
        # new parameter angle. Order remains I, X, Z, Y throughout.
        first, second, sign = {'RX': (2, 3, 1.), 'RY': (1, 2, 1.),
                               'RZ': (1, 3, -1.)}[name]
        c, s = math.cos(theta), math.sin(theta)
        matrix = np.eye(4)
        matrix[first, first] = matrix[second, second] = c
        matrix[first, second] = sign*s
        matrix[second, first] = -sign*s
        derivative = np.zeros((4, 4))
        if parametric:
            derivative[first, first] = derivative[second, second] = -s
            derivative[first, second] = sign*c
            derivative[second, first] = -sign*c
        matrix.flags.writeable = derivative.flags.writeable = False
        return matrix, derivative
    raise ValueError('unknown gate')


@lru_cache(maxsize=512)
def depolarizing_ptm(p:float, n_wires:int):
    if not 0<=p<=1: raise ValueError('depolarizing probability must be in [0,1]')
    if n_wires==1: scale=1-4*p/3; size=4
    elif n_wires==2: scale=1-16*p/15; size=16
    else: raise ValueError('only 1/2 wire noise')
    arr=np.eye(size)*scale; arr[0,0]=1.
    return arr


@lru_cache(maxsize=512)
def relaxation_ptm(t:float, T1:float, T2:float) -> np.ndarray:
    if not (math.isfinite(t) and math.isfinite(T1) and math.isfinite(T2) and t>=0 and T1>0 and T2>0):
        raise ValueError('relaxation time constants must be finite, positive')
    if T2>2*T1*(1+1e-12):
        raise ValueError('T2 exceeds 2*T1, no nonnegative pure-dephasing rate')
    long=math.exp(-t/T1);trans=math.exp(-t/T2)
    a=np.diag([1.,trans,long,trans])
    a[2,0]=1-long  # E†(Z)=(1-gamma)Z+gamma I
    return a


def readout_z_ptm(p01:float,p10:float) -> np.ndarray:
    if not 0<=p01<=1 or not 0<=p10<=1:raise ValueError('readout probabilities must be in [0,1]')
    a=np.eye(4)
    a[2,0]=p10-p01; a[2,2]=1-p01-p10
    # This channel acts only on a diagonal Z readout, and should not be applied
    # to unrelated X/Y observables as a physical quantum channel.
    return a


def two_wire_product(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    # For bit-order local index=code(wire_a) + 4*code(wire_b)
    return np.kron(b,a)


@dataclass(frozen=True)
class Stage:
    wires: tuple[int,...]
    matrix: np.ndarray
    derivative: np.ndarray
    param: int=-1
    name: str=''
    angle: float | None = None
    phi: float | None = None
    scale: float = 1.0  # bound angle = angle + scale * shift[param]
    data: tuple = ()    # physical parameters for state simulators: a channel's numbers or a unitary


MAX_PARAMETERS = 4096
# Terms per observable cone. The default keeps an accidental exact run to about
# 0.5 GB; beyond it the limit is memory, at about 0.5 KB per term at peak
# (measured on 127 qubits, including the hash table's headroom and rehashing).
DEFAULT_MAX_TERMS = 1_000_000
MAX_TERMS = 1_000_000_000


class ConeOverflow(RuntimeError):
    """Unsupported exact Pauli expansion within the declared term budget."""


class NoisyPauliCircuit:
    def __init__(self,nqubits:int,nparams:int=0,max_terms:int=DEFAULT_MAX_TERMS):
        if not 1<=nqubits<=16384 or not 0<=nparams<=MAX_PARAMETERS or max_terms<1:
            raise ValueError(f'nqubits 1..16384, nparams 0..{MAX_PARAMETERS}, max_terms>=1')
        self.nqubits,self.nparams,self.max_terms=nqubits,nparams,max_terms
        self.stages:list[Stage]=[]
        self.readout:dict[int,tuple[float,float]]={}

    def _stage(self,wires,matrix,derivative=None,param=-1,name='',angle=None,phi=None,scale=1.,data=()):
        wires=tuple(wires)
        if not 1<=len(wires)<=2 or len(set(wires))!=len(wires) or any(not 0<=w<self.nqubits for w in wires):
            raise ValueError('invalid wires')
        d=4**len(wires)
        matrix=np.asarray(matrix,dtype=float)
        if matrix.shape!=(d,d) or not np.all(np.isfinite(matrix)):
            raise ValueError('invalid PTM')
        if derivative is None: derivative=np.zeros_like(matrix)
        derivative=np.asarray(derivative,dtype=float)
        if derivative.shape !=(d,d) or not np.all(np.isfinite(derivative)):
            raise ValueError('invalid PTM derivative')
        if param!=-1 and not 0<=param<self.nparams:raise ValueError('invalid parameter')
        if not math.isfinite(scale) or (param!=-1 and scale==0.):raise ValueError('invalid parameter scale')
        self.stages.append(Stage(wires,matrix,derivative*scale,param,name,angle,phi,float(scale),data))
        return self

    def gate(self,name:str,wires:Sequence[int],theta:float=0.,param:int=-1,scale:float=1.):
        name=name.upper();wires=tuple(wires)
        if name in GATE1 and len(wires)==1 and param==-1:
            U=GATE1[name];dU=None
        elif name in GATE2 and len(wires)==2 and param==-1:
            U=GATE2[name];dU=None
        elif name in ('RX','RY','RZ') and len(wires)==1:
            U,dU=rotation(name[1],theta)
        elif name=='FSIM' and len(wires)==2:
            U,dU,_=fsim(theta,0.)
        elif name=='RZZ' and len(wires)==2:
            U,dU=rzz(theta)
        else:
            raise ValueError(f'unsupported gate {name} or wires/parameter')
        t,d=cached_gate_ptm(name,theta,param!=-1)
        # Fixed rotations keep their angle too, so a circuit can be read back.
        return self._stage(wires,t,d,param,name,theta if name in ('RX','RY','RZ','RZZ','FSIM') else None,scale=scale)

    def unitary(self,wires:Sequence[int],matrix):
        """A fixed one- or two-qubit unitary; wire a is the least significant bit."""
        u=np.asarray(matrix,dtype=complex)
        t,_=transfer(u)
        return self._stage(wires,t,name='UNITARY',data=(u,))

    def fsim(self,wires:Sequence[int],theta:float,phi:float,param:int=-1,phi_param:int=-1,scale:float=1.):
        U,dt,df=fsim(theta,phi)
        # Two independently variable angles are represented by successive tangent-stage
        # derivatives of the same PTM, with no physical evolution between them.
        # Use a direct two-parameter API later; disallow separate parameterization
        # rather than delivering incorrect chain rules.
        if phi_param!=-1:raise ValueError('phi gradient unsupported in this prototype')
        t,d=transfer(U,dt if param!=-1 else None)
        return self._stage(wires,t,d,param,'FSIM',theta,phi,scale)

    def channel_pauli(self,wire:int,axis:str,probability:float):
        if not 0<=probability<=1:raise ValueError('invalid probability')
        axis=axis.upper(); p={'X':X,'Y':Y,'Z':Z}.get(axis)
        if p is None:raise ValueError('invalid Pauli')
        t=(1-probability)*np.eye(4)+probability*transfer(p)[0]
        return self._stage((wire,),t,name=f'{axis}_ERROR',data=(probability,))

    def channel_depolarize(self,wires:Sequence[int],probability:float):
        return self._stage(wires,depolarizing_ptm(probability,len(wires)),name='DEPOLARIZE',data=(probability,))

    def channel_relax(self,wire:int,duration:float,T1:float,T2:float):
        return self._stage((wire,),relaxation_ptm(duration,T1,T2),name='T1_T2',data=(duration,T1,T2))

    def set_readout(self,wire:int,p01:float,p10:float):
        if not 0<=wire<self.nqubits or not 0<=p01<=1 or not 0<=p10<=1:raise ValueError('invalid readout')
        self.readout[wire]=(p01,p10)
        return self

    def shifted(self,deltas:Sequence[float]):
        """Return a prepared circuit with all shared parameters shifted by deltas.

        Noise/readout matrices and static geometry are reused unchanged; only
        parameter-dependent gate PTMs are replaced. No cache of *results*.
        """
        if len(deltas)!=self.nparams or not all(math.isfinite(x) for x in deltas):
            raise ValueError('one finite shift per shared parameter required')
        out=NoisyPauliCircuit(self.nqubits,self.nparams,self.max_terms)
        out.readout=self.readout.copy()
        for st in self.stages:
            if st.param<0:out.stages.append(st);continue
            theta=st.angle+st.scale*float(deltas[st.param])
            if st.name in ('RX','RY','RZ','RZZ'):
                mat,der=cached_gate_ptm(st.name,theta,True)
            elif st.name=='FSIM':
                u,du,_=fsim(theta,st.phi)
                mat,der=transfer(u,du)
            else:raise ValueError('unsupported parameter rebind')
            out.stages.append(Stage(st.wires,mat,der*st.scale,st.param,st.name,theta,st.phi,st.scale,st.data))
        return out

    @staticmethod
    def mask(paulis:Sequence[tuple[int,str]]) -> tuple[int,int]:
        x=z=0
        for q,axis in paulis:
            if axis not in ('I','X','Y','Z') or q<0:raise ValueError('invalid Pauli')
            if axis in ('X','Y'):x|=1<<q
            if axis in ('Y','Z'):z|=1<<q
        return (x,z)

    def initial_terms(self,observables, *, gradients=True, initial=None, coefficient_cutoff=1e-16):
        # observables iterable [(coefficient, {wire:'Z',...}),...]
        touched = {q for stage in self.stages for q in stage.wires}
        terms={}
        for coefficient,ops in observables:
            key=self.mask(tuple(ops.items()))
            if (key[0] | key[1])>>self.nqubits: raise ValueError('observable wire out of range')
            if not math.isfinite(coefficient):raise ValueError('nonfinite observable coefficient')
            if initial in ('zero', 'plus'):
                coefficient, ops = _idle_observable((coefficient, ops), touched, self.readout, initial)
                key = self.mask(tuple(ops.items()))
            if coefficient == 0.:
                continue
            v=terms.setdefault(key,np.zeros(self.nparams+1 if gradients else 1))
            v[0]+=coefficient
        cutoff = coefficient_cutoff * _observable_norm(terms)
        for wire,(p01,p10) in self.readout.items():
            if initial in ('zero', 'plus') and wire not in touched:
                continue
            terms=_transform(terms,Stage((wire,),readout_z_ptm(p01,p10),np.zeros((4,4)),-1,'READOUT'),self.nparams if gradients else 0,self.max_terms,gradients=gradients,coefficient_cutoff=cutoff)
        return terms

    def evaluate(self,observables,initial:str='zero',initial_bloch=None,include_stats:bool=False, *, gradients=True):
        if initial not in ('zero','plus','bloch'):
            raise ValueError('initial must be zero, plus or bloch')
        if initial=='bloch' and (initial_bloch is None or len(initial_bloch)!=self.nqubits):
            raise ValueError('initial_bloch must contain one Bloch vector per qubit')
        terms=self.initial_terms(observables,gradients=gradients,initial=initial)
        # A wire can be contracted as soon as backward propagation reaches its
        # first forward operation. No earlier channel can touch that wire again.
        first_touch = {}
        for index, stage in enumerate(self.stages):
            for wire in stage.wires:
                first_touch.setdefault(wire, index)
        boundary = initial if initial in ('zero', 'plus') else None
        if boundary:
            untouched = sum(1 << q for q in range(self.nqubits) if q not in first_touch)
            terms = _contract(terms, untouched, boundary)
        peak=len(terms)
        cutoff = 1e-16 * _observable_norm(terms)
        for index in range(len(self.stages)-1, -1, -1):
            stage = self.stages[index]
            retiring = sum(1 << q for q in stage.wires if first_touch[q] == index) if boundary else 0
            terms=_transform(terms,stage,self.nparams if gradients else 0,self.max_terms,
                             gradients=gradients,retiring=retiring,initial=boundary,
                             coefficient_cutoff=cutoff)
            peak=max(peak,len(terms))
        ans=np.zeros(self.nparams+1 if gradients else 1)
        for (x,z),coeff in terms.items():
            if initial=='zero': factor=1 if x==0 else 0
            elif initial=='plus':factor=1 if z==0 else 0
            else:
                factor=1.
                a,b=x,z
                while a or b:
                    bit=(a|b)&-(a|b);q=bit.bit_length()-1
                    px,py,pz=initial_bloch[q]
                    factor*= (py if a&bit and b&bit else px if a&bit else pz)
                    if abs(factor)<1e-300:break
                    a^=a&bit;b^=b&bit
            if factor:ans+=coeff*factor
        if include_stats:
            return float(ans[0]),ans[1:].copy(),{'max_terms':peak,'final_terms':len(terms),'stages':len(self.stages)}
        return float(ans[0]),ans[1:].copy()


def _observable_norm(terms):
    """l1 norm of the value coefficients; the scale for relative roundoff filtering."""
    return math.fsum(abs(float(value[0])) for value in terms.values())


def _idle_observable(term, touched, readout, initial='zero'):
    """Contract idle product inputs, including readout, without I/Z expansion."""
    coefficient, ops = term
    active = {}
    for wire, axis in ops.items():
        if wire in touched:
            active[wire] = axis
        elif axis == 'I':
            continue
        elif axis == 'Z' and wire in readout:
            p01, p10 = readout[wire]
            coefficient *= 1-2*p01 if initial == 'zero' else p10-p01
        elif axis != ('Z' if initial == 'zero' else 'X'):
            coefficient = 0.
    return coefficient, active


def _contract(terms, retiring, initial):
    """Contract and merge known product-input factors before the next cap check."""
    out = {}
    for (x, z), value in terms.items():
        if (x if initial == 'zero' else z) & retiring:
            continue
        key = (x & ~retiring, z & ~retiring)
        if key in out:
            out[key] += value
        else:
            out[key] = value.copy()
    return out


def _transform(terms:dict, stage:Stage,nparams:int,limit:int, *, gradients=True, retiring=0, initial=None, coefficient_cutoff=1e-16):
    wires=stage.wires;k=len(wires)
    occupied=0
    for q in wires:occupied|=1<<q
    # Cache per local input row; preserve all coefficients above FP-roundoff noise.
    rows=[[(j,float(v),float(stage.derivative[i,j]) if gradients else 0.) for j,v in enumerate(stage.matrix[i])
           if abs(v)>5e-15 or (gradients and abs(stage.derivative[i,j])>5e-15)] for i in range(4**k)]
    new={}
    for (x,z),arr in terms.items():
        li=0
        for h,q in enumerate(wires):li+=(((x>>q)&1)+2*((z>>q)&1))<<(2*h)
        for lo,t,dt in rows[li]:
            xx=x&~occupied;zz=z&~occupied
            for h,q in enumerate(wires):
                v=(lo>>(2*h))&3
                if v&1:xx|=1<<q
                if v&2:zz|=1<<q
            if retiring:
                if (xx if initial == 'zero' else zz) & retiring:
                    continue
                xx &= ~retiring
                zz &= ~retiring
            key=(xx,zz)
            if key not in new:
                if len(new)>=limit:raise ConeOverflow(f'exact Pauli expansion requires >{limit} terms at {stage.name}')
                new[key]=np.zeros(nparams+1)
            dst=new[key]
            dst+=t*arr
            if gradients and stage.param!=-1:dst[stage.param+1]+=dt*arr[0]
    return {k:v for k,v in new.items() if np.max(np.abs(v))>coefficient_cutoff}


def load_chip_grid(rows:int,cols:int,depth:int=2,
                   gate:str='CZ',p2:float=.005,t:float=2e-7,T1:float=70e-6,T2:float=90e-6,
                   readout:bool=True,nparams:int|None=None,max_terms:int=DEFAULT_MAX_TERMS,active_qubits:int|None=None):
    """Illustrative 2D brickwork; geometry and parameters are NOT chip calibration files."""
    if rows<1 or cols<1 or depth<0:raise ValueError('invalid grid')
    total=rows*cols
    if active_qubits is None:active_qubits=total
    if not 1<=active_qubits<=total:raise ValueError('active_qubits must fit the grid')
    circuit=NoisyPauliCircuit(active_qubits,depth if nparams is None else nparams,max_terms)
    # Only one global parameter per layer; this is a synthetic hardware-inspired workload.
    for layer in range(depth):
        # prepare non-diagonal support without entangling every wire around the readout
        for q in range(active_qubits):
            circuit.gate('RY',(q,),theta=float(0.21+.06*(layer%3)),param=layer)
        edges=[]
        # alternating checkerboard matchings, not all-coupler layers
        if layer%2==0:
            for r in range(rows):
                for c in range((layer//2)%2,cols-1,2):edges.append((r*cols+c,r*cols+c+1))
        else:
            for r in range((layer//2)%2,rows-1,2):
                for c in range(cols):edges.append((r*cols+c,(r+1)*cols+c))
        for a,b in edges:
            if a>=active_qubits or b>=active_qubits:continue
            if gate=='FSIM':circuit.fsim((a,b),.11, .23)
            else:circuit.gate(gate,(a,b))
            circuit.channel_depolarize((a,b),p2)
            circuit.channel_relax(a,t,T1,T2)
            circuit.channel_relax(b,t,T1,T2)
    if readout:
        for q in range(active_qubits): circuit.set_readout(q,.02,.014)
    return circuit
