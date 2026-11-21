"""Circuits and observables from OpenQASM, Qiskit and plain text.

These converters let existing circuits run on Quantum Circuit Simulator without rewriting
them. Every gate is decomposed exactly into the native set (H, SX, X, Y, Z, RX,
RY, RZ, CZ, ECR); global phases are dropped, since they do not change an
expectation value. A gate outside the table is decomposed by Qiskit's
transpiler when Qiskit is installed.

A Qiskit Parameter becomes one shared parameter when every angle
it enters is linear in it, `k*p + b`, so one prepared plan serves every value
of it. A circuit whose angles combine several parameters or are nonlinear in
one (for example `p*q` or `sin(p)`), or has more than 4096 parameters, needs
numeric values and is rebuilt for each binding.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field

from .api import Circuit

NATIVE_ONE = {"h", "sx", "x", "y", "z"}
ROTATIONS = {"rx", "ry", "rz"}
# Gates the converter decomposes itself; Qiskit transpiles anything else to these.
SUPPORTED_GATES = sorted(NATIVE_ONE | ROTATIONS | {
    "id", "sxdg", "s", "sdg", "t", "tdg", "p", "u1", "u2", "u3", "u", "r", "cx", "cy", "cz",
    "ecr", "swap", "rzz", "rxx", "ryy", "rzx", "cp", "cu1", "crx", "cry", "crz", "ch", "ccx", "cswap"})
_IGNORED = {"barrier", "delay", "id", "i"}
GATE_END = "END"   # marker op: a multi-qubit source gate ends here (no operation of its own)
MAX_SHARED_PARAMETERS = 4096


class ConversionError(ValueError):
    """A circuit or observable that Quantum Circuit Simulator cannot represent exactly."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


class _NotShareable(Exception):
    pass


@dataclass(frozen=True)
class Angle:
    """A linear angle `base + scale * parameter` for circuits built in Python.

    Programs use it where a Qiskit ParameterExpression would go, so generated
    circuits (Trotter steps, QAOA, ansatze) keep shared parameters through
    conversion and compilation without needing Qiskit.
    """
    parameter: str
    scale: float = 1.0
    base: float = 0.0

    def __neg__(self):
        return Angle(self.parameter, -self.scale, -self.base)

    def __mul__(self, k):
        return Angle(self.parameter, self.scale * float(k), self.base * float(k))

    __rmul__ = __mul__

    def __truediv__(self, k):
        return self * (1.0 / float(k))

    def __add__(self, b):
        return Angle(self.parameter, self.scale, self.base + float(b))

    __radd__ = __add__


@dataclass
class Program:
    """A circuit as instructions in the converters' gate vocabulary.

    instructions are (gate name, qubits, angles), with angles numbers or
    Angle objects; parameters lists the parameter names in the order a
    sequence of values follows. Generators in quantum_circuit_simulator.hamiltonians and
    quantum_circuit_simulator.algorithms return Programs; every entry point that accepts
    OpenQASM or a Qiskit circuit accepts one.
    """
    n_qubits: int
    instructions: list
    parameters: list = field(default_factory=list)

    def bind(self, values):
        """A parameter-free copy with numeric angles, from a mapping or a sequence."""
        if not isinstance(values, dict):
            values = dict(zip(self.parameters, values, strict=True))
        def number(a):
            return a.base + a.scale * float(values[a.parameter]) if isinstance(a, Angle) else a
        return Program(self.n_qubits, [(g, q, [number(a) for a in angles]) for g, q, angles in self.instructions])

    def to_qasm(self, values=None):
        """OpenQASM 2 text; parameters need values, since OpenQASM 2 has none."""
        program = self.bind(values) if self.parameters else self
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{self.n_qubits}];"]
        for name, qubits, angles in program.instructions:
            args = f"({','.join(repr(float(a)) for a in angles)})" if angles else ""
            lines.append(f"{name}{args} " + ",".join(f"q[{q}]" for q in qubits) + ";")
        return "\n".join(lines) + "\n"


@dataclass
class Noise:
    """A uniform noise model applied after every converted gate.

    depolarizing_1q and depolarizing_2q are total probabilities of a
    non-identity Pauli error after each one- and two-qubit gate of the source
    circuit (not after each primitive of its decomposition). readout is a
    (p01, p10) pair applied to every wire, or a mapping of wire to pair.
    """
    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    readout: object = None

    @classmethod
    def coerce(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            unknown = set(value) - {"depolarizing_1q", "depolarizing_2q", "readout"}
            if unknown:
                raise ConversionError(f"unknown noise fields {sorted(unknown)}",
                                      "use depolarizing_1q, depolarizing_2q and readout")
            return cls(**value)
        raise ConversionError("noise must be a Noise or a mapping")


@dataclass
class Converted:
    """A `Circuit` and how its shared parameters map to the source."""
    circuit: Circuit
    parameters: list = field(default_factory=list)  # source parameters in shared-index order
    source_order: list | None = None  # the source's own parameter order, for sequences of values

    def shifts(self, values):
        """Shift vector from a mapping {parameter or name: value}, or a sequence in
        the source's parameter order (Qiskit's circuit.parameters)."""
        if values is None:
            values = {}
        if isinstance(values, dict):
            by_name = {getattr(k, "name", k): v for k, v in values.items()}
            missing = [getattr(p, "name", p) for p in self.parameters if getattr(p, "name", p) not in by_name]
            if missing:
                raise ConversionError(f"missing values for parameters {missing}")
            return [float(by_name[getattr(p, "name", p)]) for p in self.parameters]
        values = [float(v) for v in values]
        order = self.source_order if self.source_order is not None else self.parameters
        if len(values) != len(order):
            raise ConversionError(f"expected {len(order)} parameter values, got {len(values)}")
        given = dict(zip(order, values))
        return [given[p] for p in self.parameters]


# ---------------------------------------------------------------- circuits
def from_qiskit(circuit, *, values=None, noise=None, max_terms=1_000_000):
    """Convert a Qiskit QuantumCircuit.

    With values=None, Qiskit Parameters become shared parameters when they
    can (see the module docstring); pass values (a mapping or a sequence in
    circuit.parameters order) to bind them numerically instead.
    """
    try:
        import qiskit  # noqa: F401
    except ImportError as exc:
        raise ConversionError("Qiskit is not installed", "pip install qiskit, or pass OpenQASM 2 text") from exc
    if values is not None:
        circuit = circuit.assign_parameters(_binding(circuit, values))
    if any(i.operation.name not in SUPPORTED_GATES and i.operation.name not in _IGNORED
           and i.operation.name != "measure" for i in circuit.data):
        from qiskit import transpile
        circuit = transpile(circuit, basis_gates=SUPPORTED_GATES, optimization_level=0)
    instructions = []
    for item in circuit.data:
        op = item.operation
        if getattr(op, "condition", None) is not None:
            raise ConversionError("classically conditioned operations are not supported")
        instructions.append((op.name, [circuit.find_bit(q).index for q in item.qubits], list(op.params)))
    converted = _build(circuit.num_qubits, instructions, Noise.coerce(noise), max_terms)
    converted.source_order = list(circuit.parameters)
    return converted


def source_instructions(circuit, values=None):
    """(n_qubits, instructions, source parameter order) from OpenQASM text or a Qiskit circuit.

    Instructions are (name, qubits, parameters) in the converters' gate vocabulary,
    the form the compiler routes before lowering.
    """
    if isinstance(circuit, Program):
        program = circuit.bind(values) if values is not None and circuit.parameters else circuit
        return program.n_qubits, list(program.instructions), list(program.parameters) or None
    if isinstance(circuit, str):
        if re.search(r"OPENQASM\s+3", circuit):
            from qiskit import qasm3
            circuit = qasm3.loads(circuit)
        else:
            n, instructions = _parse_qasm2(circuit)
            return n, instructions, None
    if values is not None:
        circuit = circuit.assign_parameters(_binding(circuit, values))
    if any(i.operation.name not in SUPPORTED_GATES and i.operation.name not in _IGNORED
           and i.operation.name != "measure" for i in circuit.data):
        from qiskit import transpile
        circuit = transpile(circuit, basis_gates=SUPPORTED_GATES, optimization_level=0)
    instructions = []
    for item in circuit.data:
        op = item.operation
        if getattr(op, "condition", None) is not None:
            raise ConversionError("classically conditioned operations are not supported")
        instructions.append((op.name, [circuit.find_bit(q).index for q in item.qubits], list(op.params)))
    return circuit.num_qubits, instructions, list(circuit.parameters)


def _binding(circuit, values):
    params = list(circuit.parameters)
    if isinstance(values, dict):
        by_name = {getattr(k, "name", k): v for k, v in values.items()}
        return {p: float(by_name[p.name]) for p in params}
    values = list(values)
    if len(values) != len(params):
        raise ConversionError(f"expected {len(params)} parameter values, got {len(values)}")
    return dict(zip(params, map(float, values)))


def from_qasm(text, *, noise=None, max_terms=1_000_000):
    """Convert OpenQASM. OpenQASM 2 needs no dependencies; OpenQASM 3 needs Qiskit."""
    if re.search(r"OPENQASM\s+3", text):
        try:
            from qiskit import qasm3
        except ImportError as exc:
            raise ConversionError("OpenQASM 3 needs Qiskit", "pip install qiskit qiskit-qasm3-import") from exc
        return from_qiskit(qasm3.loads(text), noise=noise, max_terms=max_terms)
    n, instructions = _parse_qasm2(text)
    return _build(n, instructions, Noise.coerce(noise), max_terms)


def _build(n, instructions, noise, max_terms):
    """Decompose source instructions into native ops, mapping shared parameters."""
    ops, params = _lower(instructions, noise)
    return _emit(n, ops, params, noise.readout if noise else None, max_terms)


def _lower(instructions, noise=None):
    """Native ops (prim, wires, base, parameter index, scale) and the shared parameters.

    prim is a native gate name, or "DEPOL" with its probability as base when
    `noise` adds gate errors. The compiler routes these ops before emitting.
    """
    measured = set()
    ops, params = [], []
    for name, qubits, angles in instructions:
        name = name.lower()
        if name in _IGNORED:
            continue
        if name == "measure":
            measured.update(qubits)
            continue
        if name == "reset":
            raise ConversionError("reset is not supported", "remove resets; Quantum Circuit Simulator starts from |0...0>")
        if measured.intersection(qubits):
            raise ConversionError("a gate acts after a measurement on the same qubit",
                                  "mid-circuit measurement is not supported; keep measurements at the end")
        if name not in _DECOMPOSE:
            raise ConversionError(f"unsupported gate {name!r}",
                                  "install Qiskit so unknown gates are transpiled, or use: " + ", ".join(SUPPORTED_GATES))
        for prim, wires, angle in _DECOMPOSE[name](qubits, angles):
            ops.append((prim, list(wires), angle))
        if len(qubits) >= 2:
            # Marks the end of a multi-qubit source gate: the compiler applies the gate's error here,
            # after the one-qubit frame changes of its decomposition, as for a native gate.
            ops.append((GATE_END, list(qubits), None))
        if noise and len(qubits) == 1 and noise.depolarizing_1q:
            ops.append(("DEPOL", list(qubits), noise.depolarizing_1q))
        if noise and len(qubits) == 2 and noise.depolarizing_2q:
            ops.append(("DEPOL", list(qubits), noise.depolarizing_2q))
    # Map symbolic angles to shared parameters, or fail with a clear request for values.
    resolved = []
    for prim, wires, angle in ops:
        if prim in ("RX", "RY", "RZ"):
            try:
                base, source, scale = _angle(angle)
            except _NotShareable:
                raise ConversionError(f"angle {angle} is not linear in one parameter (k*p + b)",
                                      "pass numeric values for the parameters (values=...)") from None
            index = -1
            if source is not None:
                if source not in params:
                    params.append(source)
                index = params.index(source)
            resolved.append((prim, wires, base, index, scale))
        else:
            resolved.append((prim, wires, angle, -1, 1.0))
    if len(params) > MAX_SHARED_PARAMETERS:
        raise ConversionError(f"{len(params)} parameters exceed the {MAX_SHARED_PARAMETERS} shared parameters",
                              "pass numeric values (values=...); the circuit is then rebuilt for each binding")
    return resolved, params


def _emit(n, ops, params, readout, max_terms):
    """A `Circuit` from native ops. "RELAX" ops carry (duration, t1, t2) as base."""
    c = Circuit(n, len(params), max_terms)
    for prim, wires, base, index, scale in ops:
        if prim == GATE_END:
            continue
        if prim == "DEPOL":
            c.depolarize(wires, base)
        elif prim == "RELAX":
            c.relax(wires[0], *base)
        elif prim in ("RX", "RY", "RZ"):
            c.gate(prim, wires, base, index, scale if index >= 0 else 1.0)
        else:
            c.gate(prim, wires)
    if readout is not None:
        pairs = readout if isinstance(readout, dict) else {q: readout for q in range(n)}
        for q, (p01, p10) in pairs.items():
            c.readout(int(q), p01, p10)
    return Converted(c, params)


def _angle(value):
    """(base, parameter or None, scale) of an angle that is numeric or k*p + b."""
    if isinstance(value, (int, float)):
        return float(value), None, 1
    if isinstance(value, Angle):
        return value.base, value.parameter, value.scale
    free = getattr(value, "parameters", None)
    if not free:
        return float(value), None, 1
    if len(free) != 1:
        raise _NotShareable
    (p,) = free
    slope = value.gradient(p)
    if getattr(slope, "parameters", None) or float(slope) == 0:
        raise _NotShareable
    return float(value.bind({p: 0.0})), p, float(slope)


def _neg(angle):
    return -angle


def _half(angle):
    if isinstance(angle, Angle) or getattr(angle, "parameters", None):
        return angle / 2
    return float(angle) / 2


HALF_PI, QUARTER_PI = math.pi / 2, math.pi / 4


def _cx(c, t):
    return [("H", [t], None), ("CZ", [c, t], None), ("H", [t], None)]


def _rzz(a, b, theta):
    return _cx(a, b) + [("RZ", [b], theta)] + _cx(a, b)


def _rzx(a, b, theta):
    return [("H", [b], None)] + _rzz(a, b, theta) + [("H", [b], None)]


_DECOMPOSE = {
    **{g: (lambda g: lambda q, a: [(g.upper(), q, None)])(g) for g in NATIVE_ONE},
    **{g: (lambda g: lambda q, a: [(g.upper(), q, a[0])])(g) for g in ROTATIONS},
    "sxdg": lambda q, a: [("RX", q, -HALF_PI)],
    "s": lambda q, a: [("RZ", q, HALF_PI)],
    "sdg": lambda q, a: [("RZ", q, -HALF_PI)],
    "t": lambda q, a: [("RZ", q, QUARTER_PI)],
    "tdg": lambda q, a: [("RZ", q, -QUARTER_PI)],
    "p": lambda q, a: [("RZ", q, a[0])],
    "u1": lambda q, a: [("RZ", q, a[0])],
    # U(theta, phi, lambda) = RZ(phi) RY(theta) RZ(lambda), up to a global phase.
    "u": lambda q, a: [("RZ", q, a[2]), ("RY", q, a[0]), ("RZ", q, a[1])],
    "u3": lambda q, a: [("RZ", q, a[2]), ("RY", q, a[0]), ("RZ", q, a[1])],
    "u2": lambda q, a: [("RZ", q, a[1]), ("RY", q, HALF_PI), ("RZ", q, a[0])],
    # R(theta, phi) = RZ(phi) RX(theta) RZ(-phi).
    "r": lambda q, a: [("RZ", q, _neg(a[1])), ("RX", q, a[0]), ("RZ", q, a[1])],
    "cx": lambda q, a: _cx(q[0], q[1]),
    # CY = (I x S) CX (I x S^dagger).
    "cy": lambda q, a: [("RZ", [q[1]], -HALF_PI)] + _cx(q[0], q[1]) + [("RZ", [q[1]], HALF_PI)],
    "cz": lambda q, a: [("CZ", q, None)],
    # Qiskit's ECR = RZX(pi/4), X on the first qubit, RZX(-pi/4); decomposed so
    # the result does not depend on the native ECR's wire convention.
    "ecr": lambda q, a: _rzx(q[0], q[1], QUARTER_PI) + [("X", [q[0]], None)] + _rzx(q[0], q[1], -QUARTER_PI),
    "swap": lambda q, a: _cx(q[0], q[1]) + _cx(q[1], q[0]) + _cx(q[0], q[1]),
    "rzz": lambda q, a: _rzz(q[0], q[1], a[0]),
    "rxx": lambda q, a: [("H", [q[0]], None), ("H", [q[1]], None)] + _rzz(q[0], q[1], a[0])
                        + [("H", [q[0]], None), ("H", [q[1]], None)],
    # RX(pi/2) maps Y to Z under conjugation: RX(pi/2)^dag Y RX(pi/2) = Z.
    "ryy": lambda q, a: [("RX", [q[0]], HALF_PI), ("RX", [q[1]], HALF_PI)] + _rzz(q[0], q[1], a[0])
                        + [("RX", [q[0]], -HALF_PI), ("RX", [q[1]], -HALF_PI)],
    "rzx": lambda q, a: _rzx(q[0], q[1], a[0]),
    # CP(l) = RZ(l/2) on control, CX, RZ(-l/2) on target, CX, RZ(l/2) on target.
    "cp": lambda q, a: [("RZ", [q[0]], _half(a[0]))] + _cx(q[0], q[1]) + [("RZ", [q[1]], _neg(_half(a[0])))]
                       + _cx(q[0], q[1]) + [("RZ", [q[1]], _half(a[0]))],
}
_DECOMPOSE["cu1"] = _DECOMPOSE["cp"]


def _crz(c, t, theta):
    return [("RZ", [t], _half(theta))] + _cx(c, t) + [("RZ", [t], _neg(_half(theta)))] + _cx(c, t)


def _ccx(a, b, t):
    """Toffoli in six CX and T gates (Nielsen and Chuang, Fig. 4.9)."""
    T, TD = ("RZ", QUARTER_PI), ("RZ", -QUARTER_PI)
    def one(gate, q):
        return [(gate[0], [q], gate[1])]
    return ([("H", [t], None)] + _cx(b, t) + one(TD, t) + _cx(a, t) + one(T, t) + _cx(b, t) + one(TD, t)
            + _cx(a, t) + one(T, b) + one(T, t) + [("H", [t], None)] + _cx(a, b) + one(T, a) + one(TD, b)
            + _cx(a, b))


_DECOMPOSE.update({
    "crz": lambda q, a: _crz(q[0], q[1], a[0]),
    "crx": lambda q, a: [("H", [q[1]], None)] + _crz(q[0], q[1], a[0]) + [("H", [q[1]], None)],
    "cry": lambda q, a: [("RY", [q[1]], _half(a[0]))] + _cx(q[0], q[1]) + [("RY", [q[1]], _neg(_half(a[0])))]
                        + _cx(q[0], q[1]),
    # CH = RY(pi/4) CZ RY(-pi/4) on the target, up to global phase; applied right to left.
    "ch": lambda q, a: [("RY", [q[1]], -QUARTER_PI), ("CZ", q, None), ("RY", [q[1]], QUARTER_PI)],
    "ccx": lambda q, a: _ccx(*q),
    "cswap": lambda q, a: _cx(q[2], q[1]) + _ccx(q[0], q[1], q[2]) + _cx(q[2], q[1]),
})


# ---------------------------------------------------------------- OpenQASM 2
_FUNCTIONS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "ln": math.log, "sqrt": math.sqrt}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}


def _evaluate(text, env):
    """Evaluate an OpenQASM 2 parameter expression without eval()."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id in env:
                return env[node.id]
            raise ConversionError(f"unknown name {node.id!r} in expression {text!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS \
                and len(node.args) == 1:
            return _FUNCTIONS[node.func.id](walk(node.args[0]))
        raise ConversionError(f"unsupported expression {text!r}")
    try:
        return walk(ast.parse(text.replace("^", "**"), mode="eval"))
    except SyntaxError as exc:
        raise ConversionError(f"invalid expression {text!r}") from exc


def _split_top(text, sep=","):
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


_APPLY = re.compile(r"^([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s+(.+)$", re.S)


def _parse_qasm2(text):
    """A dependency-free OpenQASM 2.0 reader for qelib1 gates and user gate definitions."""
    text = re.sub(r"//[^\n]*", "", text)
    registers, size, definitions, instructions = {}, 0, {}, []
    # Gate definitions contain braces; pull them out before splitting on ';'.
    gate_def = re.compile(r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s+([^{]+)\{([^}]*)\}", re.S)
    for name, params, qargs, body in gate_def.findall(text):
        definitions[name] = ([p.strip() for p in params.split(",") if p.strip()],
                             [q.strip() for q in qargs.split(",")], body)
    text = gate_def.sub("", text)

    def qubits_of(arg):
        m = re.fullmatch(r"(\w+)\s*\[\s*(\d+)\s*\]", arg)
        if m:
            reg, index = m.group(1), int(m.group(2))
            if reg not in registers or index >= registers[reg][1]:
                raise ConversionError(f"qubit {arg} is not declared")
            return [registers[reg][0] + index]
        if arg in registers:
            start, length = registers[arg]
            return list(range(start, start + length))
        raise ConversionError(f"unknown qubit argument {arg!r}")

    def apply(name, params, args, depth=0):
        if name in definitions:
            if depth > 20:
                raise ConversionError("gate definitions nest too deeply")
            names, qnames, body = definitions[name]
            env = dict(zip(names, params))
            for statement in body.split(";"):
                statement = statement.strip()
                if not statement or statement.startswith("barrier"):
                    continue
                m = _APPLY.match(statement)
                if not m:
                    raise ConversionError(f"cannot parse {statement!r} in gate {name}")
                inner = [_evaluate(e, env) for e in _split_top(m.group(2) or "")]
                inner_args = [args[qnames.index(q.strip())] for q in _split_top(m.group(3))]
                apply(m.group(1), inner, inner_args, depth + 1)
            return
        instructions.append((name, args, params))

    for statement in text.split(";"):
        statement = statement.strip()
        if not statement or statement.startswith(("OPENQASM", "include")):
            continue
        if statement.startswith("qreg"):
            m = re.fullmatch(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", statement)
            if not m:
                raise ConversionError(f"cannot parse {statement!r}")
            registers[m.group(1)] = (size, int(m.group(2)))
            size += int(m.group(2))
            continue
        if statement.startswith(("creg", "barrier")):
            continue
        if statement.startswith(("opaque", "if")):
            raise ConversionError(f"{statement.split()[0]} is not supported")
        if statement.startswith("measure"):
            m = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", statement)
            if not m:
                raise ConversionError(f"cannot parse {statement!r}")
            instructions.append(("measure", qubits_of(m.group(1).strip()), []))
            continue
        if statement.startswith("reset"):
            instructions.append(("reset", qubits_of(statement.split(None, 1)[1].strip()), []))
            continue
        m = _APPLY.match(statement)
        if not m:
            raise ConversionError(f"cannot parse {statement!r}")
        name, params = m.group(1), [_evaluate(e, {}) for e in _split_top(m.group(2) or "")]
        groups = [qubits_of(a) for a in _split_top(m.group(3))]
        width = max(len(g) for g in groups)
        if any(len(g) not in (1, width) for g in groups):
            raise ConversionError(f"register sizes differ in {statement!r}")
        for i in range(width):  # registers broadcast element-wise
            apply(name, params, [g[i] if len(g) > 1 else g[0] for g in groups])
    if not size:
        raise ConversionError("no qreg declared")
    return size, instructions


# ---------------------------------------------------------------- observables
_FACTOR = re.compile(r"([IXYZ])\s*_?\s*(\d+)")


def parse_observable(spec, n_qubits=None):
    """Observable terms [(coefficient, {wire: axis})] from many forms.

    Accepted: a string such as "0.5*Z0*Z1 - X3 + 2 Y2" (wire-indexed factors)
    or "0.5*IXZ" (a Qiskit-ordered label, rightmost character on wire 0);
    a Qiskit SparsePauliOp or Pauli; a mapping {label: coefficient}; or a list
    of (coefficient, {wire: axis}) pairs, which is returned as is.
    """
    if isinstance(spec, list) and all(isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[1], dict)
                                      for t in spec):
        return [(float(c), {int(q): a for q, a in p.items()}) for c, p in spec]
    if isinstance(spec, dict):
        return [(_real_coefficient(c), _label(label)) for label, c in spec.items()]
    to_list = getattr(spec, "to_list", None)
    if callable(to_list):  # SparsePauliOp
        return [(_real_coefficient(c), _label(label)) for label, c in to_list()]
    if type(spec).__name__ == "Pauli":
        label = spec.to_label()
        sign = {"": 1., "-": -1.}.get(label.rstrip("IXYZ"), None)
        if sign is None:
            raise ConversionError("Pauli with an imaginary phase is not Hermitian")
        return [(sign, _label(label.lstrip("-")))]
    if not isinstance(spec, str):
        raise ConversionError(f"cannot read an observable from {type(spec).__name__}")
    terms = []
    text = spec.replace(" ", "")
    # Split on signs, but not on the sign of an exponent such as 1e-3.
    chunks = [c for c in re.split(r"(?<![0-9.][eE])(?=[+-])", text) if c]
    for chunk in chunks:
        sign, body = (chunk[0], chunk[1:]) if chunk[0] in "+-" else ("", chunk)
        coefficient, factors = 1.0, body
        m = re.match(r"^((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\*?(.*)$", body)
        if m:
            coefficient, factors = float(m.group(1)), m.group(2)
        coefficient *= -1 if sign == "-" else 1
        factors = factors.strip("*")
        if not factors:
            terms.append((coefficient, {}))
            continue
        if re.fullmatch(r"[IXYZ]+", factors) and not _FACTOR.search(factors):
            terms.append((coefficient, _label(factors)))
            continue
        pauli, rest = {}, _FACTOR.sub("", factors).replace("*", "")
        if rest:
            raise ConversionError(f"cannot parse observable term {body!r}",
                                  'write terms like "0.5*Z0*Z1 - X3" or Qiskit labels like "0.5*IXZ"')
        for axis, wire in _FACTOR.findall(factors):
            q = int(wire)
            if q in pauli:
                raise ConversionError(f"wire {q} appears twice in {body!r}")
            if axis != "I":
                pauli[q] = axis
        terms.append((coefficient, pauli))
    if not terms:
        raise ConversionError("empty observable")
    if n_qubits is not None:
        for _, pauli in terms:
            if any(q >= n_qubits for q in pauli):
                raise ConversionError(f"observable acts on a wire beyond the {n_qubits} qubits")
    return terms


def readout_expand(terms, readout):
    """Apply classical readout error to observable terms, for any Pauli factor.

    Measuring X or Y means rotating into the Z basis and reading out, so a
    readout error (p01, p10) turns each factor P on a wire into
    alpha I + beta P, with alpha = p10 - p01 and beta = 1 - p01 - p10, the
    same algebra Quantum Circuit Simulator uses for Z. `readout` is a (p01, p10) pair for
    every wire or a mapping of wire to pair.
    """
    if not readout:
        return terms
    merged = {}
    for coefficient, pauli in terms:
        expanded = [(coefficient, {})]
        for q, axis in sorted(pauli.items()):
            pair = readout.get(q) if isinstance(readout, dict) else readout
            if not pair:
                expanded = [(c, {**p, q: axis}) for c, p in expanded]
                continue
            p01, p10 = pair
            alpha, beta = p10 - p01, 1 - p01 - p10
            expanded = [(c * alpha, p) for c, p in expanded if alpha] + [(c * beta, {**p, q: axis}) for c, p in expanded]
        for c, p in expanded:
            key = tuple(sorted(p.items()))
            merged[key] = merged.get(key, 0.) + c
    return [(c, dict(k)) for k, c in merged.items() if c != 0.]


def _without_readout(noise):
    """(noise without readout, readout) so readout can be applied to observables instead."""
    if noise is None or noise.readout is None:
        return noise, None
    return Noise(noise.depolarizing_1q, noise.depolarizing_2q, None), noise.readout


def _label(label):
    label = label.strip()
    if not re.fullmatch(r"[IXYZ]+", label):
        raise ConversionError(f"Pauli label {label!r} must use only I, X, Y and Z")
    n = len(label)
    return {n - 1 - i: a for i, a in enumerate(label) if a != "I"}


def _real_coefficient(c):
    c = complex(c)
    if abs(c.imag) > 1e-12 * max(1., abs(c.real)):
        raise ConversionError("observable coefficient has an imaginary part; the operator is not Hermitian")
    return c.real


# ---------------------------------------------------------------- one call
def expectation(circuit, observable, *, values=None, gradients=False, backend="native", truncation=None,
                noise=None, device=None, threads=1, max_terms=1_000_000):
    """One-call expectation value of an observable on a circuit.

    circuit may be a `Circuit`, a Qiskit QuantumCircuit or OpenQASM
    text; observable any form parse_observable accepts. values gives the
    Qiskit parameter values (or the shift vector for a `Circuit`).
    Returns an Evaluation; its gradient follows the converted parameters'
    order, which `convert(...).parameters` reports. A quantum_circuit_simulator.devices
    Device supplies the noise when `noise` is not given.
    """
    if device is not None and noise is None:
        noise = device.noise_model()
    noise, readout = _without_readout(Noise.coerce(noise))
    converted = convert(circuit, noise=noise, max_terms=max_terms, values=values)
    if device is not None and converted.circuit.n_qubits > device.n_qubits:
        raise ConversionError(f"the circuit has {converted.circuit.n_qubits} qubits; {device.name} has "
                              f"{device.n_qubits}", "choose a larger device")
    terms = readout_expand(parse_observable(observable, converted.circuit.n_qubits), readout)
    plan = converted.circuit.prepare(terms, backend=backend, truncation=truncation, threads=threads)
    shifts = converted.shifts(values) if converted.parameters else None
    return plan.evaluate(shifts, gradients=gradients)


def convert(circuit, *, noise=None, max_terms=1_000_000, values=None):
    """A Converted from a Circuit, a Qiskit circuit or OpenQASM text.

    For a Qiskit circuit, values are bound numerically only when its
    parameters cannot be shared.
    """
    if isinstance(circuit, Circuit):
        if noise is not None:
            raise ConversionError("noise applies to converted circuits; add channels to the Circuit directly")
        return Converted(circuit, list(range(circuit.n_parameters)))
    if isinstance(circuit, str):
        return from_qasm(circuit, noise=noise, max_terms=max_terms)
    if isinstance(circuit, Program):
        converted = _build(circuit.n_qubits, circuit.instructions, Noise.coerce(noise), max_terms)
        converted.source_order = list(circuit.parameters)
        return converted
    try:
        return from_qiskit(circuit, noise=noise, max_terms=max_terms)
    except ConversionError:
        if values is None:
            raise
        return from_qiskit(circuit, values=values, noise=noise, max_terms=max_terms)
