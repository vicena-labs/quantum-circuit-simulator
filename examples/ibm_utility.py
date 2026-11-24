"""IBM's 127-qubit utility experiment at five steps, ideal and with IBM Kyiv's noise.

    python examples/ibm_utility.py

Kim et al., Nature 618, 500 (2023): per step RX(theta) on every qubit, then RZZ(-pi/2) on every
coupler of the heavy-hex chip. At five steps the circuit has been classically simulable since
2023; this reproduces it, it does not claim to be first. The noise is the device snapshot's
average gate errors and per-qubit readout errors, without idle T1/T2 decay. No truncation.
"""
import math
import time

from quantum_circuit_simulator import devices
from quantum_circuit_simulator.circuits import kicked_ising
from quantum_circuit_simulator.interop import expectation

kyiv = devices.get("ibm_kyiv")
print(f"{'theta':>7} {'ideal':>10} {'noisy':>10} {'seconds':>8}")
for k in range(9):
    theta = k * math.pi / 16
    qasm, magnetisation = kicked_ising(steps=5, theta_h=theta)
    start = time.perf_counter()
    ideal = expectation(qasm, magnetisation).value
    noisy = expectation(qasm, magnetisation, device=kyiv).value
    print(f"{theta:7.4f} {ideal:10.6f} {noisy:10.6f} {time.perf_counter() - start:8.2f}")
