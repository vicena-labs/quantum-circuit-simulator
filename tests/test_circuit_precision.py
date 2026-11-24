"""Generated circuits must carry exact angles: pi/2 printed to 12 digits is no
longer Clifford, and every coupler then branches into terms of size 5e-12."""
import math

import numpy as np

from quantum_circuit_simulator.circuits import kicked_ising
from quantum_circuit_simulator.interop import convert


def test_kicked_ising_couplers_stay_clifford():
    qasm, _ = kicked_ising(steps=1, theta_h=math.pi / 8)
    coupler = next(s for s in convert(qasm).circuit._impl.stages if s.name == "RZZ")
    assert coupler.angle == -math.pi / 2
    assert np.count_nonzero(np.abs(coupler.matrix) > 1e-14) == 16
