from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
import torch

from qbbr.agents.quantum.encoding import angle_encode

N_QUBITS = 6
_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch")
def _readout(state):
    angle_encode(state, list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


def test_zero_state_leaves_all_qubits_at_plus_z():
    out = _readout(torch.zeros(N_QUBITS))
    assert np.allclose(out, 1.0, atol=1e-6)


def test_state_of_ones_flips_all_qubits_to_minus_z():
    # s_i = 1.0 -> RY(pi): a full flip from |0> to |1>, <Z> = -1.
    out = _readout(torch.ones(N_QUBITS))
    assert np.allclose(out, -1.0, atol=1e-6)


def test_half_state_is_an_equal_superposition():
    # s_i = 0.5 -> RY(pi/2): <Z> = cos(pi/2) = 0.
    out = _readout(torch.full((N_QUBITS,), 0.5))
    assert np.allclose(out, 0.0, atol=1e-6)


def test_each_qubit_only_depends_on_its_own_feature():
    state = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = _readout(state)
    assert out[0] == pytest.approx(-1.0, abs=1e-6)
    for i in range(1, N_QUBITS):
        assert out[i] == pytest.approx(1.0, abs=1e-6)
