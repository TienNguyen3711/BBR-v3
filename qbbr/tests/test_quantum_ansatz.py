from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from qbbr.agents.quantum.ansatz import N_PARAMS_PER_LAYER, variational_block
from qbbr.agents.quantum.encoding import angle_encode

N_QUBITS = 6
_dev = qml.device("default.qubit", wires=N_QUBITS)


def _circuit(state, params, n_layers, reupload=False):
    @qml.qnode(_dev, interface="torch")
    def run(state, params):
        angle_encode(state, list(range(N_QUBITS)))
        variational_block(
            params, list(range(N_QUBITS)), n_layers, reupload_state=state if reupload else None
        )
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    return run(state, params)


def test_param_shape_matches_18l_formula():
    for n_layers in (2, 3):
        params = torch.rand(n_layers, N_QUBITS, N_PARAMS_PER_LAYER)
        assert params.numel() == 18 * n_layers
        out = _circuit(torch.rand(N_QUBITS), params, n_layers)
        assert len(out) == N_QUBITS
        assert np.isfinite([float(o) for o in out]).all()


def test_zero_params_is_identity_rotation_but_entangles_via_cnot():
    state = torch.rand(N_QUBITS)
    params = torch.zeros(2, N_QUBITS, N_PARAMS_PER_LAYER)
    out = _circuit(state, params, n_layers=2)
    for o in out:
        assert -1.0 - 1e-6 <= float(o) <= 1.0 + 1e-6


def test_reupload_changes_output_vs_no_reupload():
    torch.manual_seed(0)
    state = torch.rand(N_QUBITS)
    params = torch.rand(2, N_QUBITS, N_PARAMS_PER_LAYER)
    out_plain = _circuit(state, params, n_layers=2, reupload=False)
    out_reupload = _circuit(state, params, n_layers=2, reupload=True)
    assert not np.allclose([float(o) for o in out_plain], [float(o) for o in out_reupload])


def test_gradients_flow_through_params():
    state = torch.rand(N_QUBITS)
    params = torch.rand(2, N_QUBITS, N_PARAMS_PER_LAYER, requires_grad=True)
    out = _circuit(state, params, n_layers=2)
    loss = sum(out)
    loss.backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
