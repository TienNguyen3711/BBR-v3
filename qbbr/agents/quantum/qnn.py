from __future__ import annotations

import os
import warnings

import pennylane as qml

from qbbr.agents.quantum.ansatz import variational_block
from qbbr.agents.quantum.encoding import angle_encode

N_QUBITS_DEFAULT = 6
N_LAYERS_DEFAULT = 2
N_PARAMS_PER_LAYER = 3  # matches qbbr.agents.quantum.ansatz.N_PARAMS_PER_LAYER

_DEVICE_ENV = "QBBR_QUANTUM_DEVICE"
_PREFERRED_DEVICE = "lightning.qubit"
_REFERENCE_DEVICE = "default.qubit"
_DIFF_METHOD_BY_DEVICE = {"lightning.qubit": "adjoint", "default.qubit": "backprop"}


def _resolve_device(n_qubits: int) -> tuple[qml.devices.Device, str]:
    requested = os.environ.get(_DEVICE_ENV, _PREFERRED_DEVICE)
    for name in (requested, _REFERENCE_DEVICE):
        try:
            return qml.device(name, wires=n_qubits), name
        except Exception as exc:  # device not installed / unusable
            if name == _REFERENCE_DEVICE:
                raise
            warnings.warn(
                f"{_DEVICE_ENV}={name!r} unavailable ({exc!r}); "
                f"falling back to {_REFERENCE_DEVICE!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
    raise RuntimeError("unreachable")


def build_qnn(
    n_qubits: int = N_QUBITS_DEFAULT,
    n_layers: int = N_LAYERS_DEFAULT,
    reupload: bool = False,
    diff_method: str | None = None,
) -> qml.qnn.TorchLayer:
    wires = list(range(n_qubits))
    dev, dev_name = _resolve_device(n_qubits)
    if diff_method is None:
        diff_method = _DIFF_METHOD_BY_DEVICE.get(dev_name, "backprop")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        angle_encode(inputs, wires)
        variational_block(weights, wires, n_layers, reupload_state=inputs if reupload else None)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": (n_layers, n_qubits, N_PARAMS_PER_LAYER)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
