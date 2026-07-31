from __future__ import annotations

import pennylane as qml

from qbbr.agents.quantum.ansatz import variational_block
from qbbr.agents.quantum.encoding import angle_encode

N_QUBITS_DEFAULT = 6
N_LAYERS_DEFAULT = 2
N_PARAMS_PER_LAYER = 3  # matches qbbr.agents.quantum.ansatz.N_PARAMS_PER_LAYER


def build_qnn(
    n_qubits: int = N_QUBITS_DEFAULT,
    n_layers: int = N_LAYERS_DEFAULT,
    reupload: bool = False,
    diff_method: str = "backprop",
) -> qml.qnn.TorchLayer:
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        angle_encode(inputs, wires)
        variational_block(weights, wires, n_layers, reupload_state=inputs if reupload else None)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": (n_layers, n_qubits, N_PARAMS_PER_LAYER)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
