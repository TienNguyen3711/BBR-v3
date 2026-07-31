from __future__ import annotations

from typing import Any, Sequence

import pennylane as qml

from qbbr.agents.quantum.encoding import angle_encode

N_PARAMS_PER_LAYER = 3  # R_Z, R_Y, R_Z per qubit


def variational_block(
    params: Any, wires: Sequence[int], n_layers: int, reupload_state: Any | None = None
) -> None:
    n_qubits = len(wires)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])
        for i in range(n_qubits):
            qml.RZ(params[layer, i, 0], wires=wires[i])
            qml.RY(params[layer, i, 1], wires=wires[i])
            qml.RZ(params[layer, i, 2], wires=wires[i])
        if reupload_state is not None and layer < n_layers - 1:
            angle_encode(reupload_state, wires)
