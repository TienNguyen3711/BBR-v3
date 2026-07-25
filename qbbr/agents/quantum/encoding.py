"""Stage 3 angle encoding: R_Y(pi * s_i) on qubit i, one gate per state feature.

Six qubits jointly encode the 6-dim state s_t into |psi(s_t)> in C^64, on
which the qbbr.agents.quantum.ansatz entangling layers operate. Amplitude
encoding was rejected in the design (deep state-prep circuits, harder
gradients) -- see main.tex Sec. "Stage 3: Quantum Encoding".

Not yet implemented -- depends on the PennyLane device wired up in
qbbr.agents.quantum.qnn.
"""
from __future__ import annotations

from typing import Any


def angle_encode(state: Any, wires: Any) -> None:
    """Apply RY(pi * s_i) on each wire i for a 6-dim state vector s in [0,1]^6."""
    raise NotImplementedError("RY(pi*s_i) angle encoding not yet implemented.")
