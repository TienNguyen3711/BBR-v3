"""PennyLane QNode/TorchLayer wiring: encoding -> ansatz -> <Z_i> readout.

Analytic backpropagation on the `default.qubit` simulator is the
training-time gradient path; the parameter-shift rule is retained only as
the hardware-compatible path (two circuit evaluations per parameter per
gradient) -- see main.tex Sec. "Gradients".

Not yet implemented -- depends on qbbr.agents.quantum.encoding and
qbbr.agents.quantum.ansatz.
"""
from __future__ import annotations

from typing import Any


def build_qnn(n_qubits: int = 6, n_layers: int = 2) -> Any:
    """Construct the PennyLane QNode mapping a 6-dim state to per-qubit <Z_i>."""
    raise NotImplementedError(
        "PennyLane QNN construction not yet implemented; see encoding.py/ansatz.py."
    )
