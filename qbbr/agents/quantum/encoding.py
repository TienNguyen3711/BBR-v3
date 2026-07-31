from __future__ import annotations

import math
from typing import Any, Sequence

import pennylane as qml


def angle_encode(state: Any, wires: Sequence[int]) -> None:
    for i, wire in enumerate(wires):
        qml.RY(math.pi * state[i], wires=wire)
