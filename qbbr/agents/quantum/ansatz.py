"""Stage 4 variational block: CNOT entangling ring + R_Z R_Y R_Z rotations, repeated L times.

6 x 3 x L = 18L parameters (36-54 for L in {2,3}); actor and critic share
this architecture with independent weights. Data re-uploading (re-encoding
s_t between variational layers via qbbr.agents.quantum.encoding) is the
planned capacity knob if L<=3 underfits, rather than widening the register
-- see main.tex Sec. "Stage 4: QNN Actor-Critic Core".

Not yet implemented.
"""
from __future__ import annotations

from typing import Any


def variational_block(
    params: Any, wires: Any, n_layers: int, reupload_state: Any | None = None
) -> None:
    """Apply n_layers repetitions of (CNOT ring) -> (RZ RY RZ per qubit)."""
    raise NotImplementedError("CNOT-ring + RZ-RY-RZ variational ansatz not yet implemented.")
