"""Ablation grid over the design's free parameters.

Per main.tex's "Statistical Rigor and Ablations": alpha in {0.5, 1, 2},
(delta, beta) around (1.0, 0.5), L in {2, 3} (with and without data
re-uploading), risk features on/off, and quantum vs. classical cores.

Not yet implemented.
"""
from __future__ import annotations

from typing import Any, Iterable


def ablation_grid() -> Iterable[dict[str, Any]]:
    """Yield one config dict per point in the ablation grid described above."""
    raise NotImplementedError("Ablation grid not yet implemented.")
