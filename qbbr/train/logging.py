"""Per-run metric logging: config snapshot, seed, and rollout metrics.

Writes one directory per run (the configs/*.yaml used, plus per-episode
reward/throughput/retransmission summaries) so qbbr.eval.ablation sweeps
can be compared after the fact.

Not yet implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def start_run(config: dict[str, Any], out_dir: str | Path) -> Path:
    """Create a fresh run directory, snapshot `config` and the active seed into it."""
    raise NotImplementedError("Run logging not yet implemented.")


def log_episode(run_dir: str | Path, episode: int, metrics: dict[str, float]) -> None:
    """Append one episode's metrics to the run directory created by start_run."""
    raise NotImplementedError("Run logging not yet implemented.")
