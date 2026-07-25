"""Scenario B (coexistence): agent runs concurrently with the four CCAs over one Starlink link.

Configured by configs/eval_scenarioB.yaml; fairness/efficiency measured via
qbbr.eval.metrics.alpha_fair_efficiency_ratio, swept over
alpha in {0, 1, 2, infinity} (throughput-max, proportional, ..., max-min).

Not yet implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_scenario_b(config_path: str | Path) -> Any:
    raise NotImplementedError("Scenario B runner not yet implemented.")
