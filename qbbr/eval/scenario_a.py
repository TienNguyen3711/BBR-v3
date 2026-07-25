"""Scenario A (isolation): agent-augmented BBR vs. stock BBR-v3/Cubic/Vegas/Hybla, run individually.

Configured by configs/eval_scenarioA.yaml; metrics mirror the base TMC paper
(throughput, retransmissions, RTT variance) for direct comparability.

Not yet implemented -- depends on qbbr.env.testbed_env / qbbr.env.fluid_env
and qbbr.eval.stats.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_scenario_a(config_path: str | Path) -> Any:
    raise NotImplementedError("Scenario A runner not yet implemented.")
