"""BBR-v3 fluid-model dynamics engine (Eqs. 22-29, TMC base paper).

Integrates pacing rate, inflight volume, and the ProbeBW cruise/drawdown
indicators (I^dwn, I^crs) under an agent-substituted pacing_gain in place of
the stock 5/4--3/4 factors, coupled to the M/G/1 queue estimate
(qbbr.features.telemetry) and the risk process p_tot(t)
(qbbr.risk.ptot) driving loss events.

Not yet implemented -- this is the Phase 2 deliverable described in
main.tex's "Trace-Calibrated Fluid-Model Simulator": per-location parameters
come from qbbr.env.calibration, and traces are used only to calibrate and
validate this integrator, never replayed directly for training (naive
replay has zero counterfactual action coverage).
"""
from __future__ import annotations

from typing import Any


def step_fluid_state(state: dict[str, Any], pacing_gain: float, dt: float, params: dict[str, Any]) -> dict[str, Any]:
    """Advance the fluid-model state by dt seconds under the given pacing_gain."""
    raise NotImplementedError("Eqs. 22-29 fluid-model integrator not yet implemented.")
