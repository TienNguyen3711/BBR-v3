"""Inter-satellite-link failure probability (Eq. 15, TMC base paper).

p^isl = lambda_on / (lambda_on + lambda_off): a two-state (up/down) Poisson
availability model for the optical ISLs on a bent-pipe path. Requires the
ISL failure/recovery arrival rates, which Starlink does not expose through
any public API or telemetry available to this project.

Not yet implemented. qbbr.risk.ptot's "stub_constant"/"empirical_proxy"
modes stand in for this (and qbbr.risk.atmospheric, qbbr.risk.handover)
until real ISL telemetry is sourced.
"""
from __future__ import annotations


def compute_isl_failure_probability(lambda_on: float, lambda_off: float) -> float:
    """p^isl: steady-state probability a given ISL is in the disrupted state."""
    raise NotImplementedError(
        "Eq. 15 ISL availability model requires lambda_on/lambda_off arrival "
        "rates not exposed by Starlink's public API; see qbbr.risk.ptot for the interim proxy."
    )
