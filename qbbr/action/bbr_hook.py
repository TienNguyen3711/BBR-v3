"""Kernel hook applying the agent's chosen pacing_gain to a live BBR-v3 flow.

Per main.tex's "Action Space and Protocol Hook": overrides
bbr2_set_pacing_rate() at the start of each ProbeBW_CRUISE sub-state only,
leaving the probing gains of UP/DOWN/REFILL stock. Verified against the
IETF BBR draft in the design doc; extension to inflight_hi/lo and early
ProbeRTT triggering is deferred to later phases (configs/action_inflight.yaml).

Not yet implemented -- requires kernel-level access to the tested Starlink
terminal, reserved for qbbr.env.testbed_env.
"""
from __future__ import annotations


def set_pacing_gain(gain: float) -> None:
    """Override bbr2_set_pacing_rate() for the current ProbeBW_CRUISE interval."""
    raise NotImplementedError("bbr2_set_pacing_rate() hook not yet implemented.")


def in_probe_bw_cruise() -> bool:
    """True iff the kernel's BBR-v3 state machine is currently in ProbeBW_CRUISE."""
    raise NotImplementedError("ProbeBW_CRUISE sub-state detection not yet implemented.")
