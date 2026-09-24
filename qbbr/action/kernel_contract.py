"""Single-owner actuator semantics; phase timing is supplied by the caller."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math

GAIN_UNITS = {0.75: 192, 0.90: 230, 1.00: 256, 1.10: 282, 1.25: 320}
LEASE_S = 1.0
HEARTBEAT_S = 0.1


def quantize_request(gain: float | None) -> int:
    if gain is None or (not isinstance(gain, bool) and gain == -1):
        return -1
    if isinstance(gain, bool) or gain not in GAIN_UNITS:
        raise ValueError('Expected one of five declared gains or None/-1')
    return GAIN_UNITS[gain]


@dataclass(frozen=True)
class CommandState:
    requested: int = -1
    refreshed_s: float = -math.inf
    latched: int = -1
    was_cruise: bool = False
    applied: int = 256

    def write(self, gain: float | None, now: float):
        return replace(self, requested=quantize_request(gain), refreshed_s=now)

    def observe(self, now: float, *, cruise: bool, eligible: bool, native: int):
        """One ACK-equivalent observation; expiry and cancellation happen here."""
        latched = self.latched
        if self.requested < 0 or now >= self.refreshed_s + LEASE_S:
            latched = -1
        elif eligible and not self.was_cruise:
            latched = self.requested
        if not eligible:
            latched = -1
        return replace(self, latched=latched, was_cruise=cruise,
                       applied=latched if eligible and latched >= 0 else native)
