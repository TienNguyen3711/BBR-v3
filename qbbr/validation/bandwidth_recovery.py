"""Fit a post-handover BBR delivery-bandwidth recovery time constant.

The fit intentionally needs observed handover events and delivery rate samples.
It refuses to invent a value from the fluid model; the output is the parameter
for ``FluidParams.bandwidth_estimate_recovery_s`` once field coverage exists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class BandwidthRecoveryFit:
    geographic_location: str
    direction: str
    events_used: int
    recovery_time_constant_s: float
    median_recovery_to_target_s: float
    target_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)


def fit_bandwidth_recovery(
    rows: Iterable[Mapping[str, object]],
    target_fraction: float = 0.9,
    post_handover_window_s: float = 10.0,
    min_events: int = 3,
) -> list[BandwidthRecoveryFit]:
    """Estimate per-location recovery from handover-labelled interval records.

    For each observed handover, the post-event 90th-percentile delivery rate
    is the local recovered reference. The first sample reaching
    ``target_fraction`` of that reference yields an exponential-equivalent
    time constant: ``tau = t / -log(1-target_fraction)``. This is a robust,
    auditable estimator rather than a claim about an unobserved BBR variable.
    """

    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be strictly between zero and one")
    if post_handover_window_s <= 0 or min_events < 1:
        raise ValueError("post_handover_window_s and min_events must be positive")
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        location = row.get("geographic_location")
        direction = row.get("direction")
        run_id = row.get("run_id")
        if location and direction and run_id:
            grouped[(str(location), str(direction), str(run_id))].append(row)

    recovery_by_scenario: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (location, direction, _run_id), samples in grouped.items():
        samples = sorted(samples, key=lambda row: float(row.get("offset_s", -1.0)))
        for event_index, event in enumerate(samples):
            if event.get("handover_event") is not True:
                continue
            event_time = float(event.get("offset_s", -1.0))
            post = [
                row for row in samples[event_index + 1 :]
                if 0.0 < float(row.get("offset_s", -1.0)) - event_time <= post_handover_window_s
                and row.get("delivery_rate_bps") is not None
            ]
            if len(post) < 2:
                continue
            rates = np.asarray([float(row["delivery_rate_bps"]) for row in post])
            reference = float(np.percentile(rates, 90))
            if reference <= 0.0:
                continue
            reached = next(
                (
                    float(row["offset_s"]) - event_time
                    for row in post
                    if float(row["delivery_rate_bps"]) >= target_fraction * reference
                ),
                None,
            )
            if reached is not None and reached > 0.0:
                recovery_by_scenario[(location, direction)].append(reached)

    fits = []
    for (location, direction), recovery_times in sorted(recovery_by_scenario.items()):
        if len(recovery_times) < min_events:
            continue
        median_time = float(np.median(recovery_times))
        fits.append(
            BandwidthRecoveryFit(
                geographic_location=location,
                direction=direction,
                events_used=len(recovery_times),
                recovery_time_constant_s=median_time / -np.log(1.0 - target_fraction),
                median_recovery_to_target_s=median_time,
                target_fraction=target_fraction,
            )
        )
    if not fits:
        raise ValueError(
            "No scenario has enough handover-labelled delivery-rate recoveries; "
            "do not enable bandwidth_estimate_recovery_s from unobserved data."
        )
    return fits
